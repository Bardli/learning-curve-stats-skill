---
name: learning-curve-stats
description: Use when analyzing machine learning training curves, validation curves, TensorBoard/W&B/CSV/JSONL metric histories, overfitting, convergence, instability, plateauing, best checkpoint selection, or run/seed comparisons. Provides statistical tools and interpretation rules for turning noisy learning curves into evidence an AI can reason about.
---

# Learning Curve Stats

Use this skill when a user wants an AI to understand training curves through statistics instead of only visual inspection.

## Core Rule

Prefer raw metric history over screenshots. Read scalar histories from TensorBoard, W&B, MLflow, CSV, JSONL, or logs when possible. If only an image is available, treat conclusions as approximate and say the values were estimated visually.

Normalize each metric before interpreting shape:

- For losses, error rates, perplexity, WER, CER: lower is better, so define `score = -metric`.
- For accuracy, Dice, IoU, F1, AUC, reward: higher is better, so define `score = metric`.
- Compute trend statistics on `score`, then translate back to the original metric in the report.

## Analysis Workflow

1. Load `(step, metric)` pairs, sort by step, remove duplicate steps, and flag NaN/Inf.
2. Identify whether each metric is lower-is-better or higher-is-better.
3. Smooth the curve, but keep the raw curve for spike and instability checks.
4. Compute first-order, second-order, smoothness, stability, plateau, and train-validation gap metrics.
5. Report concrete evidence: values, step ranges, best checkpoint, and uncertainty.
6. Compare runs only at matched training budgets or matched wall-clock budgets unless the user asks otherwise.

## Statistical Tools

### Smoothing

Use smoothing to estimate the underlying trend, not to hide failures.

- `EMA`: good default for streaming training curves. Report the span or alpha.
- `rolling_median`: robust against isolated spikes.
- `Savitzky-Golay`: useful when estimating slope and acceleration from evenly sampled curves.
- `LOWESS`: useful for noisy non-linear curves when `statsmodels` is available.

Interpretation:

- Heavy smoothing can hide divergence, spikes, and short overfitting windows.
- If the conclusion changes under reasonable smoothing settings, say the diagnosis is unstable.

### Slope

`slope = d(score) / d(step)`.

Use a regression line over a window instead of single-point differences when the curve is noisy.

Interpretation:

- `slope > 0`: the model is improving in that window.
- `slope ~= 0`: training has plateaued or the metric is too noisy to show progress.
- `slope < 0`: the metric is getting worse.
- Compare `early_slope`, `mid_slope`, and `late_slope` to see whether learning is slowing down.

Useful derived numbers:

- `improvement_per_1k_steps`: slope scaled to 1000 steps.
- `time_to_90pct_best`: steps needed to reach 90% of the total observed improvement.
- `remaining_gain`: recent improvement divided by total improvement.

### Acceleration

`acceleration = d2(score) / d(step)^2`, usually estimated from the smoothed curve.

Interpretation:

- `acceleration > 0`: improvement speed is increasing, often after warmup or a scheduler change.
- `acceleration ~= 0`: approximately linear progress or a flat plateau.
- `acceleration < 0`: improvement is slowing, often approaching convergence.
- For raw loss curves, remember the sign was flipped into `score = -loss`.

Use acceleration carefully. It is sensitive to noise and smoothing choice. Prefer reporting it by window, not per point.

### Smoothness And Noise

Compute smoothness on the raw curve and the residual against the smoothed curve.

- `residual_std = std(raw_score - smooth_score)`: noise around the trend.
- `roughness = mean(abs(diff(score, n=2)))`: average second-difference magnitude.
- `rolling_std`: local variability over a fixed window.
- `signal_to_noise = abs(total_trend_change) / residual_std`.

Interpretation:

- High `residual_std` or `roughness`: noisy optimization, too high learning rate, small batch size, RL/environment variance, unstable data pipeline, or noisy validation.
- Low roughness with poor final metric: stable but underfitting or learning rate too low.
- High roughness near scheduler changes may be acceptable if the validation trend still improves.

### Spikes And Instability

Detect spikes on raw values, not smoothed values.

- `spike_count`: number of points whose residual exceeds a robust threshold such as `3.5 * MAD`.
- `max_spike`: largest absolute residual.
- `sign_change_rate`: fraction of adjacent slope signs that change.
- `oscillation_score`: rolling standard deviation divided by absolute rolling slope.
- `max_drawdown`: largest drop from a previous best score.

Interpretation:

- Repeated spikes in train loss suggest optimization instability, mixed precision overflow, bad batches, or too high LR.
- Repeated spikes only in validation can mean small validation set, evaluation nondeterminism, or distribution heterogeneity.
- Large drawdown after a previous best checkpoint means checkpoint selection matters.

### Plateau And Convergence

Use recent-window slope and improvement thresholds.

- `plateau_detected`: recent slope is near zero and recent improvement is below a threshold.
- `plateau_start_step`: first step after which improvement remains small.
- `best_step`: step with best validation metric.
- `overtraining_steps`: final step minus `best_step`.

Interpretation:

- Plateau with small train-validation gap: model may be capacity-limited, data-limited, or LR too low.
- Plateau with large train-validation gap: likely overfitting.
- Plateau immediately after warmup or LR drop may indicate a scheduler or optimization bug.

### Train-Validation Gap

Align train and validation metrics by step before computing gaps.

- `gap = train_score - val_score` for higher-is-better score.
- For losses, after sign normalization, a growing positive gap still means train improves relative to validation.
- `gap_slope`: slope of the gap over time.
- `overfit_start_step`: point where train keeps improving while validation stops improving or worsens.

Interpretation:

- Gap stable and both improve: healthy training.
- Gap grows while validation worsens: overfitting.
- Train and validation both bad: underfitting, optimization issue, insufficient training, or bad data.
- Validation better than train can happen with dropout, augmentation, label smoothing, or easier validation data; do not call it impossible.

### Multi-Run Or Multi-Seed Comparison

Use identical metric definitions and matched step budgets.

- `mean_curve`: average score across seeds at each step.
- `std_curve`: seed-to-seed variability.
- `confidence_interval`: uncertainty around the mean.
- `rank_stability`: whether the best method stays best across seeds.
- `best_at_budget`: best run at a fixed step or time budget.

Interpretation:

- A method that wins by less than seed variance is not a clear improvement.
- Faster early slope is useful only if final or budget-matched validation performance is also acceptable.
- Report both final metric and best validation metric when overfitting is present.

## Report Template

When reporting a learning-curve diagnosis, include:

- Data source and metric direction.
- Smoothing method and parameters.
- Best validation metric and `best_step`.
- Recent slope and whether a plateau was detected.
- Roughness/noise and spike summary.
- Train-validation gap and overfitting status.
- If comparing runs: matched-budget result and seed variability.
- Concrete next action: adjust LR, scheduler, batch size, regularization, data, checkpoint, or training length.

## Common Diagnoses

- `healthy_convergence`: train and validation improve, slope decreases smoothly, gap stays stable.
- `overfitting`: train improves while validation worsens or gap grows persistently.
- `underfitting`: train and validation both poor, both improve slowly, gap small.
- `unstable_optimization`: high roughness, frequent spikes, sign changes, or drawdowns.
- `lr_too_high`: noisy or divergent training, spikes after LR changes, validation drawdown.
- `lr_too_low`: very small slope, low roughness, slow improvement.
- `plateaued`: recent slope near zero and recent gain small.
- `needs_more_steps`: positive recent slope with no validation degradation.
- `checkpoint_sensitive`: best validation occurs much earlier than final step.

## Cautions

- Always state whether conclusions come from raw scalar data or image-only visual inspection.
- Do not compare curves with different x-axes unless converted to the same unit: step, token, epoch, sample, or wall-clock time.
- Do not let smoothing remove evidence of divergence.
- For irregular evaluation intervals, compute slopes with actual step differences.
- For metrics with bounded ranges such as accuracy or Dice, late acceleration naturally shrinks near the upper bound.
