# Chess Engine Value Network - Experiment Journal & Technical Spec

**Document type:** experiment journal + working technical spec  
**Repo:** `C:\Users\USER\Desktop\chess_engine`  
**Last updated:** 2026-04-01

## 1. Problem statement and current model

This project is a **value regression network** for a chess engine. It is not an AlphaZero-style policy-value model.

Current core setup:

- Input tensor: `18 x 8 x 8`
- Perspective: `STM-relative`
- Canonical target:
  - `y = tanh(cp / 600)`
- Teacher baseline checkpoint:
  - `C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1\ckpt_best.pt`
- Current network family:
  - `architecture_v2`
- Head/output:
  - network predicts logit `z`
  - final value is `p = tanh(z)`

This journal documents the sequence of experiments that were run to answer two questions:

1. Why does the model under-estimate non-trivial advantages in the middle bands?
2. Why does the model become falsely decisive around truly neutral positions?

## 2. Current failure statements

### 2.1 Failure A: mid-band magnitude compression

Definition:

- In roughly `0.2 <= |y| <= 0.7`, the model often predicts the correct sign.
- But the predicted magnitude is too small.
- This is a calibration / effective-gradient problem, not primarily a sign problem.

Operational symptoms:

- `oracle_stable_0.7_slope` is well below `1.0`
- `oracle_midband_mae_sum_stable` stays high even when sign accuracy is decent

### 2.2 Failure B: ultra-center over-confidence

Definition:

- On truly neutral and stable positions near `0`, the model predicts `|p|` too large.
- This produces many false-decisive outputs in positions that should remain close to zero.

Operational symptoms:

- `oracle_center_amp_ratio` is far above `1.0`
- `oracle_center_false_0.1eq` and `oracle_center_false_0.2eq` are high
- pooled center metrics show the same pattern on a separate center bundle

### 2.3 Current mechanistic interpretation

The current evidence supports:

- `Failure A` is **objective-side**
- `Failure B` is a combination of:
  - `B1`: raw near-zero labels are not clean enough to be used as direct center supervision
  - `B2`: gradients from mid-band / tail interfere with center behavior in shared parameters

This is the current best-supported decomposition. It is stronger than a vague claim like "the dataset is bad" and more precise than "the head is miscalibrated."

## 3. Why MSE is not the main metric

Raw global MSE is not the right decision metric for this project.

Reason:

- The target is `tanh(cp / 600)`, so geometric distance in `y`-space is not aligned with the actual failure modes we care about.
- The model can look acceptable under global MSE while still:
  - compressing non-trivial advantages
  - being catastrophically over-confident near zero

For `p = tanh(z)`, the `y`-space squared loss is:

```text
L_y = (p - y)^2 = (tanh(z) - y)^2
```

Near the target logit `z* = atanh(y)`, the local geometry induces a curvature factor proportional to `(1 - y^2)^2`. That means large `|y|` regions receive weaker effective learning pressure under plain `y`-space regression. This is the main theoretical reason Failure A was suspected to be objective-side.

## 4. Primary evaluation metrics

The project relies on **oracle-based metrics** rather than a single global MSE.

### 4.1 Stable-oracle metrics for Failure A

- `oracle_stable_0.7_slope`
  - Linear slope of `pred ~ oracle` on the stable subset up to `|oracle| <= 0.7`
  - Interprets magnitude calibration
  - Higher is better, ideal near `1.0`

- `oracle_midband_mae_sum_stable`
  - Sum of stable-oracle MAE across the main non-center bands
  - Lower is better
  - This is the cleanest scalar summary for Failure A

- `oracle_band_mae_0.05_0.2_stable`, `oracle_band_mae_0.2_0.5_stable`, `oracle_band_mae_0.5_0.7_stable`
  - Used to localize where a method helps or hurts

### 4.2 Center metrics for Failure B

- `oracle_center_amp_ratio`
  - `mean(|pred|) / mean(|oracle|)` on stable near-zero oracle positions
  - Ideal is near `1.0`
  - Values far above `1.0` indicate center inflation

- `oracle_center_false_0.1eq`, `oracle_center_false_0.2eq`
  - Fraction of stable oracle-center positions with `|pred| >= 0.1` or `>= 0.2`
  - Lower is better

- `pooled_center_mae`, `pooled_center_amp_ratio`, `pooled_center_false_0.1eq`, `pooled_center_false_0.2eq`
  - Same family of checks, but evaluated on a pooled center bundle built in the Failure B suite
  - These are especially useful for cross-suite comparison

### 4.3 Composite scores

Two composite scores appear in the experiments:

- `selection_score_v2`
  - Used inside the objective resolution suite
- `failure_b_score`
  - Used inside the Failure B suite

Important caveat:

- These composite scores are useful for ranking inside a suite.
- They are **not** the final scientific argument by themselves.
- Primitive metrics remain the main evidence.

### 4.4 Dataset-size caveats

- The main oracle diagnostic subset has `n = 240` positions in the summary report.
- The pooled center bundle used in the Failure B suite has `n = 22`.

So:

- the center conclusions are directionally consistent and repeated across suites
- but the pooled center bundle is still small and should be interpreted carefully

## 5. Experiment chronology

### 5.1 Oracle root cause diagnostic

**Primary output:**  
`experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_root_cause_summary.json`

**Question**

- Are Failures A and B real under a fixed-node oracle, or are they mostly artifacts of noisy labels?

**Design**

- Build an oracle-evaluated subset with Stockfish fixed-node supervision
- Compare:
  - training label vs oracle
  - current checkpoint vs oracle
- Separate stable and unstable behavior where relevant

**Key outputs**

- `teacher_closer_to_oracle_rate_600_overall = 0.2167`
- `teacher_closer_to_oracle_rate_600_near_zero = 0.1414`
- `train_vs_oracle_mae_600_near_zero = 0.0360`
- `teacher_vs_oracle_mae_600_near_zero = 0.1199`
- `stable_0.7_slope_600 = 0.5751`

**Interpretation**

- Near zero, the current baseline checkpoint is much worse than the raw label relative to oracle.
- This directly rejects the naive claim that "the whole dataset is simply too bad" as the main explanation.
- It also confirms that center over-confidence is a real model failure, not just a data-wide noise story.
- `stable_0.7_slope_600 = 0.5751` is direct evidence for mid-band compression.

**Conclusion**

- Failure A is real.
- Failure B is real.
- The current baseline checkpoint is not simply a denoised estimator of the raw label around center.

### 5.2 Root cause ablation suite

**Primary outputs**

- `experiments/root_cause_ablation_suite/outputs/reports/compare_runs.csv`
- `experiments/root_cause_ablation_suite/outputs/reports/gradient_mass_summary.csv`

**Question**

- Is Failure A mainly caused by the objective geometry, by scale, or by sampling?

**Design**

- Compare:
  - `baseline`
  - `A1_curvature_compensated`
  - `A2_band_balanced`
  - `C1_scale800`
  - `C2_scale1200`

The key theoretical idea behind `A1` is to counter the `y`-space curvature bias by reweighting the effective `y` loss in a way that restores gradient mass toward larger `|y|`.

**Key outputs from `compare_runs.csv`**

- Baseline:
  - `oracle_stable_0.7_slope = 0.5751`
  - `oracle_midband_mae_sum_stable = 0.5988`
  - `oracle_center_amp_ratio = 5.8512`
  - `oracle_gate_score = 1.0421`

- `A1_curvature_compensated`:
  - `oracle_stable_0.7_slope = 0.6467`
  - `oracle_midband_mae_sum_stable = 0.5656`
  - `oracle_center_amp_ratio = 7.1506`
  - `delta_gate_vs_baseline = -0.0322`

- `A2_band_balanced`:
  - `oracle_stable_0.7_slope = 0.5700`
  - `oracle_midband_mae_sum_stable = 0.5915`
  - `oracle_center_amp_ratio = 5.5514`

- `C1_scale800`:
  - `oracle_stable_0.7_slope = 0.6010`
  - `oracle_midband_mae_sum_stable = 0.5881`
  - `oracle_center_amp_ratio = 6.1521`

- `C2_scale1200`:
  - `oracle_stable_0.7_slope = 0.6077`
  - `oracle_midband_mae_sum_stable = 0.5908`
  - `oracle_center_amp_ratio = 6.6808`

**Key outputs from `gradient_mass_summary.csv`**

Effective gradient mass share:

- `A2_band_balanced`
  - `[0.000,0.050]`: `0.4169`
  - `[0.050,0.200]`: `0.3045`
  - `[0.200,0.500]`: `0.1868`
  - `[0.500,0.700]`: `0.0686`
  - `[0.700,1.000]`: `0.0233`

- `A1_curvature_compensated`
  - `[0.000,0.050]`: `0.3299`
  - `[0.050,0.200]`: `0.2474`
  - `[0.200,0.500]`: `0.1910`
  - `[0.500,0.700]`: `0.1343`
  - `[0.700,1.000]`: `0.0973`

**Interpretation**

- `A1` gives the exact predicted qualitative shift:
  - less gradient mass at center
  - much more gradient mass in upper mid-band and tail
- This aligns with the improvement in slope and mid-band MAE.
- Scale-only changes (`C1`, `C2`) do not produce the same quality of improvement.
- `A2` helps some metrics, but it does not fix the geometry as directly as `A1`.

**Conclusion**

- Failure A is strongly supported as an **objective-side** problem.
- Tanh scale alone is not the root cause.
- Sampling and scale can move metrics, but they do not explain the main effect.

### 5.3 Objective resolution suite

**Primary outputs**

- `experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv`
- `experiments/objective_resolution_suite/outputs/reports/full_bootstrap_summary.csv`
- `experiments/objective_resolution_suite/outputs/reports/replicate_oracle_aggregate.csv`

**Question**

- Which objective family is the best checkpoint basis going forward?
- Can an A1-family objective repair Failure A without making Failure B unacceptable?

**Variants compared**

- `L0_control_hybrid`
- `L1_z_strong_hybrid`
- `L2_curvature_y_only`
- `L3_full_A1`
- `L4_A1_plus_A2`
- `S1_A1_center_w020_m010`

**Key outputs from `full_primary_metrics.csv`**

- `baseline`
  - `oracle_midband_mae_sum_stable = 0.5988`
  - `oracle_stable_0.7_slope = 0.5751`
  - `oracle_center_amp_ratio = 5.8512`
  - `selection_score_v2 = 1.5029`

- `L0_control_hybrid`
  - `oracle_midband_mae_sum_stable = 0.5903`
  - `oracle_stable_0.7_slope = 0.5824`
  - `oracle_center_amp_ratio = 5.6834`
  - `selection_score_v2 = 1.4720`
  - `dominates_baseline = True`

- `L1_z_strong_hybrid`
  - `oracle_midband_mae_sum_stable = 0.5887`
  - `oracle_stable_0.7_slope = 0.5880`
  - `oracle_center_amp_ratio = 5.7350`
  - `selection_score_v2 = 1.4784`

- `L3_full_A1`
  - `oracle_midband_mae_sum_stable = 0.5654`
  - `oracle_stable_0.7_slope = 0.6461`
  - `oracle_center_amp_ratio = 7.1469`
  - `selection_score_v2 = 1.6334`

- `L4_A1_plus_A2`
  - `oracle_midband_mae_sum_stable = 0.5710`
  - `oracle_stable_0.7_slope = 0.6179`
  - `oracle_center_amp_ratio = 6.3792`
  - `selection_score_v2 = 1.5514`

- `S1_A1_center_w020_m010`
  - `oracle_midband_mae_sum_stable = 0.5749`
  - `oracle_stable_0.7_slope = 0.6295`
  - `oracle_center_amp_ratio = 6.6094`
  - `selection_score_v2 = 1.5776`

**Key outputs from `full_bootstrap_summary.csv`**

Important bootstrap flags:

- `L4_A1_plus_A2`
  - `supports_improvement_midband_teacher_vs_oracle_mae_sum_stable = True`
  - `supports_improvement_stable_0.7_slope = True`
  - center improvement flags remain `False`

- `L3_full_A1`
  - `supports_improvement_stable_0.7_slope = True`
  - but `supports_improvement_midband_teacher_vs_oracle_mae_sum_stable = False` at the suite threshold

This matters because `L4` is the A-family checkpoint with the cleanest support for both main Failure A metrics together.

**Supporting outputs from `replicate_oracle_aggregate.csv`**

Replicate aggregate only includes `baseline`, `A1_curvature_compensated`, and `A2_band_balanced`, but it supports the same direction:

- `A1` mean stable slope: `0.8272` vs baseline `0.7618`
- `A1` mean stable midband MAE sum: `0.4551` vs baseline `0.4617`
- `A1` mean center amp ratio: `7.9726` vs baseline `6.5306`

So across replicated oracle subsets, `A1` again improves Failure A and worsens Failure B.

**Interpretation**

- A1-family objectives are the best repair discovered for Failure A.
- But they worsen Failure B.
- Naive center penalty (`S1`) does not solve Failure B and does not dominate `L4`.
- `L0` is the safest balanced checkpoint if one must use a model immediately.

**Conclusion**

- For immediate balanced use: `L0_control_hybrid` is the safest choice.
- For continuing to solve both A and B: `L4_A1_plus_A2` is the most reasonable base checkpoint.

### 5.4 Failure B resolution suite

**Primary outputs**

- `experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_primary_metrics.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_pooled_center_metrics.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/center_label_purity_summary.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/center_purity_lookup_report.json`
- `experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json`
- `experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_influence.csv`
- `experiments/failure_b_resolution_suite/outputs/replicates/l0_l1_controls/reports/replicate_oracle_aggregate.csv`

**Question**

- Is Failure B mainly caused by center-label impurity, by gradient interference, or by both?
- Do raw-center penalties solve it?

#### 5.4.1 Label purity evidence for B1

From `center_label_purity_summary.csv`:

- For `raw |y| <= 0.05` and oracle center threshold `0.05`
  - `raw_center_count = 96`
  - `oracle_center_clean_count = 22`
  - `precision = 0.21875`
  - `recall = 0.9545`

- For `raw |y| <= 0.10` and oracle center threshold `0.05`
  - `raw_center_count = 147`
  - `precision = 0.14966`
  - `recall = 1.0000`

From `center_purity_lookup_report.json`:

- `base_rate = 0.14966`
- `smoothed_clean_rate_max = 0.26657`

**Interpretation**

- Raw center thresholds have high recall but poor precision.
- They are good for **mining candidates**, not for direct supervision.
- The lookup/proxy signal improves ranking only slightly; it does not become a genuinely clean label source.

This is the main evidence for `B1`.

#### 5.4.2 Gradient interference evidence for B2

From `gradient_interference_cosines.csv`:

- Baseline:
  - cosine(`center_raw_0_005`, `mid_05_07`) = `-0.6423`
  - cosine(`near_center_005_02`, `mid_05_07`) = `-0.5549`

- A1 objective:
  - cosine(`center_raw_0_005`, `mid_05_07`) = `-0.6805`
  - cosine(`near_center_005_02`, `mid_05_07`) = `-0.6719`

From `gradient_interference_influence.csv`, one small step on mid-band gradients increases center decisiveness:

- Baseline:
  - `mid_02_05 -> center_clean_005`: `delta_mean_abs_pred = +0.00213`
  - `mid_05_07 -> center_clean_005`: `delta_mean_abs_pred = +0.00253`

- A1 objective:
  - `mid_02_05 -> center_clean_005`: `delta_mean_abs_pred = +0.00352`
  - `mid_05_07 -> center_clean_005`: `delta_mean_abs_pred = +0.00277`

From `gradient_interference_summary.json`:

- mean center-probe inflation caused by mid-band gradients:
  - baseline objective: `0.00233`
  - A1 objective: `0.00314`

**Interpretation**

- Center and upper mid-band gradients are anti-aligned in shared parameters.
- Under A1-style objectives, the interference is stronger.

This is the main evidence for `B2`.

#### 5.4.3 Pilot results

From `combined_failure_b_primary_metrics.csv`:

- Best current Failure B scores:
  - `A2_band_balanced = 0.6173`
  - `L0_control_hybrid = 0.6182`
  - `baseline = 0.6506`

- A-family / center-penalty / B-pilots:
  - `L4_A1_plus_A2 = 0.7330`
  - `S1_A1_center_w020_m010 = 0.7426`
  - `P_B2_raw_center_strong = 0.7625`
  - `L3_full_A1 = 0.8049`
  - `P_B1_proxy_center_weighted = 0.8559`

From `combined_failure_b_pooled_center_metrics.csv`:

- `A2` and `L0` both improve pooled center metrics relative to baseline
- `P_B1` and `P_B2` both remain too decisive near zero

**Interpretation**

- `P_B2_raw_center_strong` fails because raw-center penalties do not solve the impurity problem and do not remove interference.
- `P_B1_proxy_center_weighted` fails because the proxy purity ceiling is too low to become the main training signal.
- `S1` shows that naive center suppression from the raw labels is not enough, even during a full run.

**Conclusion**

- Failure B is not explained by a single cause.
- The best current reading is:
  - `B1` is real
  - `B2` is real
  - raw-center penalties are not a solution

### 5.5 OC1: late oracle-center correction pilot

**Primary outputs**

- `experiments/l4_oracle_center_correction_pilot/outputs/reports/combined_oracle_center_pilot_primary_metrics.csv`
- `experiments/l4_oracle_center_correction_pilot/outputs/reports/pilot_history.json`
- `experiments/l4_oracle_center_correction_pilot/outputs/cache/oracle_probe/oracle_candidate_report.json`

**Question**

- Can a short late-stage correction from `L4`, using an oracle center auxiliary set, repair Failure B?

**Design**

- Start from `L4`
- Build a small oracle-corrected center auxiliary bundle
- Fine-tune late

**Oracle mining result**

- `192` candidates
- `88` stable
- `35` center-clean
- `87` aux rows kept

**Authoritative evaluation result**

From `combined_oracle_center_pilot_primary_metrics.csv`:

- `L4 failure_b_score = 0.7330`
- `OC1 failure_b_score = 0.8916`

Center got worse:

- `pooled_center_mae`: `0.1090 -> 0.1283`
- `pooled_center_amp_ratio`: `4.799 -> 5.916`
- `pooled_center_false_0.1eq`: `0.5000 -> 0.5909`

**Important caution**

The internal `pilot_history.json` was later found to be unreliable for scientific conclusion because the epoch-end eval path was wrong. The authoritative result is the re-evaluated comparison CSV.

**Conclusion**

- `OC1` failed.
- It rejected that particular implementation of late oracle correction.
- It did **not** cleanly reject the whole idea yet, because `OC1` also had logic issues.

### 5.6 OC2: joint oracle full-model pilot

**Primary outputs**

- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_primary_metrics.csv`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_pooled_center_metrics.csv`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/oc2_pilot_history.json`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/decision_summary.json`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/cache/oracle_probe_2d/oracle_candidate_report.json`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/cache/oracle_role_bundle/manifest.json`

**Question**

- If the late-polish logic is fixed properly, does a short full-model joint correction still have a path to repair Failure B without breaking Failure A?

**Design improvements relative to OC1**

- Full model trainable, not just head + few blocks
- True joint objective with one optimizer step
- Proper epoch-end `eval()` behavior
- 2D candidate mining over:
  - `raw |y|`
  - `|pred_L4|`
- Split oracle bank into:
  - `center_anchor`
  - `center_hard`
  - `center_ambiguous`
- Hard midband gate plus center-only selection

**Oracle bank coverage**

From the cache manifests:

- `192` candidates in the 2D candidate bundle
- `93` stable
- `39` center-clean
- `78` aux rows kept
- Role counts:
  - `14 center_anchor`
  - `25 center_hard`
  - `39 center_ambiguous`

This means OC2 did not fail because the oracle bank was trivial or empty.

**Hardware / runtime**

From `joint_batch_autotune.json`:

- selected `main_batch_size = 384`
- estimated peak VRAM `2.205 GB` on a `4 GB` GPU

From `oc2_pilot_history.json`:

- `1` epoch
- epoch time about `981.1 s`

With `400k` main samples and `1042` steps/epoch, the aux bank was replayed heavily. This is relevant for interpretation, but it does not by itself invalidate the result.

**Result**

From `decision_summary.json`:

- `best_any_center_score = 0.6035`
- `best_gate_center_score = 0.6035`
- `l4_center_score = 0.5253`
- `has_gate_checkpoint = true`

So OC2 passed the midband gate, but still lost on center.

From `combined_oc2_final_pilot_primary_metrics.csv`:

Compared with `L4`:

- `oracle_midband_mae_sum_stable`: `0.5710 -> 0.5745`
- `oracle_stable_0.7_slope`: `0.6179 -> 0.6442`
- `oracle_band_mae_0.2_0.5_stable`: `0.1889 -> 0.1853`
- `oracle_band_mae_0.5_0.7_stable`: `0.2525 -> 0.2396`

But center worsened:

- `pooled_center_mae`: `0.1090 -> 0.1241`
- `pooled_center_amp_ratio`: `4.799 -> 5.430`
- `oracle_center_amp_ratio`: `6.379 -> 7.245`
- `oracle_center_false_0.2eq`: `0.2353 -> 0.3529`
- `pooled_center_wrong_sign_0.1`: `0.0909 -> 0.1364`

**Interpretation**

- OC2 is a better negative test than OC1.
- It fixes the main OC1 implementation problems.
- It keeps the A-side gains broadly intact.
- It still fails to improve B.

**Practical conclusion from OC2**

- Short late correction is not a promising primary path anymore.
- This does **not** prove mathematically that every short fine-tune is impossible.
- But it is enough practical evidence to de-prioritize late polish as the main direction.

**Residual caveat**

The model uses BatchNorm heavily. In a train-mode aux pass, a small oracle bank can also shift running statistics. That means OC2 is still not a perfect disproof of every possible short correction regime. But the practical result remains strongly negative.

## 6. What is proven, what is still a hypothesis

### 6.1 Strongly supported by the outputs

- Failure A is objective-side.
- Failure B is not explained by "global dataset badness."
- `B1` center-label impurity is real.
- `B2` gradient interference is real.
- A1-family objectives improve Failure A and worsen Failure B.
- Raw-center penalties are not a working solution to Failure B.
- Proxy-weighted raw-center penalties are not a working solution to Failure B.
- Late polish from `L4` is not a reliable path:
  - `OC1` failed
  - `OC2` also failed, despite much cleaner design

### 6.2 Still hypotheses, not yet proven

- The exact internal representation mechanism of Failure B inside the backbone is still a hypothesis, even if "shared-feature interference" is strongly suggested.
- A full-training run with clean oracle center anchors from epoch `0` has not yet been executed, so it remains the main open experimental hypothesis.
- Search compensation may help downstream playing strength, but it has not been quantified here and is not a substitute for fixing evaluator behavior.

## 7. Current checkpoint guidance

### 7.1 If a checkpoint is needed now

- Safest balanced choice:
  - `L0_control_hybrid`
- If the priority is center behavior more than A-side repair:
  - `A2_band_balanced` is also strong

### 7.2 If a checkpoint is needed as the base for future model-fix work

- Use:
  - `L4_A1_plus_A2`

Reason:

- It is the most reasonable current base if the goal is still to repair both A and B at the model level.
- It preserves the strongest credible A-side gains without yet changing architecture or encoding.

## 8. Recommended next direction

### 8.1 What should not be the main path

The outputs now strongly argue against prioritizing:

- another naive raw-center penalty
- another scale-only adjustment
- another late oracle-center polish as the main branch

### 8.2 The most plausible model-side direction now

The best-supported next direction is:

- keep the `L4`-style main objective
- intervene **from epoch 0**, not as late polish
- add clean oracle center supervision as a separate source of risk
- downweight raw ambiguous center labels in the main loss
- handle BatchNorm carefully so a small oracle bank does not corrupt running stats

A mathematically consistent form is:

```text
L_total
= E_raw [ w_raw(x) * L_L4(f(x), y_raw) ]
+ lambda_anchor * E_center_anchor [ Huber(f(x) - y_oracle) + lambda_margin * ReLU(|f(x)| - m)^2 ]
+ lambda_ambig * E_center_ambiguous [ Huber(f(x) - y_oracle) ]
```

where:

- `L_L4` keeps the best current solution for Failure A
- `w_raw(x)` downweights ambiguous raw-center labels
- `center_anchor` provides trusted center supervision
- `center_ambiguous` prevents collapsing every raw-center candidate to zero

### 8.3 Why this direction is the most reasonable one

The outputs point here for a coherent reason:

- Objective changes during full training clearly move Failure A.
- Late correction, even after fixing OC1's logic errors, still does not repair Failure B.
- Failure B is explained by both:
  - noisy center supervision
  - shared-parameter interference

That combination naturally suggests that the correction must be injected into the **global training dynamics**, not only at the end.

### 8.4 Practical implementation notes for the next run

The next serious run should keep constraints minimal:

- do not change architecture yet
- do not re-encode the whole dataset
- do not change the target scale yet

But it should add:

- full-training center-aware loss from epoch `0`
- trusted oracle center bank
- BN-safe handling on oracle batches
  - either freeze BN running stats on oracle-only passes
  - or ensure mixed batches are used so aux supervision does not distort statistics
- checkpoint selection by:
  - hard midband gate
  - then center score

## 9. Final state of the project

At the current point in the project:

- Failure A is understood well enough to design objectives for it.
- Failure B is understood well enough to reject several wrong solution classes.

The outputs are now pointing in one clear direction:

- Failure B is **not behaving like** a late calibration bug that can be fixed reliably by short polish.
- The best current evidence points to a full-training dynamics problem shaped by label impurity and gradient interference.
- Therefore, the most rational next experiment is a **center-aware full run that injects the fix from epoch 0** on top of the `L4` objective family, not another polish notebook.

## 10. Source files used for this spec

Main evidence files:

- `experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_root_cause_summary.json`
- `experiments/root_cause_ablation_suite/outputs/reports/compare_runs.csv`
- `experiments/root_cause_ablation_suite/outputs/reports/gradient_mass_summary.csv`
- `experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv`
- `experiments/objective_resolution_suite/outputs/reports/full_bootstrap_summary.csv`
- `experiments/objective_resolution_suite/outputs/reports/replicate_oracle_aggregate.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_primary_metrics.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_pooled_center_metrics.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/center_label_purity_summary.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/center_purity_lookup_report.json`
- `experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json`
- `experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv`
- `experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_influence.csv`
- `experiments/failure_b_resolution_suite/outputs/replicates/l0_l1_controls/reports/replicate_oracle_aggregate.csv`
- `experiments/l4_oracle_center_correction_pilot/outputs/reports/combined_oracle_center_pilot_primary_metrics.csv`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_primary_metrics.csv`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_pooled_center_metrics.csv`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/oc2_pilot_history.json`
- `experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/decision_summary.json`
