# Experiment Suites

This folder contains the diagnostic and ablation suites that drove the current training strategy.

## Main suites

- `oracle_root_cause_diagnostic`: oracle-based benchmark and stable subset extraction
- `root_cause_ablation_suite`: isolates objective, scale, and sampling effects
- `objective_resolution_suite`: compares the L-family objectives and selects the best base checkpoint
- `failure_b_resolution_suite`: tests hypotheses for ultra-center over-confidence
- `l4_oracle_center_correction_pilot`: late-stage oracle-center correction pilot
- `oc2_joint_oracle_full_model_pilot`: cleaner full-model short pilot after OC1
- `stability_weighted_near_zero_finetune`: early near-zero weighting experiment
- `teacher_root_cause_lab`: exploratory lab notebook for early teacher diagnostics

Each suite directory contains a notebook, helper module, and `outputs/` folder with reports and cached artifacts.
