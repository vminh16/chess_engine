# Train V6: Phase C1 Broad Objective Refresh

Phase này thay thế hướng chạy tiếp B3/B4 theo spec cũ. Mục tiêu là kiểm tra một train recipe mới từ đầu, bám theo:

- [phase_c1_broad_objective_train_spec_vi_2026-04-28.md](C:/Users/USER/Desktop/chess_engine/docs/design/phase_c1_broad_objective_train_spec_vi_2026-04-28.md)

## Thành phần

- [broad_objective_train_helpers.py](C:/Users/USER/Desktop/chess_engine/train_v6_broad_objective/broad_objective_train_helpers.py)
  - train loop resume-safe kế thừa từ Report1
  - objective broad C1
  - random/natural sampling
  - tắt oracle aux gradient steering
  - broad-validation checkpoint selection
- [phase_c1_train_helpers.py](C:/Users/USER/Desktop/chess_engine/train_v6_broad_objective/phase_c1_train_helpers.py)
  - wrapper env-aware cho Colab/local
  - cấu hình model 16/256 + `SimplifiedGlobalHead`
- [build_phase_c1_notebook.py](C:/Users/USER/Desktop/chess_engine/train_v6_broad_objective/build_phase_c1_notebook.py)
  - tạo notebook Colab từ code cells ổn định
- [train_phase_c1_broad_objective.ipynb](C:/Users/USER/Desktop/chess_engine/train_v6_broad_objective/train_phase_c1_broad_objective.ipynb)
  - notebook launch train

## Run mặc định

- run name: `dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1`
- model: 16 blocks / 256 hidden / `SimplifiedGlobalHead`
- train: from scratch, không load B1/B2 checkpoint
- sampling: `random`
- oracle aux batch: `0`
- validation: 200000 samples / 4 shards mỗi epoch
- final selected eval trong train: 200000 samples / 4 shards
- offline promotion bắt buộc: 500000 samples / 10 shards

## Biến môi trường chính

- `CHESS_RUN_SUFFIX`
  - thêm suffix vào `run_name`
- `CHESS_EPOCHS_OVERRIDE`
  - ép số epoch, dùng cho smoke test
- `CHESS_STAGE_DATA_LOCAL`
  - trên Colab mặc định `1`, stage data sang `/content/chess_engine_data/process`
- `CHESS_DATA_ROOT`
  - override data root nếu không dùng default
- `CHESS_RUNS_ROOT`
  - override runs root nếu không dùng default

## Smoke test

Trên Colab hoặc local CUDA:

```python
%env CHESS_EPOCHS_OVERRIDE=1
%env CHESS_RUN_SUFFIX=smoke
```

Sau smoke test, kiểm tra:

- `runs/<run_name>/reports/runtime_check.json`
- `runs/<run_name>/reports/run_config.json`
- `runs/<run_name>/reports/history.csv`
- `runs/<run_name>/reports/decision_summary.json`
- `runs/<run_name>/reports/selected_checkpoint_eval.json`

## Offline full-test sau train

Nếu `decision_summary.json` có `best_gate_checkpoint`, dùng:

```powershell
python evaluation\phase_b_offline_benchmark\run_phase_b_offline_benchmark.py `
  --run-dir runs\dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1 `
  --checkpoint-policy best_gate
```

Nếu chưa có broad gate, dùng `--checkpoint-policy best_any` để lấy giá trị thông tin nhưng không promote.

## Ghi chú quan trọng

- Trong C1, `best_gate` nghĩa là broad-validation gate, không phải oracle gate Phase B.
- Oracle role bundle 78 rows vẫn được load để tương thích eval helper, nhưng không còn tham gia gradient objective.
- Không chạy B4 nếu C1 chưa pass offline full-test broad core metrics.
