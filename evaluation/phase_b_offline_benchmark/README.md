# Phase B Offline Benchmark

Folder này chứa suite benchmark offline riêng cho checkpoint champion của `Pha B`.

Mục tiêu:
- benchmark **toàn bộ `test set`** bằng đúng logic metric đã dùng trong pipeline `FT1 / Report1 / Phase B`,
- chạy được cả local và Colab,
- tách artifact benchmark khỏi `runs/` để không làm bẩn workspace nghiên cứu.

## Thành phần

- [offline_benchmark_helpers.py](C:/Users/USER/Desktop/chess_engine/evaluation/phase_b_offline_benchmark/offline_benchmark_helpers.py)
  - helper chính cho benchmark
  - resolve checkpoint từ `run_dir + checkpoint_policy`
  - mặc định chạy full offline eval trên `test`
  - so sánh với baseline
  - export summary/report/plot
- [build_phase_b_offline_benchmark_notebook.py](C:/Users/USER/Desktop/chess_engine/evaluation/phase_b_offline_benchmark/build_phase_b_offline_benchmark_notebook.py)
  - build notebook benchmark sạch
- [run_phase_b_offline_benchmark.py](C:/Users/USER/Desktop/chess_engine/evaluation/phase_b_offline_benchmark/run_phase_b_offline_benchmark.py)
  - CLI entry-point để chạy benchmark không cần notebook
- [phase_b_offline_benchmark.ipynb](C:/Users/USER/Desktop/chess_engine/evaluation/phase_b_offline_benchmark/phase_b_offline_benchmark.ipynb)
  - notebook env-aware
  - Colab: mount Drive, stage riêng `data/process/test` sang disk local của runtime
  - local: dùng path repo hiện tại

## Metric policy

Suite này chia metric thành 3 tầng:

### Core metrics

Đây là các metric đủ mạnh để dùng làm benchmark offline chính, vì đều lấy từ full test split lớn:

- overall `MSE / MAE / Pearson`
- `test_mse_0.1 / 0.2 / 0.5 / 0.7`
- `test_slope_0.2 / 0.7`
- `center_false_decisive`
- `center_wrong_sign`
- `sign_match_0.05_0.2 / 0.2_0.5 / 0.5_0.7`
- `abs_cal_gap_0.2_0.5 / 0.5_0.7`

### Secondary metrics

Giữ lại để continuity với các suite cũ, nhưng chỉ là metric phụ vì sample nhỏ hơn:

- `oracle_midband_mae_sum_stable`
- `oracle_stable_0.7_slope`

### Diagnostic-only metrics

Các metric này vẫn hữu ích để đọc failure mode, nhưng không nên dùng làm gate chính:

- `test_slope_0.1`
  - near-zero slope rất nhạy vì phương sai target cực nhỏ
- `center_spread_ratio`
  - hữu ích để bắt over-amplification quanh center nhưng có thể phóng đại cảm giác regression
- `max_midband_abs_cal_gap`
  - worst-bucket metric, nhạy với một bucket xấu cục bộ
- `oracle_center_score`
  - dựa trên center bundle curated rất nhỏ, không đáng tin để promote model một mình

## Artifact layout

Mỗi lần chạy benchmark sẽ ghi vào:

- `evaluation/phase_b_offline_benchmark/outputs/<benchmark_name>/reports`
- `evaluation/phase_b_offline_benchmark/outputs/<benchmark_name>/plots`

Artifact chính:
- `benchmark_config.json`
- `runtime_check.json`
- `candidate_eval_summary.json`
- `reference_eval_summary.json`
- `sample_sizes.json`
- `metrics_table.csv`
- `core_metrics_table.csv`
- `secondary_metrics_table.csv`
- `diagnostic_metrics_table.csv`
- `metric_reliability_catalog.csv`
- `absolute_calibration_curve.csv`
- `band_diagnostics.csv`
- `decision_summary.json`
- `oracle_bootstrap_compare.csv` nếu bật bootstrap
- `core_metrics.png`
- `prediction_vs_target_hexbin.png`
- `absolute_calibration.png`
- `center_behavior.png`
- `sign_match_by_band.png`

## Notebook env vars

- `CHESS_BENCHMARK_RUN_NAME`
  - mặc định: `dgrn_5m_phaseb_b1_sglobal_16b_256d_band_run1`
- `CHESS_BENCHMARK_CHECKPOINT_POLICY`
  - một trong: `best_gate`, `best_any`, `best_pareto_a`, `best_pareto_b`, `latest`
  - mặc định: `best_gate`
- `CHESS_BENCHMARK_NAME`
  - override tên benchmark output
- `CHESS_BENCHMARK_CHECKPOINT`
  - nếu set thì benchmark explicit checkpoint này thay vì resolve từ run dir
- `CHESS_REFERENCE_CHECKPOINT`
  - override baseline checkpoint
- `CHESS_DATA_ROOT`
  - override `data/process`
- `CHESS_STAGE_TEST_LOCAL`
  - trên Colab mặc định `1`
- `CHESS_BENCHMARK_EVAL_BATCH_SIZE`
- `CHESS_BENCHMARK_MAX_SAMPLES`
- `CHESS_BENCHMARK_NUM_SHARDS`
- `CHESS_BENCHMARK_BOOTSTRAP`
- `CHESS_BENCHMARK_BOOTSTRAP_N`
- `CHESS_CENTER_STRONG_THRESHOLD`
- `CHESS_BENCHMARK_ABS_CAL_BINS`

## Khuyến nghị chạy

### Benchmark champion hiện tại của B2

```python
%env CHESS_BENCHMARK_RUN_NAME=dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1
%env CHESS_BENCHMARK_CHECKPOINT_POLICY=best_gate
%env CHESS_STAGE_TEST_LOCAL=1
```

### CLI tương đương

```powershell
python evaluation\phase_b_offline_benchmark\run_phase_b_offline_benchmark.py `
  --run-dir runs\dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1 `
  --checkpoint-policy best_gate
```

### Benchmark `latest` để so với `best_gate`

```python
%env CHESS_BENCHMARK_RUN_NAME=dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1
%env CHESS_BENCHMARK_CHECKPOINT_POLICY=latest
%env CHESS_STAGE_TEST_LOCAL=1
```

### Chạy nhanh bằng sampled test

```python
%env CHESS_BENCHMARK_MAX_SAMPLES=200000
%env CHESS_BENCHMARK_NUM_SHARDS=4
```

### Chạy mạnh để ra quyết định

Mặc định helper sẽ benchmark full `test` nếu bạn không override `MAX_SAMPLES / NUM_SHARDS`.

## Ghi chú kỹ thuật

- Suite này reuse evaluator chuẩn của `train_v2_TF1/ft1_colab_helpers.py` để tránh lệch công thức metric so với pipeline hiện tại.
- Colab chỉ stage `test split`, không stage toàn bộ dataset train.
- Oracle subset đang lấy từ `split=test`, nên benchmark offline này không cần copy `train/val` khi chỉ chạy `test` eval.
- Notebook và CLI đều hỗ trợ sampled run để smoke-check, nhưng sampled run không nên dùng để quyết định promote model.
