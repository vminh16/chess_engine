# Spec Evaluation Và Promotion Cho DGRN-X

Ngày: `2026-05-05`  
Trạng thái: `design proposal`  
Phạm vi: validation, benchmark, checkpoint selection, promotion rule cho toàn bộ dòng DGRN-X

---

## 1) Kết luận điều hành

Phase B và C1 đã chứng minh rằng:

- nếu eval/gate sai, ta sẽ promote nhầm,
- nếu oracle subset hoặc proxy hẹp lấn át broad metrics, ta sẽ đi sai hướng,
- nếu test bị dùng như selection set, kết luận sẽ không còn sạch.

Vì vậy spec eval là thành phần bắt buộc, không phải phụ lục.

---

## 2) Bằng chứng local làm nền cho spec eval

### 2.1 B1/B2 full-test đã bác bỏ gate cũ

Nguồn local:

- [`evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b1_sglobal_16b_256d_band_run1_best_gate/reports/core_metrics_table.csv`](../../evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b1_sglobal_16b_256d_band_run1_best_gate/reports/core_metrics_table.csv)
- [`evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1_best_gate/reports/core_metrics_table.csv`](../../evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1_best_gate/reports/core_metrics_table.csv)

B1/B2:

- tốt hơn trên oracle subset nào đó,
- nhưng fail broad core metrics.

### 2.2 C1 xác nhận broad validation phải là selector chính

Nguồn local:

- [`runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/decision_summary.json`](../../runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/decision_summary.json)
- [`runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/selected_checkpoint_eval.json`](../../runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/selected_checkpoint_eval.json)

C1:

- chọn theo broad validation,
- center tốt hơn nhiều,
- nhưng broad gate vẫn không pass.

Đây là hành vi selector đúng: không promote chỉ vì một proxy đẹp.

### 2.3 Bộ benchmark hiện tại đã có catalog độ tin cậy

Nguồn local:

- [`evaluation/phase_b_offline_benchmark/offline_benchmark_helpers.py`](../../evaluation/phase_b_offline_benchmark/offline_benchmark_helpers.py)

Repo hiện đã gắn `reliability` cho metric:

- `high`: core metrics như `overall_mse`, `overall_pearson`, band MSE, sign match, calibration gap.
- `medium`: oracle subset secondary.
- `low/medium`: diagnostic như `test_slope_0.1eq`, `oracle_center_score`.

Spec mới phải kế thừa tinh thần này.

---

## 3) Nguyên tắc eval chung

1. `Validation` để chọn checkpoint.
2. `Test` để đánh giá frozen candidate, không chọn checkpoint.
3. `Oracle subset` chỉ là secondary continuity signal.
4. `Search screening` chỉ chạy sau khi offline quality pass.
5. `Promotion` phải đồng thời qua gate offline và search nếu phase đã có search dependency.

---

## 4) Bốn tầng đánh giá

### 4.1 Tầng 0: Sanity checks

Chạy ngay sau encode/model wiring:

- no NaN/Inf,
- output distribution không collapse,
- target sign/perspective đúng,
- plane statistics hợp lý,
- throughput không bất thường.

### 4.2 Tầng 1: Broad validation core

Chạy mỗi epoch hoặc chu kỳ gần tương đương.

Phải dùng:

- split validation,
- kích thước đủ lớn để giảm noise selection,
- cùng metric family với offline benchmark chính.

### 4.3 Tầng 2: Offline full-test

Chạy trên frozen candidate, không dùng để chọn epoch.

Khuyến nghị tiếp tục dùng:

- `500000` sample,
- `10` shard,
- benchmark suite đang có.

### 4.4 Tầng 3: Search screening

Chỉ chạy nếu tầng offline pass.

Ít nhất gồm:

- fixed-nodes short match,
- fixed-time short match,
- optional SPRT/larger run nếu infrastructure có.

---

## 5) Metric tiers

### 5.0 Quy ước alias metric

Repo hiện có hai family tên metric gần nhau:

- family `validation/test-facing` trong run report, ví dụ:
  - `test_center_false_0.1eq`
  - `test_center_false_0.2eq`
  - `test_max_midband_abs_cal_gap`
- family `offline benchmark` trong bảng core, ví dụ:
  - `center_false_decisive_0.1eq`
  - `center_false_decisive_0.2eq`
  - `abs_cal_gap_0.2_0.5eq`
  - `abs_cal_gap_0.5_0.7eq`

Contract của spec này là:

- selection trên validation được phép dùng family `test_*`,
- promotion ở offline benchmark phải map về family `core_metrics_table.csv`,
- implementation không được coi đây là metric khác nghĩa nếu semantic giống nhau.

### 5.1 Primary promotion metrics

Các metric sau là `primary`:

- `overall_mse`
- `overall_mae`
- `overall_pearson`
- `test_mse_0.1eq`
- `test_mse_0.2eq`
- `test_mse_0.5eq`
- `test_mse_0.7eq`
- `center_false_decisive_0.1eq`
- `center_false_decisive_0.2eq`
- `sign_match_0.05_0.2eq`
- `sign_match_0.2_0.5eq`
- `sign_match_0.5_0.7eq`
- `abs_cal_gap_0.2_0.5eq`
- `abs_cal_gap_0.5_0.7eq`

### 5.2 Secondary metrics

- `oracle_midband_mae_sum_stable`
- `oracle_stable_0.7_slope`

Chỉ dùng để continuity với Phase B/C1, không làm gate chính.

### 5.3 Diagnostic metrics

- `test_slope_0.1eq`
- `center_spread_ratio`
- `max_midband_abs_cal_gap`
- `oracle_center_score`
- scatter / hexbin inspection

Các metric này hữu ích để chẩn đoán failure mode, nhưng không được dùng đơn lẻ để promote.

---

## 6) Checkpoint selection

### 6.1 Selector chính

Checkpoint selector chính là `broad validation score` trên validation set.

Score nên là weighted composite trên ratio hoặc normalized delta so với reference strong baseline.

### 6.2 Hard gate

Ngoài score tổng hợp, phải có hard gate tối thiểu trên:

- `overall_mse`
- `test_mse_0.1eq`
- `center_false_decisive_0.1eq`
- `abs_cal_gap`

Không có hard gate thì composite score có thể che giấu regression nghiêm trọng ở một trục quan trọng.

### 6.3 Test set discipline

Test chỉ được chạy trên:

- `best_gate` nếu có,
- nếu không có thì `best_any` frozen candidate.

Không re-select epoch bằng test rồi gọi đó là evaluation.

---

## 7) Calibration policy

### 7.1 Calibration trong eval

Phải log:

- band slopes,
- absolute calibration gaps,
- mean absolute prediction vs target theo band,
- center spread.

### 7.2 Post-hoc calibration

Nếu model đã tốt về rank/fit nhưng còn lệch calibration nhẹ, có thể cho phép post-hoc calibration:

- temperature scaling,
- affine calibration trên held-out validation,
- hoặc isotonic-like phương pháp nếu thật sự cần.

Nhưng:

- post-hoc calibration không được che giấu model fit kém,
- calibration model phải được fit trên validation riêng, không dùng test.

Nguồn:

- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [MMCE](https://proceedings.mlr.press/v80/kumar18a.html)
- [Regression calibration study](https://proceedings.mlr.press/v202/dheur23a.html)

---

## 8) Stage-specific promotion rules

### 8.1 `v0`

Promote nếu:

- offline full-test pass,
- broad core metrics không regress so với reference,
- center safety tốt hơn hoặc ít nhất không xấu rõ.

### 8.2 `v1`

Promote nếu:

- `v0` metrics vẫn giữ,
- policy metrics tăng,
- joint training không làm hỏng value broad metrics.

### 8.3 `RL`

Promote nếu:

- offline full-test vẫn pass,
- search screening tốt hơn,
- không xuất hiện regression lớn ở center/calibration.

---

## 9) Kill criteria

Run phải dừng sớm nếu xuất hiện một trong các pattern sau:

1. validation score đẹp nhưng broad candidate regress trên primary metrics cốt lõi,
2. center safety tốt hơn nhưng overall fit tệ đi đều như C1,
3. oracle metrics tốt nhưng broad full-test lặp lại pattern B1/B2,
4. policy improve nhưng value broad gate hỏng,
5. RL gain nhỏ nhưng offline quality sụp.

---

## 10) Artifact contract

Mọi run DGRN-X phải lưu:

- run config,
- encode schema version,
- checkpoint policy,
- validation summary,
- full-test benchmark config,
- decision summary,
- metric reliability catalog nếu benchmark suite hỗ trợ.

Không promote từ artifact thiếu metadata trên.

---

## 11) Quyết định cuối cùng

Spec eval của DGRN-X chốt 3 điểm:

1. `broad validation selects`,
2. `offline full-test judges`,
3. `search screening confirms`.

Nếu thiếu một trong ba tầng này, pipeline promotion sẽ quay lại failure mode Phase B/C1.
