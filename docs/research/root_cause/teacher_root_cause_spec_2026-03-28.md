# Teacher Value Network Root-Cause Spec

**Date:** 2026-03-28  
**Scope:** chốt lại các vấn đề đã được tìm thấy của teacher hiện tại, phân tích toán học của loss/objective, tổng hợp các thí nghiệm đã chạy, và xác định roadmap thực nghiệm tiếp theo để tìm hướng sửa đúng.

## 1. Mục tiêu và phạm vi kết luận

Mục tiêu của bản spec này không phải là đề xuất một fix duy nhất ngay lập tức, mà là:

1. Chốt rõ điều gì đã được **chứng minh bằng dữ liệu**.
2. Chốt rõ điều gì mới là **giả thuyết hợp lý nhưng chưa được chứng minh**.
3. Tách riêng các failure mode đang bị trộn lẫn.
4. Đề xuất các thực nghiệm tiếp theo sao cho mỗi thực nghiệm kiểm định được một giả thuyết cụ thể.

Mọi kết luận trong bản này dựa trên:

- code train hiện tại trong `C:\Users\USER\Downloads\train.ipynb`
- loss design note trong [loss_function_design.md](../../design/loss_function_design.md)
- run `stability_weighted_near_zero_finetune`
- run `oracle_root_cause_diagnostic`
- thống kê trực tiếp trên full train split `data/process/train`

## 2. Snapshot kỹ thuật hiện tại

### 2.1. Bài toán

- Mô hình là **value network** cho chess engine, không phải policy-value AlphaZero.
- Input là tensor `18x8x8`, STM-relative.
- Target gốc được nén từ centipawn thành:

```text
y = tanh(cp / 600)
```

- Kiến trúc train hiện tại là `architecture_v2`, head trả:

```text
z = f_theta(x)
p = tanh(z)
```

### 2.2. Objective source hiện tại

Notebook train source hiện tại định nghĩa:

```text
objective
= lambda_t * MSE(tanh(z), y)
+ (1 - lambda_t) * (1 - y^2)^beta * Huber(z - atanh(y_clamped))
```

với config source:

- `LOSS_MODE = "hybrid_curriculum"`
- `Y_LOSS_WEIGHT_START = 0.70`
- `Y_LOSS_WEIGHT_END = 1.00`
- `Y_LOSS_RAMP_EPOCHS = 8`
- `Z_LOSS_BETA = 1.0`
- `Z_HUBER_DELTA = 0.5`

Nhưng checkpoint production đang được dùng để phân tích là:
[ckpt_best.pt](/C:/Users/USER/Downloads/dgrn_5m_v3_stage2_polish_run1/ckpt_best.pt)

Config thực tế của checkpoint này là:

- `loss_mode = hybrid_curriculum`
- `y_loss_weight_start = 0.99`
- `y_loss_weight_end = 0.99`
- `y_loss_ramp_epochs = 0.0`
- `z_loss_beta = 1.0`
- `z_huber_delta = 0.5`

Nghĩa là run đang phân tích **không còn là curriculum thực sự**, mà gần như là:

```text
objective ≈ 0.99 * y-space loss + 0.01 * weighted z-space Huber
```

## 3. Evidence base

### 3.1. Full-data distribution

Từ full split `train`/`val`/`test` trong `data/process`:

- `|y| <= 0.05`: `31.43%`
- `|y| <= 0.10`: `43.00%`
- `|y| <= 0.20`: `55.00%`
- `|y| <= 0.50`: `73.20%`
- `|y| <= 0.70`: `86.00%`

Tức là phân phối target hiện tại rất center-heavy.

### 3.2. Baseline teacher metrics

Từ [baseline_test_metrics.json](../../../experiments/stability_weighted_near_zero_finetune/outputs/reports/baseline_test_metrics.json):

- overall `MSE = 0.06760`
- `mse_|y|<=0.7 = 0.05143`
- `slope_|y|<=0.7 = 0.60596`
- `mse_|y|<=0.2 = 0.03104`
- `R2_|y|<=0.2 = -4.076`
- `false_decisive P(|p|>=0.3 | |y|<=0.1) = 10.62%`
- `false_decisive P(|p|>=0.4 | |y|<=0.2) = 5.32%`
- `center_spread_ratio_0.05 = 8.76`
- worst mid-band bucket: `y≈0.651`, `p≈0.385`, gap `-0.266`

### 3.3. Stability-weighted fine-tune result

Từ [stability_weighted_test_compare.csv](../../../experiments/stability_weighted_near_zero_finetune/outputs/reports/stability_weighted_test_compare.csv):

- `slope_|y|<=0.7`: `0.606 -> 0.630`
- `max_midband_gap`: `0.266 -> 0.254`
- nhưng center safety xấu đi:
  - `mse_|y|<=0.2`: `0.0310 -> 0.0335`
  - `false_decisive_0.1_0.3`: `10.62% -> 11.64%`
  - `false_decisive_0.2_0.4`: `5.32% -> 5.92%`
  - `center_spread_ratio`: `8.76 -> 9.10`

Kết luận của thí nghiệm này:

- can thiệp `Stockfish instability proxy -> lookup weight -> fine-tune`
  **không sửa đúng problem**
- nó không chứng minh volatility sai
- nhưng chứng minh cách ánh xạ proxy thành weight trong run đó là quá thô

### 3.4. Oracle diagnostic result

Oracle run dùng Stockfish fixed-node trên subset stratified theo `|y|` và `teacher_abs_err`.

Các output chính:

- [oracle_root_cause_summary.json](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_root_cause_summary.json)
- [oracle_band_summary.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_band_summary.csv)
- [oracle_stability_summary.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_stability_summary.csv)
- [oracle_budget_alignment.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_budget_alignment.csv)
- [oracle_scale_sweep.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_scale_sweep.csv)
- [oracle_subset_rows.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_subset_rows.csv)

Các thống kê quan trọng:

- `teacher_closer_to_oracle_rate_600_overall = 21.67%`
- `teacher_closer_to_oracle_rate_600_near_zero = 14.14%`
- `train_vs_oracle_mae_600_near_zero = 0.0360`
- `teacher_vs_oracle_mae_600_near_zero = 0.1199`
- `stable_0.7_slope_600 = 0.5751`
- `corr_instability_vs_teacher_oracle_600 = -0.0087`
- `corr_instability_vs_train_oracle_600 = -0.0268`

## 4. Loss analysis: current objective đang làm gì

### 4.1. Geometry của pure y-space loss

Gọi:

```text
p = tanh(z)
a = atanh(y)
r = z - a
```

Loss y-space:

```text
L_y = (p - y)^2
```

Khai triển gần nghiệm `z = a`:

```text
tanh(z) ≈ y + (1 - y^2)(z - a)
```

Suy ra:

```text
L_y ≈ (1 - y^2)^2 (z - a)^2
    = (1 - y^2)^2 r^2
```

Tức là quanh nghiệm, `y-MSE` tương đương với một logit-MSE có trọng số:

```text
w_y(y) = (1 - y^2)^2
```

Điều này gây **gradient starvation** khi `|y|` lớn.

Ví dụ:

- `y = 0.5`  -> `w_y = 0.5625`
- `y = 0.7`  -> `w_y = 0.2601`
- `y = 0.8`  -> `w_y = 0.1296`
- `y = 0.9`  -> `w_y = 0.0361`

### 4.2. Geometry của z-term hiện tại

Nhánh z hiện tại trong notebook:

```text
L_z^w = (1 - y^2)^beta * Huber(z - a)
```

với `beta = 1`.

Trong quadratic regime của Huber:

```text
L_z^w ≈ (1 - y^2)^beta * r^2
```

Nghĩa là ngay cả nhánh z hiện tại cũng **vẫn giảm trọng số theo |y|**, chỉ là giảm chậm hơn y-space.

### 4.3. Stage2 checkpoint đang tối ưu cái gì

Với checkpoint hiện tại:

```text
lambda = 0.99
beta   = 1
```

local curvature gần nghiệm xấp xỉ:

```text
C(y) ∝ 0.99 * (1 - y^2)^2 + 0.01 * (1 - y^2)
```

Do đó tails vẫn bị downweighted rất mạnh.

### 4.4. Effective gradient mass trên full train split

Tôi đã tính trực tiếp theo bands `|y|`:

| Band | Sample share | Mean `(1-y^2)^2` | Effective y-loss mass |
|---|---:|---:|---:|
| `0..0.05` | `22.97%` | `0.998` | `31.77%` |
| `0.05..0.2` | `26.48%` | `0.973` | `35.69%` |
| `0.2..0.5` | `20.45%` | `0.772` | `21.87%` |
| `0.5..0.7` | `14.38%` | `0.401` | `7.99%` |
| `0.7..1.0` | `15.73%` | `0.123` | `2.69%` |

Với stage2 config gần-pure-y-space, effective hybrid mass gần như giống hệt.

### 4.5. Kết luận về loss

Loss hiện tại chủ yếu giải quyết một việc:

- **bù phần nào gradient starvation ở tails**

Loss hiện tại **không có cơ chế trực tiếp** để giải quyết:

- `ultra-center over-amplification`
- `false decisive near 0`
- `center reliability`

Đây là một kết luận cơ học, không phải chỉ là trực giác từ scatter.

## 5. Những vấn đề đã được xác minh chắc chắn

## 5.1. Failure A: mid-band magnitude compression

Đây là failure mode đã được chứng minh mạnh nhất.

Evidence:

- `stable_0.7_slope_600 = 0.575`
- teacher vẫn nén mạnh ở stable subset
- oracle band-wise amplitude ratio ở `0.2..0.7` đều dưới `1`
- bucket tables cho thấy gap lớn nhất ở vùng `0.5..0.7`

Từ oracle subset:

- band `0.2..0.5`: `mean|pred| / mean|oracle| ≈ 0.77`
- band `0.5..0.7`: `mean|pred| / mean|oracle| ≈ 0.73`

Stable subset còn xấu hơn.

Kết luận:

- model học sign khá tốt
- nhưng map magnitude sai trong vùng `|y| < 0.7`
- đây là vấn đề deterministic, phù hợp với hình học của loss hiện tại

## 5.2. Failure B: ultra-center over-amplification

Đây là failure mode riêng, đã được chứng minh là có thật.

Từ oracle subset `|oracle| <= 0.05`:

- overall:
  - `mean|oracle| = 0.0227`
  - `mean|pred| = 0.1327`
  - amplitude ratio `= 5.85x`
  - `P(|pred|>=0.1) = 55.9%`
  - `P(|pred|>=0.2) = 20.6%`
  - `P(|pred|>=0.3) = 5.9%`
- stable subset:
  - amplitude ratio `= 4.84x`
  - `P(|pred|>=0.1) = 41.7%`
  - `P(|pred|>=0.2) = 8.3%`

Kết luận:

- even when oracle says the position is truly ultra-center, teacher vẫn hay phát score quá mạnh
- failure này **không biến mất trên stable subset**

## 5.3. Loss hiện tại không giải quyết Failure B

Loss hiện tại center-heavy về mặt gradient. Nghĩa là:

- center không bị thiếu signal
- nhưng center vẫn hỏng

Do đó Failure B không thể được giải thích bằng câu chuyện “center bị underweighted”.

Nó là một failure mode khác với Failure A.

## 6. Những giả thuyết đã bị hạ ưu tiên hoặc bác bỏ

## 6.1. Label noise / oracle volatility không phải nguyên nhân chính

Oracle diagnostic cho thấy:

- `teacher_closer_to_oracle_rate_600_overall = 21.67%`
- `teacher_closer_to_oracle_rate_600_near_zero = 14.14%`

Nghĩa là phần lớn thời gian:

- **train label gần oracle hơn teacher**

Nếu label noise là culprit chính, teacher phải gần oracle hơn label thường xuyên hơn thế.

Thêm nữa:

- `corr_instability_vs_teacher_oracle_600 ≈ 0`
- `corr_instability_vs_train_oracle_600 ≈ 0`

Kết luận:

- volatility có thể tồn tại
- nhưng không phải driver chính của lỗi `|y| < 0.7` trong teacher hiện tại

## 6.2. “Shallow / same-depth labels sẽ tốt hơn” không được dữ liệu ủng hộ

Từ [oracle_budget_alignment.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_budget_alignment.csv):

`train_vs_oracle_budget_mae` giảm đều khi tăng node budget ở mọi band.

Nghĩa là current labels gần stronger oracle hơn weaker oracle.

Kết luận:

- hiện chưa có cơ sở để quay về “same depth nông hơn” như một fix chính

## 7. Những giả thuyết còn mở

## 7.1. Failure B do density skew + shortcut features

Đây là giả thuyết mạnh:

- `43%` dữ liệu có `|y| <= 0.1`
- teacher hỏng rất mạnh ở ultra-center
- loss hiện tại ưu tiên vùng center

Nhưng causal story kiểu:

- model học nhầm tactical/static shortcut
- rồi fire sai trên neutral positions

hiện vẫn là hypothesis, chưa được chứng minh trực tiếp.

## 7.2. Scale mismatch là secondary cause

Oracle scale sweep cho thấy:

- overall tốt nhất ở `800`
- stable overall tốt nhất ở `800`
- stable `<=0.2` tốt nhất ở `1200`
- stable `<=0.7` tốt nhất ở `1200`

Nhưng khi tách theo oracle band:

- `0..0.05` nghiêng về `600`
- `0.05..0.7` nghiêng về `1200`
- `>0.7` lại gần `600`

Kết luận:

- scale mismatch là tín hiệu thật
- nhưng không phải chỉ cần “đổi toàn bộ 600 thành 1200” là xong
- có thể tồn tại transform mismatch mang tính regime-dependent

## 8. Synthesis: root causes hiện tại

Từ toàn bộ evidence, hệ thống hiện có hai failure mode riêng:

1. **Failure A: magnitude compression ở mid-band**
   - đã được chứng minh
   - nguyên nhân mạnh nhất hiện tại là loss geometry + current polish objective

2. **Failure B: over-amplification ở ultra-center**
   - đã được chứng minh
   - chưa chứng minh xong nguyên nhân sâu nhất
   - loss hiện tại không nhắm trúng failure này

Từ đây, các ablation tiếp theo phải tách rõ:

- run nào nhằm sửa A
- run nào nhằm sửa B

Nếu không, rất dễ lặp lại pattern:

- cứu mid-band
- nhưng làm center xấu hơn

## 9. Roadmap thực nghiệm tiếp theo

## 9.1. Nguyên tắc thiết kế

Không thay nhiều biến cùng lúc.

Mỗi run phải kiểm định một hypothesis cụ thể:

- **A1**: Failure A có chủ yếu do curvature của loss không
- **A2**: density skew có góp phần lớn không
- **C1/C2**: scale mismatch có phải secondary cause quan trọng không
- **B1**: có cần regularization riêng cho center không

### Metric gate bắt buộc

Không dùng MSE tổng làm tiêu chí chính.

Các metric quyết định:

- `stable_0.7_slope`
- oracle MAE ở bands:
  - `0.05..0.2`
  - `0.2..0.5`
  - `0.5..0.7`
- `mean|pred| / mean|oracle|` trên `|oracle|<=0.05`
- `P(|pred|>=0.1 | |oracle|<=0.05)`
- `P(|pred|>=0.2 | |oracle|<=0.05)`
- `teacher_closer_to_oracle_rate`

## 9.2. Run A1 — Curvature-compensated loss

### Hypothesis

Failure A chủ yếu do `y-MSE + tanh` làm tails và mid-band bị starvation trong logit-space.

### Change

Giữ target `600`, fine-tune ngắn từ checkpoint tốt nhất với:

```text
L = alpha * weighted_y_mse + (1 - alpha) * huber(z - atanh(y))
```

trong đó:

```text
weighted_y_mse = w(y) * (tanh(z) - y)^2
w(y) = clip(1 / ((1 - y^2)^2 + eps), w_min, w_max)
```

Khuyến nghị:

- `alpha = 0.65`
- `w_min = 1.0`
- `w_max = 4.0`
- normalize `w` về mean `1`
- z-term dùng `beta = 0`
- `delta = 1.0`

### Tại sao đúng

Đây là preconditioning trực tiếp cho local curvature của y-loss.

Nó nhắm đúng failure A bằng toán học, thay vì sửa symptom bằng heuristic.

### Acceptance criteria

- `stable_0.7_slope` tăng rõ
- oracle MAE ở `0.2..0.7` giảm
- center metrics không xấu đi mạnh

## 9.3. Run A2 — Band-balanced sampling

### Hypothesis

Density skew của data đang góp phần đáng kể vào cả A và B.

### Change

Giữ loss baseline, chỉ đổi sampler theo bands:

- `[0,0.05]`
- `(0.05,0.2]`
- `(0.2,0.5]`
- `(0.5,0.7]`
- `(0.7,1.0]`

### Tại sao đúng

Đây là test sạch nhất cho câu hỏi:

- problem nằm ở geometry của loss
- hay nằm ở empirical risk bị dominated bởi center-heavy distribution

### Acceptance criteria

- nếu A2 giúp gần bằng A1, density skew là factor lớn
- nếu A1 thắng rõ A2, geometry của loss là factor lớn hơn

## 9.4. Run C1/C2 — Scale bracketing

### Hypothesis

Target transform `tanh(cp/600)` đang quá gắt với mapping mà model/loss hiện tại học được.

### Change

Không regenerate data, chỉ remap target từ `y_600`:

```text
y_c = tanh((600 / c) * atanh(clamp(y_600)))
```

Chạy hai run:

- `c = 800`
- `c = 1200`

### Tại sao đúng

Oracle sweep cho thấy signal nghiêng về:

- `800` cho overall
- `1200` cho stable non-tail

`800` và `1200` là hai điểm chẩn đoán tốt hơn `900`.

### Acceptance criteria

- mid-band oracle MAE giảm
- `stable_0.7_slope` tăng
- center over-amplification không tăng mạnh

## 9.5. Run B1 — Center confidence penalty

### Hypothesis

Failure B cần một regularizer riêng; loss hiện tại không encode “false decisive near 0 là nguy hiểm”.

### Change

Chỉ chạy sau khi có winner từ A1/A2/C1/C2.

Thêm penalty:

```text
L_center = 1[|y|<=tau] * ReLU(|p| - m)^2
```

Khuyến nghị:

- `tau = 0.05 .. 0.08`
- `m = 0.10`

### Tại sao đúng

Penalty này nhắm trực tiếp vào symptom của Failure B:

- pred quá quyết liệt ở ultra-center

Nó sạch hơn dạng margin tỉ lệ với `|target|`, vì khi `target≈0` margin kiểu `2*|target|` quá nhạy.

### Acceptance criteria

- amplitude ratio tại `|oracle|<=0.05` giảm mạnh
- `P(|pred|>=0.1 | |oracle|<=0.05)` giảm
- không làm `stable_0.7_slope` sụt mạnh

## 10. Decision tree sau các run

### Nếu A1 thắng rõ

Kết luận:

- geometry của current loss là culprit lớn nhất cho Failure A

Bước tiếp theo:

- merge A1 vào baseline
- rồi chạy B1 trên top of A1 nếu center vẫn hỏng

### Nếu A2 thắng rõ

Kết luận:

- distribution skew là factor lớn

Bước tiếp theo:

- ưu tiên sampler/batch construction trước
- sau đó mới tối ưu loss geometry

### Nếu C1/C2 thắng rõ

Kết luận:

- transform mismatch là factor mạnh

Bước tiếp theo:

- chuyển sang target scale mềm hơn
- hoặc thiết kế monotone transform mới

### Nếu không run nào thắng rõ

Kết luận:

- problem có thể nằm sâu hơn ở head calibration / representation

Bước tiếp theo:

- khi đó mới cân nhắc:
  - utility target
  - learned monotone transform
  - center-specific auxiliary head
  - architecture/head change

## 11. Những việc không nên làm ngay

- re-encode toàn bộ input
- relabel toàn bộ dataset
- quay về same-depth/shallow labels
- đổi backbone lớn

Lý do:

- oracle evidence hiện không đẩy mạnh sang các hướng đó
- đây là các thay đổi lớn nhưng ít sức chẩn đoán hơn loss/sampling/target-transform ablations

## 12. Chốt cuối cùng

Hiện tại có thể chốt một cách sạch như sau:

1. Teacher hiện tại đang có **hai failure mode riêng**:
   - mid-band magnitude compression
   - ultra-center over-amplification

2. Current loss/objective chỉ được thiết kế để giải bài toán:
   - tails under-confidence / z-space drift
   - chứ **không phải** center reliability

3. Với checkpoint `stage2_polish`, hybrid curriculum gần như đã bị triệt tiêu vì `lambda_y = 0.99`.

4. Label noise / volatility **không phải** root cause chính theo evidence hiện có.

5. Hướng đúng tiếp theo là:
   - `A1 curvature-compensated loss`
   - `A2 band-balanced sampling`
   - `C1/C2 scale bracketing`
   - rồi mới `B1 center penalty`

Đây là roadmap có cơ sở toán học rõ nhất và có power chẩn đoán tốt nhất từ toàn bộ dữ liệu hiện có.
