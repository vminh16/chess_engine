# Teacher Value Network Root-Cause Spec

**Date:** 2026-03-31  
**Scope:** chốt lại toàn bộ kết luận hiện tại về teacher value network, dựa trên các run đã hoàn tất tới `objective_resolution_suite`, với mục tiêu xác định:

1. failure mode nào đã được chứng minh bằng dữ liệu;
2. failure mode nào mới chỉ là giả thuyết;
3. đâu là root cause chính của chất lượng kém trong vùng `|y| < 0.7`;
4. hướng giải quyết nào còn đáng làm tiếp.

## 1. Bối cảnh bài toán

- Bài toán là **value regression** cho chess engine, không phải policy-value network.
- Input là tensor `18x8x8`, STM-relative.
- Target train hiện tại là:

```text
y_600 = tanh(cp / 600)
```

- Model hiện tại là `architecture_v2` với output head:

```text
z = f_theta(x)
p = tanh(z)
```

- Checkpoint teacher đang được dùng làm mốc baseline là:
  [ckpt_best.pt](/C:/Users/USER/Downloads/dgrn_5m_v3_stage2_polish_run1/ckpt_best.pt)

## 2. Objective thực tế của checkpoint hiện tại

Notebook source từng có hybrid curriculum, nhưng checkpoint `stage2_polish` đang phân tích có config thực tế gần như:

```text
objective
= 0.99 * MSE(tanh(z), y)
+ 0.01 * (1 - y^2)^beta * Huber(z - atanh(y_clamped))
```

với:

- `lambda_y = 0.99`
- `beta = 1.0`
- `delta = 0.5`

Nghĩa là run này về thực chất là **near-pure y-space polish**, không còn là curriculum thật sự.

## 3. Cơ sở toán học của failure mode hiện tại

Đặt:

```text
p = tanh(z)
a = atanh(y)
r = z - a
```

Với y-space loss:

```text
L_y = (p - y)^2
```

khai triển quanh nghiệm `z = a`:

```text
tanh(z) ≈ y + (1 - y^2)(z - a)
```

suy ra:

```text
L_y ≈ (1 - y^2)^2 (z - a)^2
    = (1 - y^2)^2 r^2
```

Kết luận:

- trong logit-space, `y-MSE` tương đương với một quadratic loss có trọng số `w(y) = (1 - y^2)^2`;
- khi `|y|` tăng, local curvature giảm rất mạnh;
- do đó objective hiện tại sẽ **đói gradient** ở các regime có `|y|` trung bình và lớn.

Ví dụ:

- `y = 0.5  -> (1-y^2)^2 = 0.5625`
- `y = 0.7  -> 0.2601`
- `y = 0.8  -> 0.1296`
- `y = 0.9  -> 0.0361`

Nhánh z hiện tại cũng không đảo ngược được điều này, vì với `beta = 1` thì z-term vẫn bị giảm trọng số theo `|y|`.

## 4. Evidence base đã dùng

### 4.1. Oracle và ablation artifacts chính

- [oracle_root_cause_summary.json](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_root_cause_summary.json)
- [oracle_band_summary.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_band_summary.csv)
- [oracle_scale_sweep.csv](../../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_scale_sweep.csv)
- [full_primary_metrics.csv](../../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv)
- [full_bootstrap_summary.csv](../../../experiments/objective_resolution_suite/outputs/reports/full_bootstrap_summary.csv)
- [replicate_oracle_aggregate.csv](../../../experiments/objective_resolution_suite/outputs/reports/replicate_oracle_aggregate.csv)
- [final_replicates/replicate_oracle_aggregate.csv](../../../experiments/objective_resolution_suite/outputs/final_replicates/reports/replicate_oracle_aggregate.csv)

### 4.2. Gradient audit

- [gradient_mass_summary.csv](../../../experiments/root_cause_ablation_suite/outputs/reports/gradient_mass_summary.csv)

Audit này cho thấy với objective baseline:

- `|y| <= 0.2` nhận khoảng `72.14%` effective gradient mass
- `|y| > 0.5` chỉ nhận khoảng `9.19%`

Đây là bằng chứng trực tiếp rằng current objective đang phân bổ update rất lệch về near-center.

## 5. Baseline hiện tại đang hỏng như thế nào

Từ [full_primary_metrics.csv](../../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv), baseline có:

- `test_mse_0.7eq = 0.05143`
- `test_slope_0.7eq = 0.60596`
- `oracle_teacher_mae = 0.17967`
- `oracle_stable_0.7_slope = 0.57512`
- `oracle_midband_mae_sum_stable = 0.59879`
- `oracle_center_amp_ratio = 5.851`
- `oracle_center_false_0.1eq = 0.5588`
- `oracle_center_false_0.2eq = 0.2059`
- `oracle_center_wrong_sign_0.1eq = 0.1176`
- `oracle_center_spread_ratio = 5.738` trên oracle subset, và `8.755` trên full test proxy space

Diễn giải:

- model biết direction khá tốt ở `0.2..0.7`, nhưng không mở đủ magnitude;
- đồng thời model lại phát score quá mạnh ở ultra-center.

## 6. Những gì đã được chứng minh chắc chắn

### 6.1. Failure A: mid-band magnitude compression là objective-side

Đây là kết luận mạnh nhất hiện tại.

Evidence:

- baseline `oracle_stable_0.7_slope = 0.5751`
- `L2_curvature_y_only = 0.6723`
- `L3_full_A1 = 0.6461`

và:

- baseline `oracle_midband_mae_sum_stable = 0.5988`
- `L2 = 0.5654`
- `L3 = 0.5654`

Các số này nằm trong [full_primary_metrics.csv](../../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv).

Điểm quyết định là:

- `L2_curvature_y_only` chỉ thay đổi phần **inverse-curvature weighting** của y-loss;
- nó đã sửa được Failure A gần như bằng `L3_full_A1`.

Suy ra:

- improvement chính của họ `A1` đến từ **curvature compensation**;
- không cần viện tới label noise, scale, hay architecture để giải thích Failure A.

### 6.2. z-term mạnh hơn không phải driver chính của Failure A

`L1_z_strong_hybrid` chỉ cải thiện nhẹ:

- baseline `oracle_stable_0.7_slope = 0.5751`
- `L1 = 0.5880`

và:

- baseline `oracle_midband_mae_sum_stable = 0.5988`
- `L1 = 0.5887`

Trong khi `L2` đi xa hơn hẳn.

Kết luận:

- tăng vai trò z-space có ích một chút;
- nhưng nó **không phải driver chính** của improvement ở Failure A;
- root cause fix của Failure A nằm ở việc sửa curvature/gradient allocation của y-loss.

### 6.3. Failure B: ultra-center over-confidence là failure mode riêng

Oracle cho baseline:

- `oracle_center_amp_ratio = 5.851`
- `oracle_center_false_0.1eq = 0.5588`
- `oracle_center_false_0.2eq = 0.2059`

`A1/L2/L3` đều làm các metric này xấu hơn:

- `L2 center_amp_ratio = 7.693`
- `L3 center_amp_ratio = 7.147`
- `A1 center_amp_ratio = 7.151`

Điều này cho thấy:

- khi sửa Failure A, center còn xấu hơn;
- vì vậy Failure B **không** chỉ là hậu quả phụ của Failure A;
- nó là một failure mode riêng cần objective/regularization riêng.

### 6.4. A2 chỉ cho thấy gain ở center amplitude/dispersion; chưa đủ power để kết luận về center direction

`A2_band_balanced` có:

- `oracle_center_amp_ratio = 5.551` tốt hơn baseline `5.851`
- `oracle_center_spread_ratio = 5.455` tốt hơn baseline `5.738`

nhưng:

- `oracle_center_wrong_sign_0.1eq = 0.1176`, bằng baseline
- `oracle_center_wrong_sign_0.2eq = 0.0294`, bằng baseline

Final replicate cũng cho cùng pattern:

- baseline `oracle_center_wrong_sign_0.1eq_mean = 0.0866`
- `A2 = 0.0866`

Xem [full_primary_metrics.csv](../../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv) và [final_replicates/replicate_oracle_aggregate.csv](../../../experiments/objective_resolution_suite/outputs/final_replicates/reports/replicate_oracle_aggregate.csv).

Kết luận đúng ở thời điểm hiện tại là:

- `A2` đang động vào **center confidence / dispersion**;
- chưa detect được improvement nào ở **center direction** với power hiện tại.

Đây là `absence of evidence`, không phải `evidence of absence`.

### 6.5. MSE tổng không còn đáng tin để chọn model

Đây là kết luận đã được xác nhận trực tiếp từ run mới.

Ví dụ:

- `A2` có `test_mse_0.7eq = 0.05025`, tốt nhất trong nhóm chính, nhưng gần như không sửa Failure A;
- `L2` có `test_mse_0.7eq = 0.06167`, rất xấu, nhưng lại là run sửa Failure A mạnh nhất.

Suy ra:

- `MSE` tổng đang trộn các regime khác nhau;
- nó không đủ để chọn teacher production.

## 7. Những giả thuyết đã bị hạ ưu tiên

### 7.1. Label noise / search volatility

Oracle diagnostic trước đó cho thấy:

- train label gần high-budget oracle hơn teacher trong phần lớn mẫu;
- instability proxy không tương quan rõ với teacher-vs-oracle error.

Trong chuỗi thí nghiệm mới, không có ablation nào buộc phải gọi tới label noise để giải thích kết quả.

Kết luận:

- label noise có thể tồn tại;
- nhưng **không phải root cause chính** của failure hiện tại.

### 7.2. Global tanh scale mismatch

`C1_scale800` và `C2_scale1200` chỉ cho gain vừa phải ở slope, nhưng đều làm center xấu hơn baseline.

Ví dụ:

- `C1 oracle_stable_0.7_slope = 0.6010`, tốt hơn baseline
- nhưng `oracle_center_amp_ratio = 6.152`, xấu hơn baseline `5.851`

- `C2 oracle_stable_0.7_slope = 0.6077`
- nhưng `oracle_center_amp_ratio = 6.681`

Kết luận:

- scale là yếu tố phụ;
- không phải root cause chính;
- đổi scale một mình không giải bài toán.

### 7.3. Encode/data pipeline là nghi phạm đầu bảng

Chuỗi ablation hiện tại đã tạo ra dịch chuyển metric đúng chiều chỉ bằng objective/sampler.

Do đó:

- chưa có lý do kỹ thuật để ưu tiên `re-encode data` hay thay kiến trúc ngay.

## 8. Kết luận tổng hợp về root cause

Hiện tại có thể chốt thành 2 tầng:

### Root cause chính

**Current objective đang phân bổ gradient sai theo target regime**, dẫn tới:

- under-update ở `|y|≈0.2..0.7`
- từ đó tạo ra `mid-band magnitude compression`

Đây là root cause đã được chứng minh bằng:

- phân tích local curvature;
- gradient mass audit;
- và ablation `L2/L3`.

### Failure B đã được chứng minh ở mức hành vi, nhưng mechanism chưa được chốt

Điều đã được chứng minh:

- center vẫn hỏng ngay cả khi sửa được Failure A;
- `A1/L2/L3` làm center xấu hơn;
- `A2` giúp nhẹ ở amplitude/dispersion;
- `S1` với center penalty hiện tại không cứu được center.

Điều **chưa** được chứng minh:

- Failure B là data-side hay objective-side ở mức cơ chế sâu.

Hiện có hai hypothesis chính:

#### Hypothesis B1 — center-label purity / target semantics

`|y_train|` gần `0` không tương ứng sạch với các position truly neutral theo oracle.  
Nếu đúng, mọi center penalty dựa trực tiếp trên raw `y_train` đều đang nhắm sai tập mẫu.

#### Hypothesis B2 — gradient interference / shared-feature pressure

Update được sinh ra để sửa mid-band cũng đẩy shared representation theo hướng làm logit của oracle-center positions tăng lên.  
Nếu đúng, vấn đề là objective/gradient flow chứ không chỉ là label purity.

Ở thời điểm hiện tại, Failure B mới được chốt ở mức **behavioral failure mode**, chưa chốt được mechanism.

## 9. Những khoảng trống còn lại

### 9.1. Final replicate hiện còn low-power

Final replicate mới có `2` subset cho mỗi finalist:

- baseline
- `A1`
- `A2`
- `L4`
- `S1`

Xem [final_replicates/replicate_oracle_aggregate.csv](../../../experiments/objective_resolution_suite/outputs/final_replicates/reports/replicate_oracle_aggregate.csv).

Nó đủ để xác nhận pattern lớn:

- `A1-family` tốt hơn ở mid-band
- `A2` nhẹ hơn ở center amplitude

nhưng chưa đủ mạnh để chốt winner production-level.

### 9.2. `L0` và `L1` chưa được đưa vào final replicate

Đây là khoảng trống thực nghiệm quan trọng.

Trên oracle subset chính:

- `L0_control_hybrid` có `oracle_teacher_mae = 0.17782`, `closer_rate = 0.2292`
- `L1_z_strong_hybrid` có `oracle_teacher_mae = 0.17746`, `closer_rate = 0.2333`

Vai trò đúng của `L0` và `L1` hiện tại là:

- **attribution controls**
- không phải production candidates đã được xác nhận

`L0` trả lời: train thêm `1` epoch với objective cũ có tự cải thiện hay không.  
`L1` trả lời: tăng vai trò z-space có đóng góp gì thêm ngoài “train thêm 1 epoch” hay không.

Vì vậy hiện tại chưa thể chốt chắc run production tốt nhất giữa:

- `A2`
- `L0`
- `L1`

và cũng chưa thể nói `L1` là candidate an toàn nhất.

### 9.3. Final replicate hiện tại chưa đủ power cho direction tại center

Final replicate hiện có `2` subset mỗi label, và số mẫu center rất nhỏ.

Ví dụ với `oracle_center_wrong_sign_0.1eq_mean ≈ 0.0866`, nếu tổng số mẫu center chỉ ở mức vài chục mỗi replicate thì sai số chuẩn của một rate kiểu này vẫn còn lớn.  
Vì vậy các delta nhỏ ở center direction hiện chưa thể được xem là có ý nghĩa thống kê.

## 10. Hướng giải quyết: những gì đã có thể nói, và những gì chưa

### 10.1. Failure A: hướng giải quyết đã khá rõ

Với Failure A, dữ liệu hiện có đã đủ mạnh để định hướng:

- ưu tiên **loss / objective redesign**
- trọng tâm là **curvature compensation / logit-aware weighting**

Các run `L2` và `L3` là bằng chứng trực tiếp rằng objective dạng này có thể mở lại magnitude ở mid-band.

Vì vậy, với Failure A:

- `loss` là trục chính
- `sampler` chỉ là phụ trợ
- `scale` là secondary factor
- chưa có lý do phải đổi architecture trước

### 10.2. Failure B: hướng giải quyết chưa thể chốt nếu chưa xác định mechanism

Hiện chưa nên chốt sớm rằng Failure B phải được giải bằng:

- loss,
- regularization,
- hay data filter.

Lý do là B1 và B2 dẫn tới các hướng xử lý rất khác:

- nếu `B1` đúng:
  - cần làm sạch center proxy / trusted-center subset / oracle-filtered center regularization
- nếu `B2` đúng:
  - cần objective-side disentanglement hoặc gradient-control mạnh hơn

### 10.3. Architecture hiện chưa phải ưu tiên số 1

Ở thời điểm hiện tại, chưa có evidence nào buộc phải ưu tiên đổi kiến trúc.

Lý do:

- ta đã thấy metric dịch chuyển mạnh chỉ bằng thay objective;
- chưa có experiment nào cho thấy representation hiện tại là bottleneck chính.

Điều này không có nghĩa architecture vô can. Nó chỉ có nghĩa:

- chưa tới lúc nhảy sang redesign backbone/head như bước kế tiếp đầu tiên.

## 11. Ý tưởng giải quyết theo từng failure mode

### 11.1. Ý tưởng cho Failure A

Đây là nhánh đã có evidence tốt nhất.

#### Ý tưởng A-loss

Dùng objective kiểu:

```text
weighted_y_loss with inverse-curvature compensation
+ logit-space term đủ mạnh để giữ geometry ổn định
```

Tinh thần là:

- không để local curvature tại `|y| lớn hơn` co quá nhanh;
- tránh lặp lại tình trạng `72%` gradient mass dồn vào `|y|<=0.2`.

#### Ý tưởng A-sampler

Giữ `A2` như phụ trợ, không phải lõi chính:

- band-balanced sampler có thể giảm density dominance ở center;
- nhưng hiện chưa cho thấy nó tự giải được Failure A.

### 11.2. Ý tưởng cho Failure B nếu B1 đúng

Nếu center raw labels không sạch so với oracle-center, hướng đúng là:

- xây một `trusted-center subset`
- hoặc center regularization chỉ áp trên mẫu có oracle-consistent center proxy

Đây là data-side fix, không phải architecture fix.

### 11.3. Ý tưởng cho Failure B nếu B2 đúng

Nếu update từ mid-band làm oracle-center logits tăng theo, hướng đúng là objective-side:

- center-aware regularization mạnh hơn;
- có thể là gradient disentanglement;
- hoặc auxiliary center-confidence objective tách riêng khỏi main value objective.

Ở bước này, tôi vẫn ưu tiên `loss / regularization` trước `architecture`.

### 11.4. Ý tưởng architecture chỉ nên là nhánh sau

Nếu sau khi:

- sửa objective cho A,
- và chẩn đoán xong B1/B2,

mà center vẫn hỏng, khi đó mới hợp lý để cân nhắc:

- calibration head riêng,
- confidence/uncertainty head,
- hoặc head hai nhánh cho magnitude và center-confidence.

## 12. Metric gate cần dùng từ bây giờ

### 10.1. Nếu mục tiêu là giải đúng root cause

Tiếp tục theo họ `A1`, vì đây là nhánh duy nhất đã chứng minh được cách sửa Failure A.

Nhưng không dùng `A1` thuần cho production, vì nó phá center quá mạnh.

Hướng đúng là:

```text
A1-family objective
+ center-specific control mạnh hơn hiện tại
```

Tức là:

- giữ inverse-curvature compensation;
- thêm cơ chế center-aware đủ mạnh để kéo xuống:
  - `center_amp_ratio`
  - `center_false_0.1`
  - `center_false_0.2`
  - `center_spread_ratio`

### 10.2. Nếu mục tiêu là có candidate an toàn hơn ngay

Kiểm chứng `L1_z_strong_hybrid` trước.

Lý do:

- `L1` tốt hơn baseline về `oracle_teacher_mae`, `oracle_closer_rate`, và `stable_0.7_slope`;
- nó không làm center nổ như họ `A1`;
- nó là candidate có profile thực dụng nhất hiện tại trong nhóm follow-up chưa được replicate cuối.

## 11. Metric gate cần dùng từ bây giờ

Không dùng `MSE` tổng làm metric quyết định.

Các gate chính nên là:

- `oracle_stable_0.7_slope`
- `oracle_midband_mae_sum_stable`
- `oracle_teacher_mae`
- `teacher_closer_to_oracle_rate`
- `oracle_center_amp_ratio`
- `oracle_center_false_0.1eq`
- `oracle_center_false_0.2eq`
- `oracle_center_spread_ratio`

Direction metric vẫn nên log, nhưng ở ultra-center cần diễn giải cẩn thận vì:

- sign semantics yếu khi `|oracle|` cực nhỏ;
- replicate hiện tại chưa đủ power để kết luận từ delta nhỏ.

## 13. Roadmap thực nghiệm tiếp theo

### Bước 0: chẩn đoán mechanism của Failure B

Đây là bước bị thiếu trong spec cũ.

#### D1 — Center-label purity audit

Mục tiêu:

- đo xem `|y_train| <= tau` có thực sự map tốt tới `|oracle| <= tau_oracle` hay không.

Cần đo trên subset oracle:

- precision / recall của raw train-center labels đối với oracle-center
- confusion table cho các ngưỡng `0.05` và `0.1`

Nếu precision thấp, điều đó ủng hộ `B1`.

#### D2 — Gradient interference audit

Mục tiêu:

- kiểm tra update từ band nào đang làm oracle-center logits tăng lên.

Cần đo:

- gradient norm theo band
- cosine similarity giữa gradient center và gradient mid-band
- one-step influence:
  - lấy một oracle-center probe set cố định
  - apply một optimizer step giả lập từ batch của từng band
  - đo `mean |pred|` hoặc `mean |logit|` trên probe set thay đổi ra sao

Nếu batch mid-band làm oracle-center probe tăng có hệ thống, điều đó ủng hộ `B2`.

### Bước 1: final replicate cho `L0` và `L1`

Mục tiêu:

- chốt attribution:
  - train thêm có tự giúp không?
  - z-space mạnh hơn có đóng góp thêm ngoài “train thêm” không?

Nếu `L0 ≈ L1`, bỏ `L1` như một hướng riêng.

### Bước 2: chỉ sau D1/D2 mới quyết định nhánh xử lý Failure B

#### Nếu D1 nghiêng mạnh về B1

Ưu tiên:

- trusted-center subset
- oracle-filtered center regularization
- hoặc center penalty chỉ áp lên center-clean samples

#### Nếu D2 nghiêng mạnh về B2

Ưu tiên:

- objective kiểu `A1 + center-aware control`
- và thiết kế regularization theo gradient behavior, không chỉ theo raw train-center labels

### Bước 3: objective sweep kiểu `A1 + stronger center control`

Mục tiêu:

- giữ gain của `A1/L2` ở mid-band;
- giảm center blow-up;
- chọn winner bằng oracle gates, không bằng MSE.

### Bước 4: chỉ sau đó mới đánh giá lại scale hoặc architecture

Scale và architecture hiện là nhánh sau, không phải bước đầu.

## 14. Chốt cuối cùng

Từ toàn bộ thực nghiệm đã chạy, có thể chốt một cách chặt chẽ:

1. Root cause chính hiện tại là **objective-side gradient allocation**, không phải `tanh scale`, không phải `label noise`, và chưa có bằng chứng đổ cho encoding.
2. Mạng đang có **hai failure mode riêng**:
   - `mid-band magnitude compression`
   - `ultra-center over-confidence`
3. `A1/L2/L3` chứng minh cách sửa failure đầu tiên.
4. `A2` cho thấy center amplitude/dispersion là một cơ chế riêng, nhưng chưa đủ power để kết luận về direction.
5. Mechanism của Failure B hiện chưa chốt xong; cần `center-label purity audit` và `gradient interference audit`.
6. Hiện chưa có candidate nào giải được cả hai failure mode cùng lúc.
7. Hướng đúng tiếp theo là:
   - `D1 + D2` để chốt mechanism của Failure B;
   - `R1` replicate `L0/L1` để chốt attribution;
   - rồi mới phát triển objective kiểu `A1 + stronger center control`.

Đây là kết luận phù hợp nhất với toàn bộ evidence hiện có và không cần viện tới giả thuyết ngoài dữ liệu.
