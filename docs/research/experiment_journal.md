# Mạng Giá Trị Engine Cờ Vua - Nhật ký Thí nghiệm & Đặc tả Kỹ thuật

**Loại tài liệu:** nhật ký thí nghiệm + đặc tả kỹ thuật đang làm việc  
**Kho mã:** `chess_engine`  
**Cập nhật lần cuối:** 2026-04-01

## 1. Bài toán và mô hình hiện tại

Dự án này là một **mạng hồi quy giá trị (value regression network)** cho engine cờ vua. Đây không phải là mô hình policy-value theo kiểu AlphaZero.

Thiết lập lõi hiện tại:

- Tensor đầu vào: `18 x 8 x 8`
- Góc nhìn: `STM-relative`
- Mục tiêu chuẩn:
  - `y = tanh(cp / 600)`
- Checkpoint baseline của teacher:
  - External baseline checkpoint used during analysis: `dgrn_5m_v3_stage2_polish_run1/ckpt_best.pt` (not versioned in this repository)
- Họ mạng hiện tại:
  - `architecture_v2`
- Head/đầu ra:
  - mạng dự đoán logit `z`
  - giá trị cuối là `p = tanh(z)`

Nhật ký này ghi lại chuỗi thí nghiệm đã chạy để trả lời hai câu hỏi:

1. Vì sao mô hình đánh giá thấp các lợi thế không nhỏ ở dải giữa?
2. Vì sao mô hình lại trở nên quyết đoán sai ở các vị trí thực sự trung hòa?

## 2. Phát biểu các lỗi hiện tại

### 2.1 Lỗi A: nén biên độ dải giữa (mid-band magnitude compression)

Định nghĩa:

- Trong khoảng xấp xỉ `0.2 <= |y| <= 0.7`, mô hình thường dự đoán đúng dấu.
- Nhưng độ lớn dự đoán lại quá nhỏ.
- Đây là vấn đề về hiệu chỉnh / effective-gradient, không chủ yếu là vấn đề về dấu.

Triệu chứng quan sát:

- `oracle_stable_0.7_slope` thấp đáng kể so với `1.0`
- `oracle_midband_mae_sum_stable` vẫn cao ngay cả khi độ chính xác dấu đã ổn

### 2.2 Lỗi B: quá tự tin ở vùng siêu trung tâm (ultra-center over-confidence)

Định nghĩa:

- Ở các vị trí gần `0` và thực sự trung hòa, ổn định, mô hình dự đoán `|p|` quá lớn.
- Điều này tạo ra nhiều đầu ra "quyết đoán giả" ở các vị trí lẽ ra phải nằm sát 0.

Triệu chứng quan sát:

- `oracle_center_amp_ratio` cao hơn nhiều so với `1.0`
- `oracle_center_false_0.1eq` và `oracle_center_false_0.2eq` ở mức cao
- Các metric center gộp (pooled center metrics) cũng cho cùng một mô thức trên một center bundle riêng

### 2.3 Diễn giải cơ chế hiện tại

Bằng chứng hiện tại ủng hộ rằng:

- `Lỗi A` là lỗi **phía objective**
- `Lỗi B` là tổ hợp của:
  - `B1`: nhãn near-zero thô (raw near-zero labels) chưa đủ sạch để dùng làm giám sát center trực tiếp
  - `B2`: gradient từ dải giữa / đuôi can nhiễu vào hành vi center trong các tham số dùng chung

Đây là cách phân rã hiện được hậu thuẫn mạnh nhất. Nó chặt chẽ hơn nhận định mơ hồ kiểu "dataset xấu", đồng thời chính xác hơn nhận định "head bị lệch chuẩn".

## 3. Vì sao MSE không phải metric chính

Raw global MSE không phải metric ra quyết định phù hợp cho dự án này.

Lý do:

- Mục tiêu là `tanh(cp / 600)`, nên khoảng cách hình học trong không gian `y` không khớp với các failure mode thực sự mà ta quan tâm.
- Mô hình có thể trông ổn theo global MSE nhưng vẫn:
  - nén các lợi thế không tầm thường
  - quá tự tin nghiêm trọng ở gần zero

Với `p = tanh(z)`, squared loss trong không gian `y` là:

```text
L_y = (p - y)^2 = (tanh(z) - y)^2
```

Gần logit mục tiêu `z* = atanh(y)`, hình học cục bộ tạo ra hệ số độ cong tỉ lệ với `(1 - y^2)^2`. Điều đó có nghĩa là các vùng `|y|` lớn nhận áp lực học hiệu dụng yếu hơn khi chỉ hồi quy trực tiếp trong không gian `y`. Đây là lý do lý thuyết chính khiến Lỗi A bị nghi ngờ là lỗi phía objective.

## 4. Các metric đánh giá cốt lõi

Dự án dựa vào **oracle-based metrics** thay vì một global MSE đơn lẻ.

### 4.1 Nhóm stable-oracle metrics cho Lỗi A

- `oracle_stable_0.7_slope`
  - Độ dốc tuyến tính của `pred ~ oracle` trên tập stable đến ngưỡng `|oracle| <= 0.7`
  - Diễn giải mức hiệu chỉnh độ lớn
  - Càng cao càng tốt, lý tưởng gần `1.0`

- `oracle_midband_mae_sum_stable`
  - Tổng stable-oracle MAE trên các dải chính không thuộc center
  - Càng thấp càng tốt
  - Đây là scalar summary gọn và sạch nhất cho Lỗi A

- `oracle_band_mae_0.05_0.2_stable`, `oracle_band_mae_0.2_0.5_stable`, `oracle_band_mae_0.5_0.7_stable`
  - Dùng để xác định phương pháp nào cải thiện hoặc làm xấu ở từng dải

### 4.2 Nhóm center metrics cho Lỗi B

- `oracle_center_amp_ratio`
  - `mean(|pred|) / mean(|oracle|)` trên các vị trí oracle near-zero ổn định
  - Lý tưởng gần `1.0`
  - Giá trị cao hơn nhiều `1.0` cho thấy center bị phồng biên độ

- `oracle_center_false_0.1eq`, `oracle_center_false_0.2eq`
  - Tỉ lệ vị trí oracle-center ổn định có `|pred| >= 0.1` hoặc `>= 0.2`
  - Càng thấp càng tốt

- `pooled_center_mae`, `pooled_center_amp_ratio`, `pooled_center_false_0.1eq`, `pooled_center_false_0.2eq`
  - Cùng họ kiểm tra, nhưng được đánh giá trên pooled center bundle dựng trong Failure B suite
  - Đặc biệt hữu ích cho so sánh chéo giữa các suite

### 4.3 Composite scores

Trong các thí nghiệm có hai composite score:

- `selection_score_v2`
  - Dùng trong objective resolution suite
- `failure_b_score`
  - Dùng trong Failure B suite

Lưu ý quan trọng:

- Composite score hữu ích để xếp hạng bên trong từng suite.
- Chúng **không** phải lập luận khoa học cuối cùng khi đứng riêng lẻ.
- Các metric cơ sở (primitive metrics) mới là bằng chứng chính.

### 4.4 Lưu ý về kích thước dữ liệu

- Tập con oracle chẩn đoán chính có `n = 240` vị trí trong báo cáo tổng hợp.
- Pooled center bundle dùng trong Failure B suite có `n = 22`.

Vì vậy:

- Kết luận về center nhất quán về xu hướng và lặp lại ở nhiều suite
- Nhưng pooled center bundle vẫn nhỏ và cần được diễn giải thận trọng

## 5. Dòng thời gian thí nghiệm

### 5.1 Oracle root cause diagnostic

**Đầu ra chính:**  
`../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_root_cause_summary.json`

**Câu hỏi**

- Lỗi A và B có thật dưới fixed-node oracle, hay phần lớn chỉ là artifact do nhãn nhiễu?

**Thiết kế**

- Xây dựng tập con được đánh giá bằng oracle với supervision fixed-node từ Stockfish
- So sánh:
  - training label so với oracle
  - checkpoint hiện tại so với oracle
- Tách riêng hành vi stable/unstable khi cần

**Các đầu ra chính**

- `teacher_closer_to_oracle_rate_600_overall = 0.2167`
- `teacher_closer_to_oracle_rate_600_near_zero = 0.1414`
- `train_vs_oracle_mae_600_near_zero = 0.0360`
- `teacher_vs_oracle_mae_600_near_zero = 0.1199`
- `stable_0.7_slope_600 = 0.5751`

**Diễn giải**

- Ở gần zero, checkpoint baseline hiện tại tệ hơn rõ rệt so với raw label khi đối chiếu oracle.
- Điều này bác bỏ trực tiếp nhận định ngây thơ rằng "toàn bộ dataset đơn giản là quá tệ" như nguyên nhân chính.
- Nó cũng xác nhận center over-confidence là lỗi mô hình có thật, không chỉ là câu chuyện nhiễu dữ liệu trên diện rộng.
- `stable_0.7_slope_600 = 0.5751` là bằng chứng trực tiếp của hiện tượng nén dải giữa.

**Kết luận**

- Lỗi A là có thật.
- Lỗi B là có thật.
- Checkpoint baseline hiện tại không đơn thuần là bộ ước lượng đã khử nhiễu của raw label quanh vùng center.

### 5.2 Root cause ablation suite

**Các đầu ra chính**

- `../../experiments/root_cause_ablation_suite/outputs/reports/compare_runs.csv`
- `../../experiments/root_cause_ablation_suite/outputs/reports/gradient_mass_summary.csv`

**Câu hỏi**

- Lỗi A chủ yếu do hình học của objective, do scale, hay do sampling?

**Thiết kế**

- So sánh:
  - `baseline`
  - `A1_curvature_compensated`
  - `A2_band_balanced`
  - `C1_scale800`
  - `C2_scale1200`

Ý tưởng lý thuyết cốt lõi của `A1` là bù thiên lệch độ cong trong không gian `y` bằng cách tái trọng số effective `y` loss để đưa gradient mass quay lại các vùng `|y|` lớn hơn.

**Các đầu ra chính từ `compare_runs.csv`**

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

**Các đầu ra chính từ `gradient_mass_summary.csv`**

Tỉ phần effective gradient mass:

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

**Diễn giải**

- `A1` tạo ra đúng dịch chuyển định tính đã dự đoán:
  - ít gradient mass hơn ở center
  - nhiều gradient mass hơn đáng kể ở upper mid-band và tail
- Điều này khớp với cải thiện ở slope và mid-band MAE.
- Các thay đổi chỉ về scale (`C1`, `C2`) không tạo ra chất lượng cải thiện tương đương.
- `A2` cải thiện một số metric, nhưng không xử lý hình học trực diện như `A1`.

**Kết luận**

- Lỗi A được hậu thuẫn mạnh là một vấn đề **phía objective**.
- Chỉ thay đổi tanh scale không phải nguyên nhân gốc.
- Sampling và scale có thể làm dịch chuyển metric, nhưng không giải thích hiệu ứng chính.

### 5.3 Objective resolution suite

**Các đầu ra chính**

- `../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv`
- `../../experiments/objective_resolution_suite/outputs/reports/full_bootstrap_summary.csv`
- `../../experiments/objective_resolution_suite/outputs/reports/replicate_oracle_aggregate.csv`

**Câu hỏi**

- Họ objective nào là nền checkpoint tốt nhất để đi tiếp?
- Objective thuộc họ A1 có thể sửa Lỗi A mà không làm Lỗi B vượt ngưỡng chấp nhận hay không?

**Các biến thể được so sánh**

- `L0_control_hybrid`
- `L1_z_strong_hybrid`
- `L2_curvature_y_only`
- `L3_full_A1`
- `L4_A1_plus_A2`
- `S1_A1_center_w020_m010`

**Các đầu ra chính từ `full_primary_metrics.csv`**

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

**Các đầu ra chính từ `full_bootstrap_summary.csv`**

Các cờ bootstrap quan trọng:

- `L4_A1_plus_A2`
  - `supports_improvement_midband_teacher_vs_oracle_mae_sum_stable = True`
  - `supports_improvement_stable_0.7_slope = True`
  - các cờ cải thiện center vẫn là `False`

- `L3_full_A1`
  - `supports_improvement_stable_0.7_slope = True`
  - nhưng `supports_improvement_midband_teacher_vs_oracle_mae_sum_stable = False` ở ngưỡng của suite

Điểm này quan trọng vì `L4` là checkpoint họ A có bằng chứng sạch nhất cho cả hai metric chính của Lỗi A cùng lúc.

**Đầu ra hỗ trợ từ `replicate_oracle_aggregate.csv`**

Replicate aggregate chỉ gồm `baseline`, `A1_curvature_compensated` và `A2_band_balanced`, nhưng cho cùng hướng kết luận:

- mean stable slope của `A1`: `0.8272` so với baseline `0.7618`
- mean stable midband MAE sum của `A1`: `0.4551` so với baseline `0.4617`
- mean center amp ratio của `A1`: `7.9726` so với baseline `6.5306`

Vì vậy, trên các oracle subset lặp lại, `A1` tiếp tục cải thiện Lỗi A nhưng làm xấu Lỗi B.

**Diễn giải**

- Objective họ A1 là phương án sửa Lỗi A tốt nhất đã tìm thấy.
- Nhưng chúng làm Lỗi B tệ hơn.
- Center penalty dạng ngây thơ (`S1`) không giải được Lỗi B và không trội hơn `L4`.
- `L0` là checkpoint cân bằng an toàn nhất nếu phải dùng mô hình ngay.

**Kết luận**

- Nếu cần dùng ngay theo hướng cân bằng: `L0_control_hybrid` là lựa chọn an toàn nhất.
- Nếu tiếp tục giải đồng thời A và B: `L4_A1_plus_A2` là checkpoint nền hợp lý nhất.

### 5.4 Failure B resolution suite

**Các đầu ra chính**

- `../../experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_primary_metrics.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_pooled_center_metrics.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/center_label_purity_summary.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/center_purity_lookup_report.json`
- `../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json`
- `../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_influence.csv`
- `../../experiments/failure_b_resolution_suite/outputs/replicates/l0_l1_controls/reports/replicate_oracle_aggregate.csv`

**Câu hỏi**

- Lỗi B chủ yếu do độ tạp nhãn center (center-label impurity), do gradient interference, hay do cả hai?
- Raw-center penalty có giải quyết được không?

#### 5.4.1 Bằng chứng độ sạch nhãn cho B1

Từ `center_label_purity_summary.csv`:

- Với `raw |y| <= 0.05` và ngưỡng oracle center `0.05`
  - `raw_center_count = 96`
  - `oracle_center_clean_count = 22`
  - `precision = 0.21875`
  - `recall = 0.9545`

- Với `raw |y| <= 0.10` và ngưỡng oracle center `0.05`
  - `raw_center_count = 147`
  - `precision = 0.14966`
  - `recall = 1.0000`

Từ `center_purity_lookup_report.json`:

- `base_rate = 0.14966`
- `smoothed_clean_rate_max = 0.26657`

**Diễn giải**

- Ngưỡng center từ raw label có recall cao nhưng precision thấp.
- Chúng phù hợp để **khai thác ứng viên (mining candidates)**, không phù hợp để giám sát trực tiếp.
- Tín hiệu lookup/proxy chỉ cải thiện xếp hạng nhẹ, chưa thể trở thành nguồn nhãn sạch thực thụ.

Đây là bằng chứng chính cho `B1`.

#### 5.4.2 Bằng chứng gradient interference cho B2

Từ `gradient_interference_cosines.csv`:

- Baseline:
  - cosine(`center_raw_0_005`, `mid_05_07`) = `-0.6423`
  - cosine(`near_center_005_02`, `mid_05_07`) = `-0.5549`

- A1 objective:
  - cosine(`center_raw_0_005`, `mid_05_07`) = `-0.6805`
  - cosine(`near_center_005_02`, `mid_05_07`) = `-0.6719`

Từ `gradient_interference_influence.csv`, chỉ một bước nhỏ theo gradient mid-band cũng làm tăng mức quyết đoán ở center:

- Baseline:
  - `mid_02_05 -> center_clean_005`: `delta_mean_abs_pred = +0.00213`
  - `mid_05_07 -> center_clean_005`: `delta_mean_abs_pred = +0.00253`

- A1 objective:
  - `mid_02_05 -> center_clean_005`: `delta_mean_abs_pred = +0.00352`
  - `mid_05_07 -> center_clean_005`: `delta_mean_abs_pred = +0.00277`

Từ `gradient_interference_summary.json`:

- mức phồng center-probe trung bình do gradient mid-band gây ra:
  - baseline objective: `0.00233`
  - A1 objective: `0.00314`

**Diễn giải**

- Gradient của center và upper mid-band là anti-aligned trong các tham số dùng chung.
- Với objective kiểu A1, mức can nhiễu này còn mạnh hơn.

Đây là bằng chứng chính cho `B2`.

#### 5.4.3 Kết quả các pilot

Từ `combined_failure_b_primary_metrics.csv`:

- Các Failure B score tốt nhất hiện tại:
  - `A2_band_balanced = 0.6173`
  - `L0_control_hybrid = 0.6182`
  - `baseline = 0.6506`

- Nhóm A-family / center-penalty / B-pilot:
  - `L4_A1_plus_A2 = 0.7330`
  - `S1_A1_center_w020_m010 = 0.7426`
  - `P_B2_raw_center_strong = 0.7625`
  - `L3_full_A1 = 0.8049`
  - `P_B1_proxy_center_weighted = 0.8559`

Từ `combined_failure_b_pooled_center_metrics.csv`:

- `A2` và `L0` đều cải thiện pooled center metrics so với baseline
- `P_B1` và `P_B2` đều vẫn quá quyết đoán ở gần zero

**Diễn giải**

- `P_B2_raw_center_strong` thất bại vì raw-center penalty không giải được vấn đề impurity và không loại bỏ được interference.
- `P_B1_proxy_center_weighted` thất bại vì trần độ sạch của proxy quá thấp để trở thành tín hiệu huấn luyện chính.
- `S1` cho thấy việc ép center đơn giản từ raw labels vẫn chưa đủ, kể cả khi chạy full run.

**Kết luận**

- Lỗi B không thể giải thích bằng một nguyên nhân đơn lẻ.
- Cách đọc hợp lý nhất hiện tại là:
  - `B1` là có thật
  - `B2` là có thật
  - raw-center penalty không phải lời giải

### 5.5 OC1: pilot hiệu chỉnh oracle-center giai đoạn muộn

**Các đầu ra chính**

- `../../experiments/l4_oracle_center_correction_pilot/outputs/reports/combined_oracle_center_pilot_primary_metrics.csv`
- `../../experiments/l4_oracle_center_correction_pilot/outputs/reports/pilot_history.json`
- `../../experiments/l4_oracle_center_correction_pilot/outputs/cache/oracle_probe/oracle_candidate_report.json`

**Câu hỏi**

- Có thể dùng một đợt hiệu chỉnh ngắn ở cuối, bắt đầu từ `L4` và thêm oracle center auxiliary set, để sửa Lỗi B hay không?

**Thiết kế**

- Bắt đầu từ `L4`
- Xây dựng một center auxiliary bundle nhỏ đã được oracle-corrected
- Fine-tune ở giai đoạn muộn

**Kết quả khai thác oracle**

- `192` ứng viên
- `88` ổn định
- `35` center-clean
- `87` dòng aux được giữ lại

**Kết quả đánh giá có thẩm quyền**

Từ `combined_oracle_center_pilot_primary_metrics.csv`:

- `L4 failure_b_score = 0.7330`
- `OC1 failure_b_score = 0.8916`

Center tệ hơn:

- `pooled_center_mae`: `0.1090 -> 0.1283`
- `pooled_center_amp_ratio`: `4.799 -> 5.916`
- `pooled_center_false_0.1eq`: `0.5000 -> 0.5909`

**Cảnh báo quan trọng**

File `pilot_history.json` nội bộ về sau được xác định là không đáng tin để kết luận khoa học vì đường eval cuối epoch bị sai. Kết quả có thẩm quyền là CSV so sánh đã đánh giá lại.

**Kết luận**

- `OC1` thất bại.
- Nó bác bỏ cách triển khai cụ thể đó của late oracle correction.
- Nó **chưa** bác bỏ dứt khoát toàn bộ ý tưởng, vì bản thân `OC1` còn lỗi logic.

### 5.6 OC2: joint oracle full-model pilot

**Các đầu ra chính**

- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_primary_metrics.csv`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_pooled_center_metrics.csv`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/oc2_pilot_history.json`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/decision_summary.json`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/cache/oracle_probe_2d/oracle_candidate_report.json`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/cache/oracle_role_bundle/manifest.json`

**Câu hỏi**

- Nếu sửa đúng logic late-polish, liệu một đợt joint correction ngắn trên full-model còn đường để sửa Lỗi B mà không phá Lỗi A hay không?

**Các cải tiến thiết kế so với OC1**

- Toàn bộ model có thể train, không chỉ head + vài block
- Joint objective đúng nghĩa với một optimizer step
- Hành vi `eval()` cuối epoch được xử lý đúng
- Khai thác ứng viên 2D theo:
  - `raw |y|`
  - `|pred_L4|`
- Chia oracle bank thành:
  - `center_anchor`
  - `center_hard`
  - `center_ambiguous`
- Hard midband gate + lựa chọn chỉ theo center

**Mức bao phủ của oracle bank**

Theo cache manifest:

- `192` ứng viên trong 2D candidate bundle
- `93` ổn định
- `39` center-clean
- `78` dòng aux được giữ
- Số lượng theo role:
  - `14 center_anchor`
  - `25 center_hard`
  - `39 center_ambiguous`

Điều này cho thấy OC2 không thất bại vì oracle bank quá nhỏ hoặc rỗng.

**Phần cứng / runtime**

Từ `joint_batch_autotune.json`:

- chọn `main_batch_size = 384`
- ước lượng peak VRAM `2.205 GB` trên GPU `4 GB`

Từ `oc2_pilot_history.json`:

- `1` epoch
- thời gian mỗi epoch khoảng `981.1 s`

Với `400k` mẫu chính và `1042` bước/epoch, aux bank được replay với tần suất cao. Chi tiết này liên quan đến diễn giải, nhưng tự nó không làm mất giá trị kết quả.

**Kết quả**

Từ `decision_summary.json`:

- `best_any_center_score = 0.6035`
- `best_gate_center_score = 0.6035`
- `l4_center_score = 0.5253`
- `has_gate_checkpoint = true`

OC2 vượt được midband gate nhưng vẫn thua ở center.

Từ `combined_oc2_final_pilot_primary_metrics.csv`:

So với `L4`:

- `oracle_midband_mae_sum_stable`: `0.5710 -> 0.5745`
- `oracle_stable_0.7_slope`: `0.6179 -> 0.6442`
- `oracle_band_mae_0.2_0.5_stable`: `0.1889 -> 0.1853`
- `oracle_band_mae_0.5_0.7_stable`: `0.2525 -> 0.2396`

Nhưng center xấu đi:

- `pooled_center_mae`: `0.1090 -> 0.1241`
- `pooled_center_amp_ratio`: `4.799 -> 5.430`
- `oracle_center_amp_ratio`: `6.379 -> 7.245`
- `oracle_center_false_0.2eq`: `0.2353 -> 0.3529`
- `pooled_center_wrong_sign_0.1`: `0.0909 -> 0.1364`

**Diễn giải**

- OC2 là phép thử âm (negative test) tốt hơn OC1.
- Nó đã sửa các vấn đề triển khai chính của OC1.
- Nó giữ được phần lớn lợi ích phía A.
- Nhưng vẫn không cải thiện được B.

**Kết luận thực dụng từ OC2**

- Late correction ngắn không còn là hướng chính đầy hứa hẹn.
- Điều này **không** chứng minh toán học rằng mọi short fine-tune đều bất khả thi.
- Nhưng bằng chứng thực nghiệm đã đủ mạnh để hạ ưu tiên late polish như hướng chính.

**Lưu ý tồn dư**

Mô hình dùng BatchNorm nhiều. Ở train-mode aux pass, một oracle bank nhỏ cũng có thể làm lệch running statistics. Vì vậy OC2 vẫn chưa phải phản chứng hoàn hảo cho mọi chế độ hiệu chỉnh ngắn có thể có. Tuy nhiên, kết quả thực dụng vẫn nghiêng mạnh theo hướng tiêu cực.

## 6. Điều đã được chứng minh và điều vẫn còn là giả thuyết

### 6.1 Những điểm được hậu thuẫn mạnh bởi đầu ra

- Lỗi A là lỗi phía objective.
- Lỗi B không thể giải thích bằng "dataset xấu toàn cục".
- `B1` (center-label impurity) là có thật.
- `B2` (gradient interference) là có thật.
- Objective họ A1 cải thiện Lỗi A nhưng làm xấu Lỗi B.
- Raw-center penalty không phải lời giải hiệu quả cho Lỗi B.
- Proxy-weighted raw-center penalty cũng không phải lời giải hiệu quả cho Lỗi B.
- Late polish từ `L4` không phải hướng đáng tin:
  - `OC1` thất bại
  - `OC2` cũng thất bại, dù thiết kế sạch hơn nhiều

### 6.2 Những điểm còn là giả thuyết, chưa chứng minh xong

- Cơ chế biểu diễn nội tại chính xác của Lỗi B bên trong backbone vẫn là giả thuyết, dù dấu hiệu "shared-feature interference" rất rõ.
- Một full-training run với clean oracle center anchors ngay từ epoch `0` chưa được chạy, nên vẫn là giả thuyết thực nghiệm mở quan trọng nhất.
- Search compensation có thể giúp sức mạnh chơi downstream, nhưng chưa được định lượng trong tài liệu này và không thay thế việc sửa hành vi của evaluator.

## 7. Hướng dẫn chọn checkpoint hiện tại

### 7.1 Nếu cần checkpoint dùng ngay

- Lựa chọn cân bằng an toàn nhất:
  - `L0_control_hybrid`
- Nếu ưu tiên hành vi center hơn việc sửa phía A:
  - `A2_band_balanced` cũng là lựa chọn mạnh

### 7.2 Nếu cần checkpoint làm nền cho công việc sửa mô hình tiếp theo

- Sử dụng:
  - `L4_A1_plus_A2`

Lý do:

- Đây là nền hiện tại hợp lý nhất nếu mục tiêu vẫn là sửa đồng thời cả A và B ở cấp độ mô hình.
- Nó giữ được phần cải thiện phía A đáng tin nhất mà chưa cần đổi architecture hay encoding.

## 8. Hướng đi tiếp theo được khuyến nghị

### 8.1 Những hướng không nên là đường chính

Các đầu ra hiện tại cho thấy không nên ưu tiên:

- thêm một raw-center penalty kiểu ngây thơ nữa
- thêm một điều chỉnh chỉ về scale nữa
- thêm một notebook late oracle-center polish như nhánh chính

### 8.2 Hướng phía mô hình khả dĩ nhất ở thời điểm hiện tại

Hướng tiếp theo được hậu thuẫn tốt nhất là:

- giữ objective chính kiểu `L4`
- can thiệp **ngay từ epoch 0**, không đợi đến late polish
- thêm giám sát oracle center sạch như một nguồn rủi ro riêng
- giảm trọng số nhãn center mơ hồ từ raw labels trong loss chính
- xử lý BatchNorm cẩn thận để oracle bank nhỏ không làm hỏng running stats

Một dạng biểu diễn toán học nhất quán:

```text
L_total
= E_raw [ w_raw(x) * L_L4(f(x), y_raw) ]
+ lambda_anchor * E_center_anchor [ Huber(f(x) - y_oracle) + lambda_margin * ReLU(|f(x)| - m)^2 ]
+ lambda_ambig * E_center_ambiguous [ Huber(f(x) - y_oracle) ]
```

Trong đó:

- `L_L4` giữ lại lời giải tốt nhất hiện có cho Lỗi A
- `w_raw(x)` giảm trọng số của các nhãn raw-center mơ hồ
- `center_anchor` cung cấp giám sát center đáng tin cậy
- `center_ambiguous` ngăn việc ép tất cả ứng viên raw-center về 0 một cách cực đoan

### 8.3 Vì sao đây là hướng hợp lý nhất

Các đầu ra hội tụ về hướng này một cách nhất quán:

- Thay đổi objective trong full training dịch chuyển Lỗi A rất rõ.
- Late correction, kể cả sau khi sửa lỗi logic của OC1, vẫn không sửa được Lỗi B.
- Lỗi B được giải thích đồng thời bởi:
  - nhiễu trong giám sát center
  - can nhiễu trên tham số dùng chung

Tổ hợp đó chỉ ra tự nhiên rằng hiệu chỉnh phải được bơm vào **global training dynamics**, thay vì chỉ vá ở đoạn cuối.

### 8.4 Ghi chú triển khai thực tế cho lần chạy kế tiếp

Lần chạy nghiêm túc tiếp theo nên giữ ràng buộc tối thiểu:

- chưa đổi architecture
- chưa re-encode toàn bộ dataset
- chưa đổi target scale

Nhưng cần bổ sung:

- center-aware loss trong toàn bộ quá trình train từ epoch `0`
- trusted oracle center bank
- cơ chế BN-safe khi chạy oracle batch
  - hoặc freeze BN running stats trong các oracle-only pass
  - hoặc đảm bảo có mixed batch để aux supervision không làm méo thống kê
- chọn checkpoint theo:
  - hard midband gate
  - sau đó mới xét center score

## 9. Trạng thái cuối của dự án

Tại thời điểm hiện tại:

- Lỗi A đã được hiểu đủ sâu để thiết kế objective tương ứng.
- Lỗi B đã được hiểu đủ rõ để loại bỏ nhiều lớp lời giải sai.

Các đầu ra hiện đang chỉ về một hướng rất rõ:

- Lỗi B **không hành xử như** lỗi calibration muộn có thể sửa ổn định chỉ bằng một đợt polish ngắn.
- Bằng chứng tốt nhất hiện tại nghiêng về vấn đề full-training dynamics, chịu chi phối bởi label impurity và gradient interference.
- Vì vậy, thí nghiệm hợp lý nhất tiếp theo là một **full run có nhận thức center, tiêm cơ chế sửa từ epoch 0** trên nền objective họ `L4`, thay vì thêm một notebook polish mới.

## 10. Các file nguồn dùng cho đặc tả này

Các file bằng chứng chính:

- `../../experiments/oracle_root_cause_diagnostic/outputs/reports/oracle_root_cause_summary.json`
- `../../experiments/root_cause_ablation_suite/outputs/reports/compare_runs.csv`
- `../../experiments/root_cause_ablation_suite/outputs/reports/gradient_mass_summary.csv`
- `../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv`
- `../../experiments/objective_resolution_suite/outputs/reports/full_bootstrap_summary.csv`
- `../../experiments/objective_resolution_suite/outputs/reports/replicate_oracle_aggregate.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_primary_metrics.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_pooled_center_metrics.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/center_label_purity_summary.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/center_purity_lookup_report.json`
- `../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json`
- `../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv`
- `../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_influence.csv`
- `../../experiments/failure_b_resolution_suite/outputs/replicates/l0_l1_controls/reports/replicate_oracle_aggregate.csv`
- `../../experiments/l4_oracle_center_correction_pilot/outputs/reports/combined_oracle_center_pilot_primary_metrics.csv`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_primary_metrics.csv`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/combined_oc2_final_pilot_pooled_center_metrics.csv`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/oc2_pilot_history.json`
- `../../experiments/oc2_joint_oracle_full_model_pilot/outputs/reports/decision_summary.json`
