# Báo Cáo Failure A/B FT2 và Spec Thí Nghiệm Kế Tiếp (2026-04-16)

## 1) Phạm vi

Tài liệu này tổng hợp các kết quả quan trọng từ quá trình đánh giá FT1/FT2 và chốt một spec hành động có thể kiểm chứng ngay.

Câu hỏi trọng tâm:
1. Vì sao Failure A vẫn chưa được giải?
2. Vì sao Failure B không ổn định hoặc bị thoái lui trong FT2?
3. Giảm kích thước model có khả năng giảm A, B, hay cả hai không?
4. Nên chạy tiếp hướng nào, theo thứ tự nào, và điều kiện dừng/đi tiếp là gì?

## 2) Kết luận điều hành

1. Run FT2 hiện tại chưa vượt cổng A và chưa vượt FT1 best ở B.
2. Failure A hiện là blocker lớn nhất, do cả hai metric gate đều còn cách xa ngưỡng.
3. Failure B đã cải thiện mạnh bên trong run FT2 (so với epoch 0 FT2), nhưng FT2 best center score vẫn kém FT1 best center score.
4. Data provenance đã rõ: pipeline active là STM-label, lấy từ JSONL có lọc depth/knodes, không phải script depth=12 legacy.
5. Nhãn y=0 chính xác (xấp xỉ 10.99%) là mass-point thật từ cp=0, không phải sập số do số học.
6. Giảm model size là bước đầu hợp lý, rủi ro thấp, có khả năng cải thiện độ ổn định tối ưu và giảm áp lực conflict ở head; tuy nhiên không đảm bảo tự động vượt Failure A.

Khuyến nghị tổng quát: ưu tiên shrink-first (8b/128d), sau đó thêm sign-stratified sampling nếu A vẫn trượt, rồi mới test regime-separated head.

## 3) Định nghĩa Failure và công thức

### 3.1 Biến đổi label và POV

Biến đổi target active:

$$
y = \tanh\left(\frac{cp_{label}}{600}\right)
$$

Quy đổi POV:

$$
cp_{white} =
\begin{cases}
cp_{src}, & \text{nếu source POV là white} \\
cp_{src}, & \text{nếu source POV là stm và stm = white} \\
-cp_{src}, & \text{nếu source POV là stm và stm = black}
\end{cases}
$$

$$
cp_{label} =
\begin{cases}
cp_{white}, & \text{nếu label POV là white} \\
cp_{white}, & \text{nếu label POV là stm và stm = white} \\
-cp_{white}, & \text{nếu label POV là stm và stm = black}
\end{cases}
$$

Dẫn chứng: [data/process_data.ipynb](data/process_data.ipynb#L120), [data/process_data.ipynb](data/process_data.ipynb#L251), [data/process_data.ipynb](data/process_data.ipynb#L267)

### 3.2 Cổng gate cho Failure A

Theo logic gate FT2:

$$
M \le M_{ref}(1+\delta_M), \quad S \ge S_{ref} - \delta_S
$$

Trong đó:
1. $M = oracle\_midband\_mae\_sum\_stable$
2. $S = oracle\_stable\_0.7\_slope$
3. $\delta_M = 0.05$
4. $\delta_S = 0.02$

Dẫn chứng: [train_v3_FT2/ft2_colab_helpers.py](train_v3_FT2/ft2_colab_helpers.py#L106), [train_v3_FT2/ft2_colab_helpers.py](train_v3_FT2/ft2_colab_helpers.py#L1632)

Giá trị tham chiếu L4:
1. $M_{ref} = 0.5711541192433227$
2. $S_{ref} = 0.6177681497996689$

Dẫn chứng: [runs/dgrn_5m_ft2_t4_run1/reports/l4_reference.json](runs/dgrn_5m_ft2_t4_run1/reports/l4_reference.json#L1)

Ngưỡng số học:

$$
M_{max} = 0.5711541192433227 \times 1.05 = 0.5997118252054889
$$

$$
S_{min} = 0.6177681497996689 - 0.02 = 0.5977681497996689
$$

### 3.3 Công thức Failure B (center score)

Điểm center được dùng trong stack FT1/FT2:

$$
C = MAE_{center} + 0.30 \cdot F_{0.1} + 0.20 \cdot F_{0.2} + 0.10 \cdot \max(0, A - 2.5)
$$

Trong đó:
1. $MAE_{center}$ là pooled center MAE so với oracle
2. $F_{0.1}$ là false decisive rate tại ngưỡng 0.1 equivalent
3. $F_{0.2}$ là false decisive rate tại ngưỡng 0.2 equivalent
4. $A$ là amplitude ratio

Dẫn chứng: [train_v2_TF1/ft1_colab_helpers.py](train_v2_TF1/ft1_colab_helpers.py#L797)

## 4) Bằng chứng hiện tại nói gì

### 4.1 Trạng thái FT2 (run dgrn_5m_ft2_t4_run1)

Nguồn:
1. [runs/dgrn_5m_ft2_t4_run1/reports/history.csv](runs/dgrn_5m_ft2_t4_run1/reports/history.csv#L1)
2. [runs/dgrn_5m_ft2_t4_run1/reports/decision_summary.json](runs/dgrn_5m_ft2_t4_run1/reports/decision_summary.json#L1)

Quan sát:
1. Chưa pass gate: has_best_gate = false.
2. FT2 best cho A:
   - $M_{best} = 0.6728377645$ (vượt $M_{max}$ +0.073126, tương đương +12.19%)
   - $S_{best} = 0.4961447818$ (thấp hơn $S_{min}$ -0.101623)
3. Epoch cuối tệ hơn điểm best ở A:
   - $M_{last} = 0.7173413442$ (vượt gate +19.61%)
   - $S_{last} = 0.4259472177$ (thấp hơn gate -0.171821)
4. Bên trong FT2, B có cải thiện:
   - center score giảm từ 0.6230 xuống 0.3672 (giảm 41.06%)
   - epoch cuối 0.3691, gần mức tốt nhất nhưng có rebound nhẹ.

Xu hướng theo epoch (fit tuyến tính):
1. Midband metric: giai đoạn đầu cải thiện, giai đoạn cuối đảo chiều xấu (+0.005661/epoch ở epoch 6-9)
2. Slope metric: giai đoạn đầu cải thiện, giai đoạn cuối đảo chiều xấu (-0.015201/epoch ở epoch 6-9)
3. Center score: vẫn cải thiện nhưng tốc độ chậm và gần sát đáy.

### 4.2 So sánh FT1 best và FT2 best

Nguồn:
1. [runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/history.csv](runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/history.csv#L1)
2. [runs/dgrn_5m_ft2_t4_run1/reports/history.csv](runs/dgrn_5m_ft2_t4_run1/reports/history.csv#L1)

Giá trị best:
1. FT1 best: mid=0.541597, slope=0.572491, center=0.330885
2. FT2 best: mid=0.672838, slope=0.496145, center=0.367206

Delta (FT2 - FT1):
1. mid: +0.131241 (xấu hơn)
2. slope: -0.076346 (xấu hơn)
3. center: +0.036321 (xấu hơn)

Diễn giải:
1. Objective stack FT2 trong run hiện tại đang regress cả A và B so với FT1 best checkpoint.
2. Nếu tiếp tục dùng setup FT2 y nguyên, kỳ vọng thu được thông tin mới là thấp.

### 4.3 Tổng training exposure đã sử dụng

Tổng 4 run:
1. v2_run1: 9 epoch
2. v2_local_4gb_run1: 10 epoch
3. ft1_colab_pcgrad_run1: 18 epoch
4. ft2_t4_run1: 10 epoch

Tổng cộng: 47 epoch trên line 20b/256d.

### 4.4 Runtime và utilization

Dẫn chứng: [runs/dgrn_5m_ft2_t4_run1/reports/history.csv](runs/dgrn_5m_ft2_t4_run1/reports/history.csv#L1), [train_v3_FT2/train_ft2_colab.ipynb](train_v3_FT2/train_ft2_colab.ipynb#L118)

Số liệu throughput:
1. main_samples_per_sec median xấp xỉ 805.2
2. optimizer_steps_per_sec median xấp xỉ 1.578
3. Có một epoch outlier rất nhanh, cho thấy run có tính không đồng nhất (khả năng liên quan resume/state)

Notebook FT2 đã có nhánh stage data local trên Colab:
1. CHESS_STAGE_DATA_LOCAL mặc định true
2. Dùng rsync hoặc copytree sang /content/chess_engine_data/process

## 5) Data provenance đã được xác nhận

### 5.1 Pipeline active là JSONL có lọc depth, không phải script depth=12 legacy

Bằng chứng từ builder active:
1. FIXED_DEPTH=25, MIN_KNODES=50_000, DEPTH_POLICY=at_least
2. SOURCE_CP_POV=white, LABEL_POV=stm
3. TARGET_TOTAL tính động, chia từ SPLIT_RATIO

Dẫn chứng: [data/process_data.ipynb](data/process_data.ipynb#L113), [data/process_data.ipynb](data/process_data.ipynb#L120), [data/process_data.ipynb](data/process_data.ipynb#L139), [data/process_data.ipynb](data/process_data.ipynb#L459)

Bằng chứng script legacy (không phải active source của processed shards hiện tại):
1. hard-code depth=12
2. cp theo white POV rồi tanh(cp/600)

Dẫn chứng: [data/processing_data.py](data/processing_data.py#L144), [data/processing_data.py](data/processing_data.py#L152)

### 5.2 Tỷ lệ label y=0 chính xác là thật

Kết quả quét trực tiếp shard:
1. train: 10.9885%
2. val: 11.0462%
3. test: 10.9368%
4. toàn bộ: 10.9891%

Kiểm tra bổ sung:
1. Không có tiny nonzero gần 1e-7
2. Giá trị y nonzero nhỏ nhất map ngược đúng cp=1
3. Cụm near-zero map đúng vào cp integer (...,-3,-2,-1,0,1,2,3,...)

Kết luận: mass y=0 là mass-point cp=0 thật, không phải sai số làm tròn.

## 6) Tổng hợp root-cause

### 6.1 Interference tập trung mạnh ở head

Dẫn chứng: [experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv](experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv#L5)

Giá trị đại diện (baseline_obj):
1. center_raw_0_005 và mid_05_07:
   - backbone cosine: -0.2792
   - head cosine: -0.7239
2. near_center_005_02 và mid_02_05:
   - backbone cosine: -0.0966
   - head cosine: -0.7461

Diễn giải:
1. Độ xung đột ở head lớn hơn backbone rõ rệt cho cặp objective center-vs-mid quan trọng.
2. Đây là cơ sở mạnh để ưu tiên head redesign và/hoặc shrink-first.

### 6.2 Mass balancing cần nhưng chưa đủ

Dẫn chứng: [experiments/root_cause_ablation_suite/outputs/reports/gradient_mass_summary.csv](experiments/root_cause_ablation_suite/outputs/reports/gradient_mass_summary.csv#L1)

A2/B1 đã cải thiện phân bố gradient mass, nhưng FT2 vẫn trượt cổng A trong hướng chạy hiện tại.
Điều này cho thấy hướng gradient (directional coherence, sign/noise/interference) vẫn là điểm cần giải quyết tiếp.

## 7) Giảm model size có khả năng giảm Failure A/B không?

Trả lời ngắn:
1. Failure B: khả năng cải thiện có, mức tin cậy trung bình-khá.
2. Failure A: có thể cải thiện một phần, nhưng chưa đủ cơ sở để khẳng định sẽ pass gate chỉ nhờ shrink.

Lý do:
1. Xung đột head đang mạnh; model nhỏ hơn thường giảm over-amplification quanh center và giảm dao động tối ưu.
2. Tuy nhiên gate A phụ thuộc đồng thời midband MAE và slope; nếu hướng signal vẫn nhiễu nhiều, shrink đơn lẻ có thể chưa đủ.

### 7.1 Ước lượng tham số theo công thức kiến trúc

Theo DGRN v2 hiện tại:
1. stem conv+bn
2. B DFG blocks
3. ResidualGainValueHead

Công thức tổng quát:

$$
P(H,B) \approx P_{stem}(H) + B\cdot P_{block}(H) + P_{head}(H)
$$

Với:

$$
P_{stem}(H)=164H
$$

$$
P_{block}(H)=5.5H^2 + 3mH + 6H + 3m,\quad m=\max(8, H/8)
$$

$$
P_{head}(H)\approx 19.375H^2 + 11H + 2
$$

Ước lượng:
1. 20b/256d: 9,047,682 params
2. 8b/128d: 1,116,418 params
3. Tỷ lệ giảm: xấp xỉ 8.1 lần

Dẫn chứng kiến trúc: [model/architecture_v2/model.py](model/architecture_v2/model.py#L10), [model/architecture_v2/blocks.py](model/architecture_v2/blocks.py#L58), [model/architecture_v2/head.py](model/architecture_v2/head.py#L7)

Hàm ý:
1. 8b/128d là can thiệp lớn và đủ mạnh để test giả thuyết nhanh.
2. Đây không phải tinh chỉnh nhỏ.

## 8) Kế hoạch thí nghiệm kế tiếp (spec cụ thể)

### 8.1 Nguyên tắc

1. Mỗi phase chỉ thay đổi một biến chính.
2. Giữ nguyên metric và gate để so sánh công bằng.
3. Chốt trước điều kiện stop/go trước khi chạy.
4. Không trộn architecture + objective + data change trong run đầu tiên.

### 8.2 Biến kiểm soát cố định

1. Giữ nguyên target space và metric canonical y600.
2. Giữ nguyên gate định nghĩa như FT2 helper.
3. Giữ seed family nếu không có mục tiêu test variance.
4. Giữ split semantics để so sánh công bằng.

### 8.3 Thứ tự phase

Phase 1 (shrink-first baseline):
1. Model: 8b/128d
2. Head: giữ ResidualGainValueHead
3. Objective: FT1-style trước để có baseline sạch
4. Thời lượng: 12-15 epoch
5. Mục tiêu: xác định A và B có cùng đi đúng hướng không

Phase 2 (coherence objective-data):
1. Giữ 8b/128d
2. Thêm sign-stratified sampling (hoặc cơ chế tương đương)
3. Thời lượng: 10-12 epoch
4. Mục tiêu: cải thiện slope mà không tái tăng overconfidence quanh center

Phase 3 (tách chế độ ở head):
1. Giữ setup tốt nhất từ Phase 1/2
2. Đổi sang RegimeSeparatedHead
3. Thời lượng: 10-12 epoch
4. Mục tiêu: giảm xung đột hướng gradient ở head

Phase 4 (tùy chọn):
1. GroupNorm/normalization A-B test sau khi đã có bằng chứng từ 3 phase đầu
2. Chạy ngắn 6-8 epoch

### 8.4 Tiêu chí thành công

Điều kiện pass chính (bắt buộc cả 2):
1. $M \le 0.5997118252$
2. $S \ge 0.5977681498$

Điều kiện phụ:
1. center_score <= FT1 best center_score + 0.01 (mục tiêu <= 0.3409)
2. Không có đảo chiều xấu ở cuối run cho A metrics

Điều kiện fail-fast:
1. Sau epoch 8, nếu cả 2 gate gap vẫn lớn và trend xấu xác nhận liên tiếp 3 epoch
2. Hoặc center cải thiện nhưng A tiếp tục diverge khỏi gate

### 8.5 Ma trận run tối thiểu

Matrix M1 (để chạy ngay):
1. R1: 8b/128d, FT1 objective, current head, 15 epoch
2. R2: 8b/128d, FT1 objective + sign-stratified sampling, 12 epoch
3. R3: 8b/128d, objective tốt nhất từ R1/R2 + RegimeSeparatedHead, 12 epoch

Luật ra quyết định:
1. Nếu R1 đã đóng gate A gap và giữ B ổn, tiếp tục scale line R1.
2. Nếu R1 trượt A nhưng R2 cải thiện slope rõ, giữ intervention sampling.
3. Nếu R2 vẫn bất ổn và R3 cải thiện conflict + slope, giữ regime-separated head.

### 8.6 Spec scale data 5M lên 20M

Có thể làm bằng đổi tham số trong notebook active:
1. Đổi TARGET_TOTAL trong process_data.ipynb
2. Giữ SPLIT_RATIO
3. quota sẽ tự tính lại theo TARGET_TOTAL

Dẫn chứng: [data/process_data.ipynb](data/process_data.ipynb#L139), [data/process_data.ipynb](data/process_data.ipynb#L459)

Lưu ý vận hành:
1. Build time, I/O, storage sẽ tăng mạnh
2. Nên có bằng chứng shrink-first trước khi cam kết build 20M full

## 9) Ghi chú thực thi ngay

1. Notebook FT2 đã hỗ trợ stage local trên Colab.
2. Snapshot run local hiện có nhiều file trùng tên (history(1), decision_summary(1), ...), và có tình trạng thiếu run_config trong bộ artifact đang dùng; nên ưu tiên history.csv và decision_summary.json làm nguồn chính cho snapshot này.
3. architecture_v2 đã có SimplifiedGlobalHead và RegimeSeparatedHead, nhưng model wiring hiện tại vẫn dùng ResidualGainValueHead.

Dẫn chứng:
1. [train_v3_FT2/train_ft2_colab.ipynb](train_v3_FT2/train_ft2_colab.ipynb#L118)
2. [model/architecture_v2/head.py](model/architecture_v2/head.py#L78)
3. [model/architecture_v2/head.py](model/architecture_v2/head.py#L121)
4. [model/architecture_v2/model.py](model/architecture_v2/model.py#L5)
5. [model/architecture_v2/model.py](model/architecture_v2/model.py#L39)

## 10) Khuyến nghị cuối cùng

Hướng đề xuất:
1. Dừng mở rộng line FT2 20b/256d hiện tại.
2. Chạy ngay shrink-first baseline 8b/128d (12-15 epoch) với stop/go chặt theo A/B.
3. Sau đó mới mở nhánh sign-stratified và regime-separated.

Lý do:
1. Đã dùng 47 epoch trên line 20b/256d mà vẫn chưa pass A.
2. FT2 best hiện regress so với FT1 best ở cả A và B.
3. Shrink-first là nhánh nhanh, tín hiệu cao, ít confound nhất để kiểm chứng giả thuyết gốc.
