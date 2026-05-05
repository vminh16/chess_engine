# FT2 Full-Retrain Pipeline Spec (2026-04-10)

## 1) Mục tiêu

Spec này mô tả pha retrain mới `FT2` cho `architecture_v2`, với mục tiêu:

- giữ lại phần đã được chứng minh của `L4_A1_plus_A2` cho Failure A,
- sửa Failure B theo cách can thiệp từ `epoch 0`,
- tránh lặp lại trade-off của FT1, nơi center tốt lên nhưng A-side sụp dần,
- và đưa ra rule quyết định rõ ràng nếu run thành công hoặc thất bại.

Spec này chỉ dùng:

- số liệu đã có trong repo,
- định nghĩa toán học đã có trong code,
- và các nguồn lý thuyết gốc về multi-objective / multitask optimization.

Không có nhận định nào trong spec này dựa trên "cảm giác đúng".

## 2) Nguồn sự thật

Nguồn nội bộ:

- FT1 pipeline: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py)
- FT1 spec cũ: [ft1_full_retrain_pipeline_spec.md](./ft1_full_retrain_pipeline_spec.md)
- Journal thực nghiệm: [experiment_journal.md](../research/experiment_journal.md)
- Root-cause spec: [teacher_root_cause_spec_2026-03-28.md](../research/root_cause/teacher_root_cause_spec_2026-03-28.md), [teacher_root_cause_spec_2026-03-31.md](../research/root_cause/teacher_root_cause_spec_2026-03-31.md)
- Audit tổng thể: [training_audit_report.md](../reports/training_audit_report.md)
- FT1 run hiện tại: [history.csv](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/history.csv), [decision_summary.json](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/decision_summary.json), [l4_reference.json](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/l4_reference.json)
- Dữ liệu root cause B: [center_label_purity_summary.csv](../../experiments/failure_b_resolution_suite/outputs/reports/center_label_purity_summary.csv), [gradient_interference_cosines.csv](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv), [gradient_interference_summary.json](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json)
- Bundle hiện tại: [manifest.json](../../experiments/oc2_joint_oracle_full_model_pilot/outputs/cache/oracle_role_bundle/manifest.json), [manifest.json](../../experiments/failure_b_resolution_suite/outputs/cache/pooled_center_bundle/manifest.json)

Nguồn lý thuyết gốc:

- Sener & Koltun, *Multi-Task Learning as Multi-Objective Optimization*, NeurIPS 2018: <https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html>
- Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad), ICLR 2020 submission: <https://openreview.net/forum?id=HJewiCVFPB>
- Chen et al., *GradNorm*, ICML 2018: <https://proceedings.mlr.press/v80/chen18a.html>
- Kendall et al., *Multi-Task Learning Using Uncertainty to Weigh Losses*, CVPR 2018: DOI `10.1109/CVPR.2018.00781`, metadata: <https://dblp.org/rec/conf/cvpr/KendallGC18>
- Ioffe & Szegedy, *Batch Normalization*, ICML 2015: <https://research.google/pubs/batch-normalization-accelerating-deep-network-training-by-reducing-internal-covariate-shift/>

## 3) FT1 đã chứng minh gì và FT1 đã thất bại ở đâu

### 3.1. Những điểm FT1 đã chứng minh được

FT1 không phải thất bại hoàn toàn. Nó đã chứng minh ba điều quan trọng:

1. Can thiệp từ `epoch 0` là đúng hướng.
2. Failure B không thể được giải đáng tin bằng short late-polish.
3. Failure B có thể giảm mạnh khi center-aware supervision tham gia ngay trong full dynamics.

Bằng chứng:

- FT1 `epoch 0 -> 17` giảm `center_score` từ `0.8265` xuống `0.3309`: [history.csv](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/history.csv)
- `ckpt_best_any.pt` ở `epoch 17` đạt center tốt hơn rõ rệt so với `L4` reference: [decision_summary.json](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/decision_summary.json), [l4_reference.json](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/l4_reference.json)
- OC1/OC2 đều không sửa được B theo cách đủ tin cậy, dù OC2 giữ A tốt hơn: [experiment_journal.md](../research/experiment_journal.md)

### 3.2. FT1 thất bại ở điểm nào

FT1 không có checkpoint nào pass hard gate.

Từ [decision_summary.json](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/decision_summary.json):

- `has_best_gate = false`
- `best_gate_center_score = Infinity`

Ngưỡng gate FT1 giữ nguyên so với L4 reference:

- `oracle_midband_mae_sum_stable <= 0.5711766511 * 1.05 = 0.5997354836`
- `oracle_stable_0.7_slope >= 0.6176508786 - 0.02 = 0.5976508786`

Nhưng trong toàn bộ `epoch 0..17`, không có epoch nào thỏa đồng thời cả hai điều kiện trên.

### 3.3. Trade-off của FT1 là định lượng, không phải cảm giác

Từ [history.csv](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/history.csv):

- A-side tốt nhất ở `epoch 11`:
  - `oracle_midband_mae_sum_stable = 0.5415968614`
  - `oracle_stable_0.7_slope = 0.5724912691`
  - `center_score = 0.4597633874`
- B-side tốt nhất ở `epoch 17`:
  - `center_score = 0.3308846826`
  - nhưng `oracle_midband_mae_sum_stable = 0.7362273063`
  - và `oracle_stable_0.7_slope = 0.3802610674`

Tức là:

- từ `epoch 11 -> 17`, center tiếp tục tốt lên,
- nhưng slope giảm `0.5725 -> 0.3803`,
- midband MAE sum xấu đi `0.5416 -> 0.7362`.

Đây là dấu hiệu trực tiếp của một Pareto conflict giữa các objective, không phải dấu hiệu "train chưa đủ lâu".

### 3.4. Vì sao FT1 dễ rơi vào trade-off này

FT1 có ba điểm yếu cấu trúc đã được hỗ trợ bởi dữ liệu:

1. Raw-center labels quá bẩn.

Từ [ft1_full_retrain_pipeline_spec.md](./ft1_full_retrain_pipeline_spec.md) và [experiment_journal.md](../research/experiment_journal.md):

- `raw_center_count = 147`
- `precision = 0.1496598639`

Nghĩa là chỉ khoảng `15%` mẫu raw-center trong audit đó thực sự là clean center theo oracle rule.

2. Gradient conflict là có thật.

Từ [gradient_interference_cosines.csv](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv):

- cosine giữa `center_raw_0_005` và `mid_05_07`:
  - `baseline_obj = -0.6423080094`
  - `a1_obj = -0.6804810178`

Từ [gradient_interference_summary.json](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json):

- `center_probe_delta_mean_abs_pred_from_midbands`
  - `baseline_obj = 0.0023285704`
  - `a1_obj = 0.0031441112`

Điều này cho thấy update từ midband thực sự đẩy center prediction.

3. Aux bank hiện tại quá nhỏ để mang quá nhiều trách nhiệm.

Từ [manifest.json](../../experiments/oc2_joint_oracle_full_model_pilot/outputs/cache/oracle_role_bundle/manifest.json):

- `num_rows = 78`
- `center_anchor_count = 14`
- `center_ambiguous_count = 39`

Từ [manifest.json](../../experiments/failure_b_resolution_suite/outputs/cache/pooled_center_bundle/manifest.json):

- `n = 22`

Do đó, FT1 phải học từ một main distribution rất lớn và một aux signal rất nhỏ, trong khi weights aux lại là hằng số. Đây là đúng kiểu tình huống mà literature multitask coi là dễ mất cân bằng gradient.

### 3.5. Vì sao không quay lại fine-tune ngắn

Điểm này đã có bằng chứng nội bộ trực tiếp:

- OC1 thất bại.
- OC2 là negative test sạch hơn nhưng vẫn thất bại trong sửa B: [experiment_journal.md](../research/experiment_journal.md)
- stability-weighted near-zero fine-tune tăng nhẹ slope nhưng làm center safety xấu đi:
  - `false_0.1_0.3`: `10.62% -> 11.64%`
  - `false_0.2_0.4`: `5.32% -> 5.92%`
  - `center_spread_ratio`: `8.76 -> 9.10`
  - nguồn: [stability_weighted_test_compare.csv](../../experiments/stability_weighted_near_zero_finetune/outputs/reports/stability_weighted_test_compare.csv)

Kết luận: late fine-tune có thể sửa cục bộ một metric, nhưng thực nghiệm hiện có không ủng hộ nó như đường chính để sửa đồng thời A và B.

## 4) Tại sao FT2 vẫn phải là full rerun

FT2 vẫn phải rerun từ `epoch 0` vì ba lý do:

1. FT1 đã xác nhận rằng phần cần sửa nằm ở **training dynamics**, không chỉ ở checkpoint selection.

2. Checkpoint FT1 trade-off tốt nhất không còn trên đĩa.

Trong [checkpoints](../../runs/dgrn_5m_ft1_colab_pcgrad_run1/checkpoints), các periodic snapshot chỉ còn quanh `epoch 6..8`. Không có artifact của `epoch 11` hoặc `epoch 13`, tức là không thể tiếp tục đúng từ vùng trade-off tốt nhất của FT1.

3. Nếu root cause chứa shared-feature interference, thì late polish không đủ để viết lại biểu diễn đã hình thành sớm.

Đây là điểm được hậu thuẫn bởi:

- evidence nội bộ từ OC1/OC2,
- evidence gradient conflict ở Failure B suite,
- và lý thuyết multi-objective optimization:
  - weighted-sum đơn giản không bảo đảm tìm được nghiệm tốt khi task xung đột: Sener & Koltun 2018
  - conflict giữa gradient cần được xử lý trực tiếp ở level optimization: Yu et al. 2020

## 5) Các phương án đã cân nhắc

### 5.1. Phương án A: Single-head + full rerun + dynamic balancing

Đây là phương án được chọn cho FT2.

Ưu điểm:

- dùng được ngay với nhãn hiện có,
- giữ causal attribution rõ,
- bám sát các failure mode đã đo được.

Độ phức tạp:

- train time: vẫn là `O(P)` theo số tham số, nhưng constant factor tăng vì cần nhiều gradient bookkeeping cho các loss thành phần; thực tế gần với `2` forward và `3` gradient extraction cho shared backbone mỗi optimizer step,
- train memory: `O(P)`,
- inference: không đổi so với FT1, `O(P)`.

### 5.2. Phương án B: Multi-head `value + confidence/volatility`

Đây là phương án hợp lý về mặt nghiên cứu, nhưng không phải FT2 mainline.

Lý do:

- [project_report_vi.md](../reports/project_report_vi.md) đã đề xuất head confidence / volatility,
- [root_cause_summary.json](../../experiments/teacher_root_cause_lab/outputs/reports/root_cause_summary.json) và [near_zero_stability_report.json](../../experiments/teacher_root_cause_lab/outputs/reports/near_zero_stability_report.json) cho thấy near-zero stability có liên quan đến error thật,
- nhưng repo hiện chưa có large-scale dense labels cho volatility.

Độ phức tạp:

- train/inference: vẫn `O(P)` với thêm head nhỏ `O(H_conf)`,
- chi phí lớn nhất là tạo label proxy quy mô lớn, không phải bản thân head.

Kết luận:

- rất đáng làm nếu FT2 thành công một phần nhưng vẫn còn uncertainty problem,
- chưa phải đường chính cho run kế tiếp.

### 5.3. Phương án C: Residual-to-classical

Đây là ý tưởng sáng tạo đáng giữ lại, nhưng chưa phải FT2 core.

Lý do:

- repo hiện chưa có thí nghiệm residual-to-classical được lưu như source of truth,
- target của dự án không thuần material/static,
- nếu residual hóa sai, mô hình có thể học lệch mục tiêu engine.

Độ phức tạp:

- train/inference gần như `O(P)`,
- rủi ro chính không nằm ở compute mà ở semantics của target.

Kết luận:

- chỉ nên là nhánh nghiên cứu phụ sau FT2, không nên là đường chính của rerun rất tốn kém này.

## 6) Thiết kế FT2 được chọn

### 6.1. Những gì giữ nguyên

Giữ nguyên:

- model class `DGRNChessNetV2`
- input encoding hiện tại
- full train split hiện tại
- main objective họ `L4`
- band-balanced sampler của `L4_A1_plus_A2`
- hard gate A-side của FT1 để giữ comparability

Lý do:

- Failure A đã có lời giải tốt nhất hiện tại trong họ `A1/L4`: [objective_resolution_suite](../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv)
- [training_audit_report.md](../reports/training_audit_report.md) không ủng hộ giả thuyết backbone là bottleneck gốc

### 6.2. Những gì thay đổi trong FT2

FT2 đổi đúng ba điểm:

1. **Aux batches không còn đi chung BN path với main batch**
2. **Aux weights không còn là hằng số**
3. **Role bundle được xây lại và có split train/val riêng**

Ba thay đổi này tương ứng trực tiếp với ba vấn đề của FT1:

- BN contamination risk,
- static loss balancing under conflict,
- aux data quá nhỏ và không có validation tách riêng.

## 7) Dữ liệu FT2

### 7.1. Main data

Giữ nguyên main data:

- train/val/test shards tại `data/process`
- sampler `band_balanced` như `L4_A1_plus_A2`

Main loss trên raw labels vẫn cần giữ vì đây là nguồn giám sát dày nhất. Không có bằng chứng nào trong repo cho thấy bỏ raw labels là hợp lý.

### 7.2. Oracle role bundle mới

FT2 yêu cầu rebuild `oracle_role_bundle` trước khi train.

Rule:

- dùng cùng logic tạo candidate như pipeline hiện có,
- nhưng không giới hạn mining ở `8` shards như manifest cũ,
- lưu hai split:
  - `aux_train`
  - `aux_val`
- split theo position identity, không cho cùng vị trí xuất hiện ở cả hai split.

Mục tiêu của `aux_val` không phải để tối ưu loss, mà để phát hiện overfit vào chính aux bank nhỏ.

### 7.3. Pooled center bundle

Giữ [pooled_center_bundle](../../experiments/failure_b_resolution_suite/outputs/cache/pooled_center_bundle/manifest.json) làm **evaluation-only bundle**.

Không train trên bundle này vì:

- kích thước `n = 22` quá nhỏ,
- nếu train vào nó thì metric center chính sẽ bị contaminated.

## 8) Mô hình và BatchNorm policy

### 8.1. Kiến trúc

Giữ nguyên `DGRNChessNetV2`.

FT2 không sửa backbone lớn vì chưa có bằng chứng rằng backbone là root cause chính. Đổi backbone ở thời điểm này sẽ làm mất khả năng kết luận nhân-quả từ run mới.

### 8.2. BN-safe aux handling

FT1 trộn `xb_main` và `xb_aux` thành `xb_mix`, rồi forward một lần: [ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py)

FT2 bỏ cách này.

FT2 dùng hai pass:

1. `main forward` ở `train mode`
2. `aux forward` với BatchNorm running statistics bị freeze

Lý do toán học:

BatchNorm dùng thống kê minibatch hoặc running averages để chuẩn hóa kích hoạt. Nếu minibatch aux rất nhỏ và có phân phối khác main data, estimator của mean/variance bị lệch khỏi phân phối main. Với aux bank chỉ vài chục mẫu retained, bias này không thể xem là nhiễu nhỏ.

Nguồn lý thuyết cho vai trò của batch statistics: Ioffe & Szegedy 2015.

Điểm này cũng khớp với cảnh báo còn tồn dư trong [experiment_journal.md](../research/experiment_journal.md): oracle bank nhỏ có thể làm lệch running statistics.

## 9) Loss FT2

### 9.1. Main loss giữ nguyên từ FT1

Giữ nguyên `L4` main loss:

\[
L_{\text{main}}
= \frac{\sum_i w_{\text{center}}(y_i)\, \ell_{L4}(f(x_i), y_i)}{\sum_i w_{\text{center}}(y_i)}
\]

trong đó:

\[
w_{\text{center}}(y)
= w_{\min} + (1-w_{\min}) \cdot \min\left(1, \frac{|y|}{\tau}\right)^\gamma
\]

với:

- `w_min = main_center_min_weight`
- `tau = main_center_tau_y600`
- `gamma = main_center_weight_power`

Đây là đúng công thức đang có ở [_main_center_weights](../../train_v2_TF1/ft1_colab_helpers.py).

Giữ nguyên vì:

- Failure A đã được chứng minh là objective-side,
- và `L4_A1_plus_A2` vẫn là objective tốt nhất hiện có cho A-side.

### 9.2. Aux losses giữ nguyên về semantics

Giữ nguyên ba thành phần aux:

\[
L_{\text{clean-fit}}
= \operatorname{Huber}(f(x)-y_{\text{oracle}};\delta)
\]

\[
L_{\text{margin}}
= \operatorname{ReLU}(|f(x)|-m)^2
\]

\[
L_{\text{amb}}
= \operatorname{Huber}(f(x)-y_{\text{oracle}};\delta)
\]

Đặt:

\[
L_{\text{clean-total}}
= L_{\text{clean-fit}} + \beta_{\text{margin}} L_{\text{margin}}
\]

với:

- \(\delta = \text{aux\_huber\_delta}\)
- \(m = \text{aux\_margin\_y600}\)
- \(\beta_{\text{margin}} = \text{aux\_margin\_weight}\)

Lý do:

- clean-center cần vừa kéo về oracle vừa chặn biên độ quá lớn,
- ambiguous-center chỉ nên kéo mềm, không nên ép về 0 một cách cực đoan.

### 9.3. Điểm thay đổi chính: dynamic balancing thay cho fixed weights

FT1 dùng:

\[
L_{\text{FT1}}
= L_{\text{main}}
+ 0.2\,L_{\text{clean-fit}}
+ 0.1\,L_{\text{amb}}
+ 0.4\,L_{\text{margin}}
\]

Đây là weighting tĩnh.

FT2 thay bằng weighting động kiểu GradNorm trên shared backbone parameters.

Đặt các task:

\[
t \in \{\text{main}, \text{clean-total}, \text{amb}\}
\]

với weighted loss:

\[
L = \sum_t \lambda_t L_t, \qquad \lambda_t > 0
\]

Đặt:

\[
G_t = \left\| \nabla_{W_s} (\lambda_t L_t) \right\|_2
\]

trên shared parameters \(W_s\) của `stem + blocks`.

Đặt relative inverse training rate:

\[
r_t =
\frac{L_t(k) / L_t(0)}{\frac{1}{T}\sum_j L_j(k)/L_j(0)}
\]

GradNorm cập nhật \(\lambda_t\) để \(G_t\) bám theo:

\[
\bar{G}\, r_t^\alpha
\]

thay vì để một task chi phối gradient quá mức.

FT2 policy:

- trong `4` epoch đầu, giữ weighting cố định như FT1 để hoàn tất `aux_ramp_epochs`
- từ sau `epoch 4`, kích hoạt GradNorm
- khởi tạo \((\lambda_{\text{main}}, \lambda_{\text{clean-total}}, \lambda_{\text{amb}}) = (1.0, 0.2, 0.1)\)
- renormalize weights sau mỗi update để:

\[
\lambda_{\text{main}} + \lambda_{\text{clean-total}} + \lambda_{\text{amb}} = 1.3
\]

Lý do chọn hướng này:

- FT1 cho thấy fixed weights tạo trade-off rõ ràng giữa A và B,
- GradNorm được thiết kế đúng cho trường hợp task học ở tốc độ khác nhau: Chen et al. 2018.

### 9.4. Giữ PCGrad trên backbone

FT2 vẫn giữ PCGrad cho backbone sau khi đã reweight loss.

Logic:

- GradNorm giải quyết mất cân bằng về gradient magnitudes,
- PCGrad giải quyết conflict về hướng gradient.

Hai vấn đề này là khác nhau:

- magnitude mismatch,
- directional conflict.

FT1 telemetry đã cho thấy directional conflict vẫn tồn tại ở backbone. Vì vậy, bỏ PCGrad ở FT2 là không có cơ sở.

## 10) Train loop FT2

Một optimizer step FT2 có dạng:

1. Lấy `xb_main, yb_main` từ main sampler.
2. Lấy `xb_aux, yb_aux, role_aux` từ `aux_train`.
3. Forward `xb_main` ở `train mode`.
4. Forward `xb_aux` với BN running stats freeze.
5. Tính:
   - `L_main`
   - `L_clean_total`
   - `L_amb`
6. Cập nhật `lambda_t` bằng GradNorm trên shared backbone.
7. Lấy gradient từng task trên shared backbone.
8. Chạy PCGrad trên backbone gradients.
9. Cộng head gradients theo weighted-sum thông thường.
10. Clip grad, optimizer step, scheduler step.

Điểm khác FT1:

- không concatenate main và aux vào cùng một batch,
- không dùng aux weights cố định suốt run.

## 11) Checkpointing và logging

FT2 bắt buộc sửa chính sách checkpoint.

FT1 đã cho thấy checkpoint tốt nhất theo A-side nằm ở `epoch 11`, checkpoint tốt nhất theo B-side nằm ở `epoch 17`, nhưng repo không còn artifact `epoch 11/13`.

FT2 phải lưu:

- `latest`
- `best_gate`
- `best_any`
- `best_pareto_A`
- `best_pareto_B`
- checkpoint cuối mỗi epoch

`best_pareto_A`:

- minimize `oracle_midband_mae_sum_stable`
- maximize `oracle_stable_0.7_slope`

`best_pareto_B`:

- minimize `center_score`

Mục đích:

- không để mất checkpoint ở vùng trade-off tốt nhất,
- cho phép warm-start nghiên cứu ở pha sau nếu cần.

## 12) Rule chấp nhận FT2

### 12.1. Hard gate giữ nguyên như FT1

Một epoch được xem là `gate-pass` nếu:

\[
\text{oracle\_midband\_mae\_sum\_stable}
\le
1.05 \times \text{L4 reference}
\]

và

\[
\text{oracle\_stable\_0.7\_slope}
\ge
\text{L4 reference} - 0.02
\]

Lý do:

- giữ comparability với FT1,
- tránh deploy một checkpoint center tốt nhưng A-side hỏng.

### 12.2. Rule chọn checkpoint

Rule chọn checkpoint FT2:

1. Nếu có `gate-pass` epoch:
   - chọn epoch có `center_score` thấp nhất trong tập `gate-pass`.
2. Nếu không có `gate-pass`:
   - run được xem là chưa thành công,
   - nhưng vẫn phải lưu toàn bộ frontier để chẩn đoán.

### 12.3. Điều kiện gọi là "thành công khoa học"

FT2 được xem là thành công về mặt khoa học nếu đồng thời:

1. có ít nhất một `gate-pass` epoch,
2. checkpoint được chọn có `center_score < L4 center_score`,
3. ít nhất một trong hai metric center chính cùng tốt hơn L4:
   - `pooled_center_mae`
   - `oracle_center_amp_ratio`

Điều kiện (3) tồn tại vì pooled center bundle nhỏ, nên không nên diễn giải chỉ từ một metric center duy nhất.

## 13) Nếu FT2 thành công hoặc thất bại thì làm gì

### 13.1. Nếu FT2 thành công

Bước tiếp theo:

1. chạy benchmark search fixed nodes/time so với `L4_A1_plus_A2`
2. nếu search không tụt, khóa FT2 thành teacher mới
3. chỉ sau đó mới chuyển sang:
   - confidence head,
   - distillation,
   - quantization

Lý do: [training_audit_report.md](../reports/training_audit_report.md) đã khuyến nghị giữ `architecture_v2` như teacher candidate trước khi distill.

### 13.2. Nếu FT2 thất bại kiểu 1: không có gate-pass, nhưng center cải thiện mạnh

Diễn giải:

- hướng center-aware từ epoch 0 là đúng,
- nhưng objective balancing vẫn chưa giữ được A-side.

Hướng tiếp theo:

- chuyển sang `value + confidence/volatility` head,
- tạo proxy stability labels ở quy mô lớn hơn,
- dùng head phụ để học khi nào evaluator không nên tự tin.

### 13.3. Nếu FT2 thất bại kiểu 2: A-side giữ được, center không cải thiện

Diễn giải:

- objective anchor cho A là đúng,
- nhưng aux data cho B chưa đủ sạch hoặc chưa đủ mạnh.

Hướng tiếp theo:

- mở rộng oracle role bundle trước,
- không đổi backbone ngay,
- kiểm tra lại tiêu chí mining cho `clean_center` và `center_ambiguous`.

### 13.4. Nếu FT2 thất bại kiểu 3: cả A và B đều không tốt hơn

Diễn giải:

- có khả năng design hiện tại của single-head đã chạm giới hạn,
- hoặc label semantics đang lệch mục tiêu engine quá xa.

Khi đó mới hợp lý để mở nhánh:

- `value + confidence/volatility` head,
- hoặc residual/teacher decomposition có điều kiện.

## 14) Các ý tưởng sáng tạo đáng giữ nhưng chưa đủ bằng chứng

Những ý tưởng dưới đây **không** là core FT2, nhưng nên được ghi lại để mở nhánh sau:

### 14.1. Confidence / volatility head

Ý tưởng:

- output chính: `value`
- output phụ: `confidence` hoặc `stability`

Mục tiêu:

- search biết khi nào nên tin NN,
- khi nào nên dựa nhiều hơn vào search/classical.

Điểm mạnh:

- khớp với [project_report_vi.md](../reports/project_report_vi.md)
- khớp với evidence near-zero stability trong `teacher_root_cause_lab`

Điểm yếu:

- chưa có dense labels quy mô đủ lớn.

### 14.2. Piecewise residual-to-classical

Ý tưởng:

- chỉ học residual trên decisive / non-center bands,
- không residual hóa mù quáng trên toàn phân phối.

Điểm mạnh:

- nếu target có thành phần classical lớn ở tails, residualization có thể giảm variance cần học.

Điểm yếu:

- hiện repo chưa có thực nghiệm đủ mạnh để xác nhận residualization không làm sai semantics ở center.

### 14.3. Task-specific adapters thay vì đổi backbone toàn phần

Ý tưởng:

- giữ shared backbone,
- thêm adapter nhỏ cho center-sensitive supervision,
- thay vì mở hai full head độc lập.

Điểm mạnh:

- ít phá vỡ kiến trúc hiện tại,
- cho phép mức disentanglement vừa phải.

Điểm yếu:

- chưa có thí nghiệm trong repo,
- dễ làm attribution khó hơn nếu đưa vào quá sớm.

## 15) Tóm tắt quyết định

FT2 được chốt như sau:

- rerun toàn bộ từ `epoch 0`,
- giữ `DGRNChessNetV2`,
- giữ main objective họ `L4`,
- rebuild oracle role bundle và tách `aux_train/aux_val`,
- bỏ mixed main+aux BN path,
- thay fixed aux weights bằng dynamic balancing kiểu GradNorm,
- giữ PCGrad trên backbone,
- giữ gate FT1 để so sánh công bằng với `L4`,
- lưu toàn bộ frontier checkpoint để không lặp lại mất artifact như FT1.

Lý do rerun lại toàn bộ không phải vì FT1 vô ích.

Ngược lại, chính FT1 đã cho bằng chứng mạnh nhất rằng:

- đúng là phải sửa ở full dynamics,
- nhưng recipe FT1 hiện tại chưa đủ để giữ đồng thời A và B.

FT2 vì vậy không phải "thử lại FT1".

FT2 là run mới với:

- cùng foundation đúng,
- nhưng sửa đúng ba điểm causal nhất mà FT1 đã phơi bày.
