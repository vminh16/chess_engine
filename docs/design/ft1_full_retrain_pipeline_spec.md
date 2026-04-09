# FT1 Full-Retrain Pipeline Spec (2026-04-02)

## 1) Mục tiêu

Spec này mô tả đầy đủ pipeline huấn luyện mới FT1 (full retrain) trong [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py), bao gồm:

- luồng dữ liệu và luồng train/eval,
- công thức toán học của loss,
- metric và rule chọn checkpoint,
- lý do chọn hướng retrain từ bằng chứng thống kê đã có.

Toàn bộ số liệu bên dưới được trích từ report/cơ chế hiện có trong repo, không bổ sung số liệu ngoài.

## 2) Nguồn sự thật (source of truth)

- Pipeline train mới: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py)
- Định nghĩa variant L4_A1_plus_A2: [experiments/objective_resolution_suite/objective_resolution_helpers.py](../../experiments/objective_resolution_suite/objective_resolution_helpers.py#L356)
- Định nghĩa target remap + term loss cơ sở: [experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py](../../experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py#L168), [experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py](../../experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py#L906)
- Tổng hợp bằng chứng thực nghiệm: [experiment_journal.md](../research/experiment_journal.md#L640)

## 3) Cấu hình train FT1 mới

Các tham số default quan trọng:

- Epochs = 50, `main_batch_size` = 256, `clean_center_batch_size` = 32, `ambiguous_center_batch_size` = 64, `grad_accum_steps` = 2: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L52)
- LR = 1e-4, `min_lr` = 1e-5, `weight_decay` = 1e-4, `grad_clip_norm` = 1.0: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L58)
- Center-aux hệ số: `lambda_clean_center` = 0.20, `lambda_ambiguous_center` = 0.10, `aux_margin_weight` = 0.40, `aux_margin_y600` = 0.08, `aux_huber_delta` = 0.05, `aux_ramp_epochs` = 4: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L73)
- Gate: `midband_mae_rel_tol` = 0.05, `stable_slope_abs_tol` = 0.02: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L86)

## 4) Pipeline huấn luyện mới (step-by-step)

### 4.1 Runtime validation và input artifact

FT1 yêu cầu:

- split data train/val/test,
- checkpoint tham chiếu L4,
- pooled center bundle,
- oracle role bundle.

Kiểm tra path được thực hiện tại [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L184).

### 4.2 Lập bộ objective chính

FT1 lấy variant L4_A1_plus_A2 từ catalog objective suite: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L148), [experiments/objective_resolution_suite/objective_resolution_helpers.py](../../experiments/objective_resolution_suite/objective_resolution_helpers.py#L403).

L4 kế thừa full A1 với:

- `loss_mode = curvature_compensated`,
- `y_loss_alpha = 0.65`,
- `z_loss_beta = 0.0`,
- `z_huber_delta = 1.0`,
- `y_reweight_clip_max = 4.0`,
- thêm `sampler_mode = band_balanced`.

Nguồn: [experiments/objective_resolution_suite/objective_resolution_helpers.py](../../experiments/objective_resolution_suite/objective_resolution_helpers.py#L356), [experiments/objective_resolution_suite/objective_resolution_helpers.py](../../experiments/objective_resolution_suite/objective_resolution_helpers.py#L405).

### 4.3 Train loop

Mỗi micro-step:

1. Lấy mini-batch main từ shard train (band-balanced order).
2. Lấy mini-batch aux từ role bundle gồm clean center và ambiguous center.
3. Trộn thành `xb_mix = [xb_main; xb_aux]`.
4. Tính `main_terms`, `aux_terms`, tổng objective.
5. Tính gradient riêng cho backbone (main/aux), giải conflict bằng PCGrad nếu bật.
6. Cộng gradient vào `.grad` theo tích lũy (`grad_and_accum_steps`).
7. Đến bước step thì clip grad -> optimizer.step -> scheduler.step.

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L823), [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L998).

### 4.4 Eval theo epoch và checkpoint selection

Sau mỗi epoch:

- Eval trên split val (không dùng test để chọn epoch): [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1108)
- Tính gate pass theo midband+slope.
- Tính `center_score` để xếp hạng candidate.
- Lưu 3 checkpoint:
  - latest,
  - best_any (chỉ theo center score),
  - best_gate (center score + pass gate).

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1157), [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1238).

### 4.5 Đánh giá cuối cùng

Checkpoint được chọn ưu tiên `best_gate` -> `best_any` -> `latest`, sau đó đánh giá trên test:

- [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1238)
- [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1246)

## 5) Công thức toán học

### 5.1 Target remap giữa scale

Với target gốc theo y600 và scale mục tiêu $s$:

$$
z = \operatorname{atanh}(y_{600}), \quad y_s = \tanh\left(\frac{600}{s}z\right)
$$

Nguồn: [experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py](../../experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py#L168).

### 5.2 Main objective (L4-style)

Gọi $l$ là logits, $\hat{y}=\tanh(l)$, $y$ là target đã remap theo `variant.target_scale`:

- MSE term: $\text{mse}=(\hat{y}-y)^2$
- Logit residual:
$$
\Delta_z = l - \operatorname{atanh}(y_{\text{clamp}})
$$
- Z-Huber weighted:
$$
w_z = (1-y_{\text{clamp}}^2)^{\beta_z}, \quad z_{\text{term}} = w_z\,\operatorname{Huber}(\Delta_z;\delta_z)
$$
- Curvature compensation trên y-term:
$$
y_{\text{curv}} = (1-y_{\text{clamp}}^2)^2,
\quad
c = \operatorname{clip}\left(\frac{1}{y_{\text{curv}}+\varepsilon}, 1, c_{\max}\right),
\quad
\tilde{c} = \frac{c}{\mathbb{E}[c]}
$$
$$
y_{\text{term}} = \tilde{c}\,\text{mse}
$$
- Main per-sample:
$$
\ell_{\text{main}} = \alpha\,y_{\text{term}} + (1-\alpha)\,z_{\text{term}}
$$

Trong FT1, main objective được nhân thêm trọng số theo khoảng cách center của raw target:

$$
w_{\text{center}}(y_{src}) = w_{\min} + (1-w_{\min})\left(\operatorname{clip}\left(\frac{|y_{src}|}{\tau},0,1\right)\right)^p
$$

Với default: $\tau=0.10$, $w_{\alpha}=0.35$, $p=1.0$.

Main objective của batch:

$$
L_{\text{main}} = \frac{\sum_i w_{\text{center},i}\,\ell_{\text{main},i}}{\sum_i w_{\text{center},i}}
$$

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L286), [train_v2_TF1/ft1_colab_helpers.py](../../../../train_v2_TF1/ft1_colab_helpers.py#293), [experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py](../../experiments/root_cause_ablation_suite/root_cause_ablation_helpers.py#L906).

### 5.3 Aux objective (oracle center supervision)

Đặt mask theo role:

- clean center,
- ambiguous center.

Các thành phần:

$$
L_{clean}=\operatorname{Huber}(\hat{y}_{clean}-y^\text{oracle}_{clean};\delta_{aux}),
\quad
L_{amb}=\operatorname{Huber}(\hat{y}_{amb}-y^\text{oracle}_{amb};\delta_{aux})
$$
$$
L_{margin}=\mathbb{E}\left[\operatorname{ReLU}(|\hat{y}_{clean}|-m)^2\right]
$$
$$
L_{aux}=\lambda_c L_{clean}+\lambda_a L_{amb}+\lambda_m L_{margin}
$$

Với default:

- $\lambda_c=0.20$,
- $\lambda_a=0.10$,
- $\lambda_m=0.40$,
- $m=0.08$,
- $\delta_{aux}=0.05$.

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L338), [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L359).

### 5.4 Tổng objective theo epoch ramp

$$
L_{total} = L_{main} + s(e)\,L_{aux}
$$

Với ramp factor:

$$
s(e)=\min\left(1,\frac{e+1}{E_{ramp}}\right), \quad E_{ramp}=4
$$

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L753).

### 5.5 PCGrad cho backbone

Nếu $g_m$ (main backbone grad) và $g_a$ (aux backbone grad) xung đột ($g_m^\top g_a < 0$), aux grad được project:

$$
g_a' = g_a - \frac{g_m^\top g_a}{\|g_m\|^2+\epsilon}g_m
$$
$$
g_{shared}=g_m+g_a'
$$

Head gradient dùng tổng objective trực tiếp (không project với backbone).

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L413), [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L935), [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1008).

## 6) Metric đánh giá và rule chọn model

### 6.1 Gate metric (giữ A-side)

Gate pass nếu đồng thời:

$$
\text{oracle\_midband\_mae\_sum\_stable} \le \text{L4\_midband}\times(1+0.05)
$$
$$
\text{oracle\_stable\_0.7\_slope} \ge \text{L4\_slope}-0.02
$$

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L583).

### 6.2 Center ranking score (ưu tiên B-side)

$$
\text{center\_score} = \text{mae\_vs\_oracle} + 0.30 f_{0.1} + 0.20 f_{0.2} + 0.10\max(3,\text{amp\_ratio}-2.5)
$$

Trong đó $f_{0.1}$/$f_{0.2}$ là tỷ lệ false decisive ở ngưỡng 0.1/0.2.

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L574).

### 6.3 Các metric được log trong run

- Objective train: `train_main_objective`, `train_aux_objective`, `train_clean_loss`, `train_ambiguous_loss`, `train_margin_penalty`.
- Oracle/stability: `oracle_midband_mae_sum_stable`, `oracle_stable_0.7_slope`, `oracle_center_amp_ratio`, `oracle_center_false_0.1eq`, `oracle_center_false_0.2eq`, ...
- Pooled center: `pooled_center_mae`, `pooled_center_amp_ratio`, `pooled_center_false_0.1eq`, `pooled_center_false_0.2eq`, wrong-sign, spread.
- Role-level center: clean/ambiguous metrics riêng.
- Gradient telemetry: cosine pre/post, norm, conflict rate.
- BN sanity: running-mean/running-var summaries.

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L792), [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1133).

## 7) Lý do chọn retrain (bằng chứng thống kê cụ thể)

### 7.1 Failure A được cải thiện bởi hướng A1/L4

Từ [experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv](../../experiments/objective_resolution_suite/outputs/reports/full_primary_metrics.csv):

- Baseline:
  - `oracle_midband_mae_sum_stable` = 0.5987892032
  - `oracle_stable_0.7_slope` = 0.5751220597
- L4_A1_plus_A2:
  - `oracle_midband_mae_sum_stable` = 0.5709758064 (tốt hơn)
  - `oracle_stable_0.7_slope` = 0.6179156685 (tốt hơn)

Bootstrap trong [experiments/objective_resolution_suite/outputs/reports/full_bootstrap_summary.csv](../../experiments/objective_resolution_suite/outputs/reports/full_bootstrap_summary.csv):

- `prob_candidate_better_midband...` của L4 = 0.98275
- `prob_candidate_better_stable_0.7_slope` của L4 = 1.0

=> A-side có bằng chứng mạnh rằng objective hướng A1/L4 có hiệu quả.

### 7.2 Failure B chưa được giải quyết bằng objective A1/L4

Từ [experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_primary_metrics.csv](../../experiments/failure_b_resolution_suite/outputs/reports/combined_failure_b_primary_metrics.csv):

- `failure_b_score`:
  - A2_band_balanced = 0.6173303166
  - L0_control_hybrid = 0.6182334494
  - baseline = 0.6506076048
  - L4_A1_plus_A2 = 0.7329685091 (xấu hơn baseline)

- `pooled_center_amp_ratio`:
  - baseline = 4.3179921271
  - L4_A1_plus_A2 = 4.7992355317 (xấu hơn)

=> Cách giải A-side hiện tại không tự động sửa B-side.


### 7.3 Bằng chứng B1 (label impurity) và B2 (gradient interference)

B1, từ [experiments/failure_b_resolution_suite/outputs/reports/center_label_purity_summary.csv](../../experiments/failure_b_resolution_suite/outputs/reports/center_label_purity_summary.csv):

- Ngưỡng raw_center=0.02, oracle_center=0.05:
  - n=480, raw_center_count=60, clean_oracle_count=22
  - precision = 0.2666666667
- Ngưỡng raw_center=0.10, oracle_center=0.05:
  - raw_center_count=147
  - precision = 0.1496598639

B2, từ [experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv):

- cosine giữa `center_raw_0_005` và `mid_05_07`:
  - baseline_obj `cosine_all` = -0.6423080094
  - a1_obj `cosine_all` = -0.6804810178

Thêm bằng chứng ảnh hưởng từ [experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_summary.json):

- `center_probe_delta_mean_abs_pred_from_midbands`:
  - baseline_obj = 0.0023285704
  - a1_obj = 0.0031441112

=> B1 và B2 đều có tín hiệu định lượng, nên lý do retrain full dynamics (thay vì late polish) là hợp lý.

### 7.4 Why retrain from epoch 0

Tổng hợp trong [experiment_journal.md](../research/experiment_journal.md#L640):

- Failure A objective-side.
- B1 và B2 đều là thật.
- Late polish từ L4 không đáng tin cậy: [experiment_journal.md](../research/experiment_journal.md#L647).
- Hướng hợp lý: intervene from epoch 0, center-aware full run: [experiment_journal.md](../research/experiment_journal.md#L691), [experiment_journal.md](../research/experiment_journal.md#L754).

## 8) Caveat thống kê và cách diễn giải

- Pooled center bundle trong Failure B suite có kích thước nhỏ (`n = 22`): [experiment_journal.md](../research/experiment_journal.md#L140).
- Vì vậy, cần đọc kết quả theo hướng:
  - dùng gate để chặn regression A-side,
  - dùng center score + pooled center metrics để xếp hạng,
  - Đầu tiên replicate/seed để xác nhận độ ổn định.

## 9) Tiêu chí chấp nhận cho run FT1 mới (1đềxuất thao tác)

1. Qua hard gate so với L4 reference:
   - midband MAE sum stable không vượt ngưỡng,
   - slope 0.7 không giảm quá 0.02.
2. Trong các epoch qua gate, chọn epoch có center_score thấp nhất.
3. Báo cáo bắt buộc:
   - selected_checkpoint,
   - full_primary_metrics,
   - pooled_center_metrics,
   - clean/ambiguous_role_metrics,
   - gradient_telemetry và BN_sanity.

## 10) Artifacts output của pipeline

Run FT1 sẽ sinh các artifact chính:

- `reports/runtime_check.json`
- `reports/run_config.json`
- `reports/l4_reference.json`
- `reports/history.csv`
- `reports/step_history.csv`
- `reports/decision_summary.json`
- `reports/selected_checkpoint_eval.json`
-  `checkpoints/ckpt_latest.pt`
- `checkpoints/ckpt_best_any.pt`
- `checkpoints/ckpt_best_gate.pt`

Nguồn: [train_v2_TF1/ft1_colab_helpers.py](../../train_v2_TF1/ft1_colab_helpers.py#L1238).
