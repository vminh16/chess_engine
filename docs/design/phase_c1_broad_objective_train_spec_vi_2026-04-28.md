# Phase C1 Broad Objective Refresh - Spec train mới sau Phase B

Ngày: 2026-04-28

## Kết luận điều hành

Phase B đã đủ chứng cứ để dừng hướng `B3/B4/train lâu hơn` theo cấu hình cũ. B1/B2 chứng minh `SimplifiedGlobalHead` có thể cải thiện oracle proxy hẹp, nhưng full-test cho thấy proxy đó không đại diện cho bài toán tổng quát. C1 mở một phase train mới từ đầu, không fine-tune checkpoint B, với ba thay đổi bắt buộc:

1. Sampling chính chuyển về `random` theo phân phối train tự nhiên thay vì `band_balanced` hoặc `sign_stratified`.
2. Gradient steering từ oracle role bundle 78 rows bị tắt trong objective; bundle chỉ còn phục vụ eval tương thích.
3. Checkpoint selection chuyển sang broad-validation score dựa trên các metric test-facing có độ tin cậy cao: overall MSE, near-center MSE, center false-decisive, midband/decisive MSE và absolute calibration gap.

`best_gate` trong C1 không còn nghĩa Phase B oracle gate. Trong C1, `best_gate` nghĩa là checkpoint vượt broad-validation gate so với L4 reference trong validation sample lớn hơn.

## Artifact Phase B đã chứng minh

Các số dưới đây đến từ artifact local đã đọc trong `runs/` và `evaluation/phase_b_offline_benchmark/outputs/`.

### B1

- Run: `dgrn_5m_phaseb_b1_sglobal_16b_256d_band_run1`.
- Model: 16 blocks / 256 hidden / `SimplifiedGlobalHead`.
- Sampling: `band_balanced`.
- Full test: 500000 mẫu, 10 shard, `benchmark_scope.is_full_test = true`.
- Oracle proxy:
  - `oracle_midband_mae_sum_stable = 0.5497264515`, tốt hơn L4 `0.5709758064`.
  - `oracle_stable_0.7_slope = 0.6110776455`, qua gate cũ.
- Broad/core fail:
  - `overall_mse = 0.0726871738` vs L4 `0.0695456346`.
  - `overall_mae = 0.1905413400` vs L4 `0.1798353451`.
  - `overall_pearson = 0.7570957660` vs L4 `0.7744795280`.
  - `center_false_decisive_0.1eq = 0.5423835662` vs L4 `0.4909642007`.
  - `center_score = 0.5880825006` vs L4 `0.5252589044`.
  - `center_spread_ratio = 9.6456781875`.

### B2

- Run: `dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1`.
- Model: 16 blocks / 256 hidden / `SimplifiedGlobalHead`.
- Sampling: `sign_stratified`.
- Full test: 500000 mẫu, 10 shard, `benchmark_scope.is_full_test = true`.
- Oracle proxy:
  - `oracle_midband_mae_sum_stable = 0.5581147603`, vẫn tốt hơn L4 `0.5709758064`.
  - `oracle_stable_0.7_slope = 0.6045859235`, vẫn qua gate cũ.
- Broad/core fail nặng hơn B1:
  - `overall_mse = 0.0758542925` vs L4 `0.0695456346`.
  - `overall_mae = 0.1913288635` vs L4 `0.1798353451`.
  - `overall_pearson = 0.7534007623` vs L4 `0.7744795280`.
  - `test_mse_0.1eq = 0.0565500810` vs L4 `0.0399272962`.
  - `center_false_decisive_0.1eq = 0.5267762764` vs L4 `0.4909642007`.
  - `center_score = 0.7731005988`.
  - `center_spread_ratio = 11.8069376124`.
- Calibration pattern:
  - `|y| <= 0.1`: `mean_abs_y ~ 0.0314`, `mean_abs_p ~ 0.1705`.
  - band `0.6-0.7`: `mean_y ~ 0.6518`, `mean_p ~ 0.3755`.

### Oracle role bundle

- Manifest local: `experiments/oc2_joint_oracle_full_model_pilot/outputs/cache/oracle_role_bundle/manifest.json`.
- `num_rows = 78`.
- 39 clean center, 39 ambiguous center.
- Trong Phase B, aux batch B2 thực tế lặp lại support rất hẹp này nhiều lần mỗi epoch. Vì support chỉ 78 rows, nó không đại diện cho full-test center behavior dựa trên hơn 150k center samples trong 500k full test.

## Phân tích toán học của failure

Phase B tối ưu objective xấp xỉ:

```text
J_B(theta) =
E_{Q_main}[w_center(y) * L_L4(f_theta(x), y)]
+ alpha(epoch) * E_{Q_aux78}[L_aux(f_theta(x), y_oracle, role)]
```

Trong đó:

- `Q_main` là phân phối sampler nhân tạo (`band_balanced` hoặc `sign_stratified`), không phải phân phối test tự nhiên `P_test`.
- `w_center(y)` trong L4 giảm trọng số gần center: `main_center_min_weight = 0.35` khi `|y|` gần 0.
- `Q_aux78` là empirical distribution trên 78 oracle rows, support quá hẹp.
- PCGrad chỉ xử lý backbone; head vẫn nhận tổng gradient trực tiếp. Artifact cũ đã ghi nhận conflict mạnh nhất nằm ở head.

Risk cần giảm mới là:

```text
R(theta) = E_{P_test}[MSE/MAE/calibration/center false decisive/sign behavior]
```

Nếu `Q_main != P_test` và `support(Q_aux78)` quá hẹp, thì giảm `J_B` không buộc giảm `R`. B2 là bằng chứng thực nghiệm trực tiếp: train objective giảm đều đến cuối run, nhưng broad full-test metrics xấu dần, đặc biệt near-center và calibration.

## Vì sao không ưu tiên B3/B4 hay train lâu hơn

B3 thay head sang `RegimeSeparatedHead` nhưng vẫn giữ logic objective/selection cũ nếu chạy theo spec Phase B ban đầu. Điều này có expected value thấp vì failure lớn nhất sau B1/B2 không phải thiếu head branch, mà là mismatch giữa objective/gate và metric tổng quát.

B4 là scale confirmation theo spec Phase B, không phải exploratory. Vì chưa có recipe 16/256 vượt broad gate, B4 sẽ chủ yếu scale một recipe chưa đúng objective.

Train lâu hơn không có cơ sở mạnh trong artifact hiện tại. Deep double descent và grokking có thể xảy ra trong một số setting, nhưng artifact Phase B không có dấu hiệu broad-test recovery. B2 cho thấy objective giảm trong khi full-test center/calibration xấu; đây giống proxy overfitting hơn late generalization.

## Đối chiếu lý thuyết ngoài

Các nguồn mạnh đã đối chiếu:

- Nakkiran et al., "Deep Double Descent": double descent có thể xảy ra theo model size hoặc epoch, nhưng không bảo đảm khi objective/validation đang lệch khỏi deployment risk. Link: https://arxiv.org/abs/1912.02292
- Power et al., "Grokking": late generalization được quan sát trên small algorithmic datasets, không phải bằng chứng để kỳ vọng breakthrough trong noisy chess regression khi proxy metric đang sai. Link: https://arxiv.org/abs/2201.02177
- Yu et al., "Gradient Surgery for Multi-Task Learning": gradient interference là vấn đề thật trong multi-task optimization; PCGrad xử lý conflict task-gradient, nhưng Phase B chỉ project backbone trong khi head conflict là rủi ro chính. Link: https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
- Sener & Koltun, "Multi-Task Learning as Multi-Objective Optimization": multi-task loss nên được xem là bài toán Pareto/multi-objective, không phải cộng loss proxy tùy ý nếu metric mục tiêu khác. Link: https://arxiv.org/abs/1810.04650
- Sugiyama et al., "Covariate Shift Adaptation by Importance Weighted Cross Validation": khi train/eval distribution lệch, selection/validation cần xử lý lệch phân phối; nếu không, model selection có bias. Link: https://www.jmlr.org/beta/papers/v8/sugiyama07a.html
- Guo et al., "On Calibration of Modern Neural Networks": calibration không tự đảm bảo bởi accuracy/loss thấp; cần đo và sửa trực tiếp. Link: https://proceedings.mlr.press/v70/guo17a.html
- Hinton et al., "Distilling the Knowledge in a Neural Network": distillation truyền hành vi teacher sang student; teacher sai calibration/center sẽ làm student kế thừa lỗi. Link: https://research.google/pubs/pub44873

## Objective C1

C1 dùng model 16/256 + `SimplifiedGlobalHead`, nhưng train from scratch và đổi objective. Với `p = tanh(z)` và target canonical `y`, objective mỗi batch là:

```text
J_C1(theta) =
mean Huber_delta_y(p - y)
+ lambda_z * mean Huber_delta_z(z - atanh(y))
+ lambda_center * mean_{|y| <= tau_center} ReLU(|p| - margin_center)^2
+ lambda_cal * mean_b (mean_b |p| - mean_b |y|)^2
```

Default hiện tại:

- `sampling_mode = random`.
- `clean_center_batch_size = 0`.
- `ambiguous_center_batch_size = 0`.
- `lambda_clean_center = 0`.
- `lambda_ambiguous_center = 0`.
- `aux_margin_weight = 0`.
- `broad_y_huber_delta = 0.10`.
- `broad_z_huber_weight = 0.20`.
- `broad_z_huber_delta = 1.00`.
- `broad_center_tau_y600 = 0.05`.
- `broad_center_pred_margin_y600 = 0.10`.
- `broad_center_margin_weight = 2.00`.
- `broad_abs_calibration_weight = 0.50`.
- `broad_abs_calibration_edges_y600 = (0.00, 0.05, 0.10, 0.20, 0.50, 0.70)`.

Lý do:

- Huber y-space giữ objective gần deployment MSE/MAE nhưng giảm outlier instability.
- z-space residual giữ gradient cho band lớn, tránh chỉ bóp center mà bỏ decisive band.
- Center margin đánh trực tiếp vào lỗi Phase B: over-amplify near-center.
- Absolute calibration penalty đánh trực tiếp vào pattern B2: center quá lớn, decisive band quá nhỏ.
- Random sampling giảm mismatch `Q_main != P_test`; nếu cần balanced sampler sau này thì phải importance-correct về phân phối deployment.

## Selection/Gate C1

Mỗi epoch eval trên validation lớn hơn Phase B:

- `val_num_shards = 4`.
- `val_max_samples = 200000`.

Broad score:

```text
score =
(1.00 overall_mse_ratio
+1.25 mse_0.1_ratio
+0.75 mse_0.2_ratio
+0.75 mse_0.5_ratio
+0.50 mse_0.7_ratio
+1.25 center_false_0.1_ratio
+0.75 center_false_0.2_ratio
+0.75 abs_cal_ratio) / 7.00
```

Ratio là candidate metric chia cho L4 reference metric cùng split/eval. Thấp hơn tốt hơn.

Broad gate pass nếu:

- `overall_mse <= L4 * 1.02`.
- `test_mse_0.1eq <= L4 * 1.02`.
- `test_center_false_0.1eq <= L4 + 0.01`.
- `test_max_midband_abs_cal_gap <= L4 * 1.05`.

Checkpoint policy để tương thích offline benchmark:

- `ckpt_best_any.pt`: best broad score, kể cả chưa pass gate.
- `ckpt_best_gate.pt`: best broad score trong các epoch pass broad gate.
- `ckpt_latest.pt`: periodic/resume checkpoint.

## Colab/resume safety

C1 kế thừa vòng train resume-safe từ Report1:

- Atomic save bằng helper hiện có.
- `ckpt_latest.pt` periodic mỗi 20 phút.
- Resume giữa shard bằng `resume_shard_index` và `resume_next_start`.
- Lưu RNG state Python/NumPy/Torch/CUDA.
- Resume signature kiểm tra `model_cfg`, `train_cfg`, `gate_cfg`; nếu đổi config nhưng dùng cùng run name, helper fail-fast.
- Autotune profile GPU T4 được giữ nhưng aux batch bị ép về 0 để không lặp lại oracle 78-row bundle.

## Kill criteria

Chạy C1 không quá 12 epoch cho pilot đầu.

Dừng sớm thủ công nếu sau epoch 4-6:

- `broad_score` không giảm so với epoch 1-2, hoặc
- `test_center_false_0.1eq` vẫn cao hơn L4 quá `0.03`, hoặc
- `test_mse_0.1eq` cao hơn L4 quá 10%, hoặc
- `overall_mse` và `test_max_midband_abs_cal_gap` cùng xấu hơn L4.

Không mở B4/scale-up nếu `ckpt_best_gate.pt` không tồn tại hoặc offline full-test của `best_any` không cải thiện ít nhất center false-decisive và overall MSE so với L4.

## Promotion rule

Sau train, bắt buộc chạy offline benchmark full test 500000 mẫu / 10 shard bằng policy `best_gate` nếu có, nếu không thì `best_any`.

Không promote nếu chỉ oracle proxy tốt. Promotion phải dựa trên:

- `overall_mse`, `overall_mae`, `overall_pearson`.
- `test_mse_0.1eq`, `test_mse_0.2eq`, `test_mse_0.5eq`, `test_mse_0.7eq`.
- `center_false_decisive_0.1eq`, `center_false_decisive_0.2eq`.
- `abs_cal_gap`/`test_max_midband_abs_cal_gap`.
- Scatter/hexbin không còn center spread rộng như B1/B2.

## Nếu C1 fail

Nếu C1 fail broad full-test, root cause còn lại mạnh nhất sẽ chuyển sang representation/encode aliasing hoặc label noise không giải bằng objective. Khi đó bước tiếp theo không phải B3/B4, mà là encode refresh tối thiểu:

- thêm halfmove/no-progress plane,
- thêm phase/material-phase feature nếu có pipeline ổn định,
- giữ broad validation gate của C1 để so sánh.

## File triển khai

- Helper objective/train: `train_v6_broad_objective/broad_objective_train_helpers.py`.
- Wrapper phase/env: `train_v6_broad_objective/phase_c1_train_helpers.py`.
- README và notebook Colab được đặt cùng folder `train_v6_broad_objective`.
