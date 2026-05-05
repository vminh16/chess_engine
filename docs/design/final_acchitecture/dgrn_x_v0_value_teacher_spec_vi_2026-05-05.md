# Spec DGRN-X-v0 Value Teacher

Ngày: `2026-05-05`  
Trạng thái: `design proposal`  
Phạm vi: teacher value-only, không policy gradient, không RL, không distillation loss chính  
Phụ thuộc: [`dgrn_x_encode_refresh_spec_vi_2026-05-05.md`](./dgrn_x_encode_refresh_spec_vi_2026-05-05.md)

---

## 1) Kết luận điều hành

`DGRN-X-v0` là một teacher value network mới, được thiết kế để trả lời trực tiếp những gì Phase B và C1 đã chứng minh:

- `head-only` không đủ,
- objective có thể sửa center nhưng vẫn fail broad fit,
- encode hiện tại thiếu state quan trọng.

Do đó `v0` phải:

1. đổi encode,
2. đổi torso theo inductive bias chess-specific,
3. giữ training objective đủ đơn giản để failure còn đọc được,
4. chưa mở policy, chưa mở RL, chưa mở uncertainty/WDL chính.

Đây là một **value-only teacher pilot có chủ đích**, không phải full multi-head system ngay từ đầu.

---

## 2) Bằng chứng local cần `v0`

### 2.1 Phase B chứng minh proxy hẹp không đại diện broad risk

Nguồn local:

- [`evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b1_sglobal_16b_256d_band_run1_best_gate/reports/core_metrics_table.csv`](../../evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b1_sglobal_16b_256d_band_run1_best_gate/reports/core_metrics_table.csv)
- [`evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1_best_gate/reports/core_metrics_table.csv`](../../evaluation/phase_b_offline_benchmark/outputs/offline_dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1_best_gate/reports/core_metrics_table.csv)

B1/B2 đều:

- cải thiện một phần oracle subset,
- nhưng thua L4 ở `overall_mse`, `overall_pearson`, `center_false_decisive`.

### 2.2 C1 chứng minh objective-only không đủ

Nguồn local:

- [`runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/selected_checkpoint_eval.json`](../../runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/selected_checkpoint_eval.json)
- [`runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/history.csv`](../../runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/history.csv)

C1:

- giảm mạnh `test_mse_0.1eq`,
- giảm mạnh `test_center_false_0.1eq`,
- nhưng broad gate vẫn fail,
- train objective còn giảm trong khi broad metrics không vượt được L4.

Kết luận:

- nếu tiếp tục chỉ tuning loss/head, expected information gain thấp hơn đổi representation + torso.

---

## 3) Nguyên lý thiết kế

`v0` tuân theo 5 nguyên lý:

1. **Teacher-first**  
   Teacher được phép chậm hơn runtime model; mục tiêu trước là học đúng value.

2. **Inductive bias chess-specific**  
   Backbone phải encode được:
   - locality trên `8x8`,
   - line-of-sight,
   - quan hệ quân-cờ toàn cục.

3. **Objective giản lược**  
   Không nhồi thêm nhiều head và nhiều loss mới cùng lúc.

4. **Selection theo broad risk**  
   Không quay lại oracle subset làm gate chính.

5. **Triển khai theo contract**  
   Encode, target semantics, head semantics, checkpoint policy phải được định nghĩa rõ, không ngầm suy luận.

---

## 4) Cơ sở lý thuyết

### 4.1 Representation matters hơn thay head đơn lẻ

Nguồn:

- [Representation Matters for Mastering Chess](https://arxiv.org/abs/2304.14918)
- [AlphaZero preprint](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [Lc0 transformer progress](https://lczero.org/blog/2024/02/transformer-progress/)

### 4.2 Không mở multi-task khi value chưa ổn

Nguồn:

- [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)
- [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)

Những nguồn này không nói “không bao giờ multi-task”, nhưng đủ để kết luận:

- nếu failure chính hiện tại là value calibration / broad fit,
- thì chưa nên thêm policy gradient vào cùng backbone ở `v0`.

### 4.3 Calibration nên là selection/gate trước khi là train-time regularizer cứng

Nguồn:

- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [MMCE: Differentiable Calibration](https://proceedings.mlr.press/v80/kumar18a.html)
- [A Large-Scale Study of Probabilistic Calibration in Neural Network Regression](https://proceedings.mlr.press/v202/dheur23a.html)

Kết luận cho repo này:

- calibration rất quan trọng,
- nhưng batch hard-bucket calibration loss không phải lựa chọn v0 sạch nhất.

---

## 5) Kiến trúc `v0`

### 5.1 Input

Input dùng schema `23 x 8 x 8` từ spec encode:

- `18` plane gốc,
- `+ rule50`,
- `+ phase`,
- `+ material_self`,
- `+ material_opp`,
- `+ material_delta`.

### 5.2 Torso tổng thể

`v0` dùng torso ba luồng:

1. `Grid stream`
2. `Directed ray stream`
3. `Piece relation stream`

và fusion theo residual gates.

### 5.3 Grid stream

Mục tiêu:

- giữ và biến đổi spatial tensor theo pattern local,
- làm xương sống chính của model.

Đề xuất:

- stem `3x3 conv`,
- `12-16` residual blocks,
- width `192-256`,
- activation `SiLU`,
- norm `BatchNorm2d` hoặc `GroupNorm`; v0 mặc định dùng `BatchNorm2d` để bám pattern repo hiện tại và tối ưu GPU dễ hơn.

Không dùng `Mish` cho dòng mới nếu không có lý do đặc biệt.

### 5.4 Directed ray stream

Mục tiêu:

- encode line-of-sight cho rook/bishop/queen,
- không buộc conv thường phải học lại mọi tương tác tia từ đầu.

`Ray` trong spec này phải là `8 directed scans`:

- north / south,
- east / west,
- northeast / southwest,
- northwest / southeast.

Không dùng mô tả mơ hồ kiểu “4 hướng” vì sẽ che khuất vấn đề chiều.

Implementation class ở mức spec:

- depthwise directional mixing,
- hoặc directional recurrent/scan block nhẹ,
- output cùng resolution `8x8`.

### 5.5 Piece relation stream

Mục tiêu:

- mô hình hóa tương tác quân-cờ ở mức token,
- bổ sung thông tin toàn cục mà grid conv học chậm.

Tokenization:

- mỗi quân hiện diện trên bàn tạo ra một token,
- tối đa `32` token,
- token gồm:
  - feature tại ô,
  - embedding loại quân,
  - embedding màu / perspective role.

Relational block:

- `2-4` self-attention blocks nhẹ,
- complete graph attention giữa quân,
- có relation bias theo:
  - cùng hàng/cột/chéo,
  - khoảng cách Manhattan/Chebyshev,
  - cùng phe / khác phe.

Không giữ `1` block duy nhất như spec cũ.

### 5.6 Fusion

Spec cũ dùng:

```text
PieceSummary(T) -> broadcast -> grid
```

Thiết kế đó bị loại bỏ khỏi `v0`.

Fusion đúng của `v0` là:

1. project piece tokens về embedding tại square gốc,
2. scatter trở lại `8x8`,
3. fuse với `grid` và `ray` bằng gated residual add.

Ví dụ ở mức hình thức:

```text
H_fused = H_grid + G([H_grid, H_ray, H_piece_scatter]) * F([H_grid, H_ray, H_piece_scatter])
```

Điều này giữ được locality của token stream thay vì broadcast một summary toàn cục.

---

## 6) Value head

`v0` chỉ có **một scalar head chính**.

Output:

```text
z = head(H_fused)
p = tanh(z)
```

Không bật trong `v0 core`:

- `WDL` main head,
- `draw_logit`,
- `sigma` uncertainty head,
- trực tiếp trộn `v_scalar` với `v_wdl`.

Lý do:

- repo hiện tại mới chỉ xác minh target scalar là source-of-truth chính,
- pseudo-WDL từ scalar không đủ sạch để làm primary target,
- thêm nhiều head sẽ làm việc đọc failure khó hơn.

---

## 7) Objective cho `v0`

### 7.1 Loss chính

`v0` dùng loss hybrid có curriculum, bám theo phân tích trong [`loss_function_design.md`](./loss_function_design.md):

```text
L_t =
  λ_t * (tanh(z) - y)^2
  + (1 - λ_t) * (1 - y^2)^β * Huber(z - atanh(y_clamped); δ)
```

Trong đó:

- `β ≈ 1` là điểm khởi đầu hợp lý,
- `λ_t` tăng dần theo epoch / global step,
- `δ` là Huber delta.

### 7.2 Những gì bị loại khỏi `v0 core`

Không dùng làm main loss ở `v0`:

- batch hard-bucket calibration loss,
- hard center hinge `relu(|p|-m)^2`,
- oracle auxiliary gradient,
- policy loss,
- uncertainty NLL,
- pseudo-WDL CE.

### 7.3 Lý do loại bỏ

`v0` cần tối đa hóa khả năng đọc failure:

- nếu broad fit vẫn fail, ta biết vấn đề còn ở representation / torso / target noise,
- không bị nhiễu bởi nhiều regularizer proxy cạnh tranh nhau.

---

## 8) Sampling, selection, checkpoint policy

### 8.1 Sampling

`random` theo phân phối train tự nhiên là default.

Không dùng mặc định:

- `band_balanced`,
- `sign_stratified`.

Vì Phase B đã cho evidence local rằng oversampling proxy có thể làm validation đẹp nhưng broad full-test xấu.

### 8.2 Selection

Checkpoint chính được chọn theo `broad validation score` trên split validation lớn, không dùng test.

Oracle subset:

- chỉ để continuity/diagnostic,
- không được là promotion gate chính.

Chi tiết gate thuộc eval spec.

---

## 9) Scale khởi đầu

`v0` không mở full-size ngay.

Khuyến nghị hai nấc:

1. `v0-sanity`
   - `8-10 blocks`
   - `160-192 channels`
2. `v0-main`
   - `12-16 blocks`
   - `192-256 channels`

Không nhảy thẳng lên scale lớn hơn trước khi:

- encode mới qua sanity,
- broad validation không nổ,
- throughput Colab còn chấp nhận được.

---

## 10) Không gian ablation cho `v0`

Chỉ 3 ablation có giá trị thông tin cao được phép trong vòng đầu:

1. `grid-only`
2. `grid + ray`
3. `grid + ray + piece-scatter`

Không thêm nhiều biến khác đồng thời. Đây là nguyên tắc chống confound.

---

## 11) Điều kiện pass/fail của `v0`

`v0` chỉ được coi là pass để mở `v1` nếu:

- broad validation pass,
- offline full-test không regress như B1/B2/C1 ở `overall_mse` và `overall_pearson`,
- center false decisive không xấu đi rõ,
- decisive-band không bị under-predict nghiêm trọng.

Nếu `v0` fail:

- không mở policy,
- không mở RL,
- không mở distillation.

---

## 12) Quyết định cuối cùng

`DGRN-X-v0` là:

- teacher value-only,
- encode `23 planes`,
- torso `grid + directed-ray + piece-scatter fusion`,
- head scalar duy nhất,
- objective hybrid `y-space + weighted z-space Huber curriculum`,
- selection bằng broad validation.

Đây là phiên bản nhỏ nhất nhưng đủ hiện đại để test giả thuyết “representation + torso + encode” một cách sạch.

