# Spec DGRN-X-v1 Policy Và Distillation

Ngày: `2026-05-05`  
Trạng thái: `design proposal`  
Phạm vi: mở policy sau khi `v0` pass; định nghĩa contract distillation và runtime-friendly student  
Phụ thuộc: [`dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md`](./dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md)

---

## 1) Kết luận điều hành

`v1` chỉ được mở **sau khi `v0` pass broad full-test**.

`v1` có hai mục tiêu:

1. thêm policy head theo cách không phá value backbone,
2. định nghĩa contract distillation đủ sạch để sau này nén teacher sang student.

`v1` không phải RL spec. RL được tách riêng.

Teacher topology ở `v1` được giả định là topology cố định từ [`dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md`](./dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md):

- input `58 x 8 x 8`
- residual CNN trunk chuẩn
- feature interface `F_trunk` và `g_pool`

`v1` không còn phải đoán teacher topology là gì; contract teacher đã được khóa ở `v0`.

---

## 2) Vì sao `v1` phải tách khỏi `v0`

Local evidence hiện tại chưa cho phép mở joint policy/value ngay:

- Phase B/C1 vẫn chưa có một value teacher vượt L4 trên broad core metrics.
- Khi value chưa ổn, thêm policy gradient chỉ làm khó chẩn đoán hơn.

Nguồn:

- [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)
- [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)

---

## 3) Policy head: quyết định thiết kế

### 3.1 Không dùng legal-move scorer làm main path của `v1`

Spec cũ đề xuất:

```text
score each legal move individually
```

Thiết kế đó bị loại khỏi `v1 core` vì:

- batch có số legal moves biến thiên,
- GPU throughput và kernel utilization khó tối ưu hơn,
- nested/jagged tensor support của PyTorch vẫn còn hạn chế hơn dense path.

Nguồn:

- [PyTorch nested tensors](https://docs.pytorch.org/docs/stable/nested)
- [AlphaZero preprint](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [Lc0 AlphaZero primer](https://lczero.org/dev/lc0/search/alphazero/)

### 3.2 Dùng fixed masked policy head

`v1` dùng fixed action head kiểu AlphaZero/Lc0:

- dense policy tensor trên action space cố định,
- mask các nước không hợp lệ ở loss và inference.

Action space mặc định của spec này là:

```text
8 x 8 x 73
```

Nếu implementation sau này chọn flatten tương đương như `4672` logits hoặc một mapping cố định khác, mapping đó phải được khóa cứng trong một spec/pipeline riêng trước khi train. `v1` không cho phép action space “sẽ quyết định sau”.

Policy head ở mức spec:

```text
F_trunk -> 1x1 / small conv projection -> policy logits over fixed move channels
```

Ưu điểm:

- dense GPU path,
- dễ batch lớn,
- thuận tiện cho self-play/search later,
- dễ distill hơn legal-move scorer ragged.

### 3.3 Label policy

Thứ tự ưu tiên cho policy target:

1. `search visit distribution` từ search teacher,
2. `soft engine move distribution` trên candidate set,
3. `one-hot played move` chỉ là fallback cuối.

Không dùng one-hot move played như target mặc định nếu có signal search mềm tốt hơn.

---

## 4) Training schedule cho `v1`

### 4.1 Điều kiện mở `v1`

Chỉ mở nếu `v0` đã:

- pass broad validation gate,
- pass offline full-test gate,
- không còn regression rõ ở `overall_mse`, `overall_pearson`, center safety.

### 4.2 Stage `v1-warmup`

Mục tiêu:

- học policy mà không phá value.

Khuyến nghị:

- freeze phần lớn torso,
- chỉ mở policy head và `2-4` residual blocks cuối,
- giữ value head active dưới dạng anchor.

Loss:

```text
L_v1_warmup =
  λ_pi * CE_or_KL(policy_logits, policy_target)
  + λ_anchor * SmoothL1(v_current, stopgrad(v_v0_teacher))
```

### 4.3 Stage `v1-joint`

Chỉ mở nếu warmup không làm xấu value metrics.

Loss:

```text
L_v1_joint =
  L_value_v0
  + λ_pi(t) * L_policy
  + λ_anchor * L_value_anchor
```

Trong đó `λ_pi(t)` ramp dần từ nhỏ lên, không bật full-strength từ đầu.

### 4.4 Gradient monitoring

Phải log ít nhất:

- cosine giữa `g_value` và `g_policy`,
- norm ratio,
- broad value metrics sau mỗi epoch.

Chỉ cân nhắc PCGrad/surgery nếu:

- đã đo conflict kéo dài,
- và value drift là do joint optimization thật.

---

## 5) Distillation contract

### 5.1 Chỉ distill từ teacher đã pass

Student chỉ được phép học từ teacher đã:

- pass broad validation,
- pass offline full-test,
- có calibration/post-hoc contract rõ ràng.

Không distill teacher còn fail broad gate.

### 5.2 Không giả định student copy được toàn bộ geometry của teacher

Phản biện cũ về việc "teacher hình học phức tạp distill sang student quá nghèo sẽ đổ vỡ" là quá mạnh nếu nói như một định lý tuyệt đối, nhưng cảnh báo đó đúng ở chỗ:

- capacity gap quá lớn sẽ làm student collapse,
- architectural mismatch quá cực đoan làm soft targets mất ý nghĩa.

Vì vậy `v1` yêu cầu student giữ các ràng buộc sau:

- cùng input schema encode,
- cùng fixed policy action mapping,
- cùng backbone family ở mức cao: `narrow residual CNN`,
- không dùng student thuần MLP trên flattened board làm mainline.

Student mặc định của `v1` là:

- residual CNN hẹp hơn teacher,
- ví dụ `128 channels x 8-10 residual blocks`,
- giữ shape feature map `8 x 8`,
- head value/policy cùng semantics với teacher.

### 5.3 Distillation loss

Phải có temperature:

```text
L_distill =
  α * SmoothL1(v_s, v_teacher)
  + η * ||Proj(F_s) - stopgrad(F_t)||_2^2
  + β * T^2 * KL(softmax(pi_t / T) || softmax(pi_s / T))
```

với `T > 1`.

Nếu chưa có policy tốt, chỉ distill scalar value trước.

Nguồn nền cho contract này:

- [Distilling the Knowledge in a Neural Network](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/)
- [AlphaZero preprint](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)

### 5.4 Student outputs

Student core output:

- scalar value bắt buộc,
- policy optional nếu runtime cần move ordering.

Không bắt buộc:

- WDL,
- uncertainty,
- draw head riêng.

---

## 6) Quantization-aware constraints cho student

Nếu student phục vụ runtime/search:

- ưu tiên `ReLU` hoặc activation fuse-friendly,
- tránh runtime BatchNorm khi export cuối,
- ưu tiên conv/linear pattern chuẩn,
- tránh attention lớn ở runtime path.

Nguồn:

- [PyTorch quantization docs](https://docs.pytorch.org/docs/stable/quantization.html)
- [Stockfish NNUE introduction](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)
- [Lc0 transformer progress](https://lczero.org/blog/2024/02/transformer-progress/)

---

## 7) Những gì `v1` cố ý chưa làm

`v1` không:

- mở self-play RL,
- thay broad validation bằng policy metrics,
- cho phép policy thắng value trong checkpoint selection,
- định nghĩa runtime deployment cuối cùng.

---

## 8) Điều kiện pass/fail của `v1`

Pass nếu:

- policy objective giảm ổn định,
- policy metrics tăng,
- value broad metrics không regress đáng kể,
- teacher vẫn pass full-test,
- student distill pilot không collapse nếu đã mở distillation.

Fail nếu:

- policy học nhưng value trượt broad gate,
- policy target quá noisy làm training không ổn định,
- student chỉ khớp train nhưng fail validation/search screening.

---

## 9) Quyết định cuối cùng

`v1` của DGRN-X là:

- fixed masked policy head, không legal-move scorer làm mainline,
- staged policy warmup rồi mới joint,
- distillation có temperature và capacity contract,
- student phải gần teacher về inductive bias cốt lõi, không được thuần flatten-MLP.

Đây là cách mở rộng hiện đại nhưng vẫn giữ được khả năng kiểm soát failure.
