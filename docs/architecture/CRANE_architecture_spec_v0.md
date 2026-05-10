# CRANE-v0 Architecture Specification

**Ngày:** 2026-05-10  
**Trạng thái:** `finalized`  
**Phạm vi:** Đặc tả kiến trúc mạng giá trị (Value Network) cho dòng CRANE  
**Tên mạng:** CRANE — **C**onv-**R**ay **A**ttention **N**etwork for **E**valuation  
**Tiền tố:** Tài liệu này thay thế toàn bộ `dgrn_x_v0.1_architecture_spec.md` và `dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md`  
**Phụ thuộc:** [`CRANE_encode_spec_v0.md`](./CRANE_encode_spec_v0.md)
**Encode schema:** `crane_v0_stm_spatial18_scalar5`

---

## 1. Kết luận điều hành

CRANE-v0 là một **value-only teacher network** được thiết kế theo 3 nguyên tắc:

1. **Serial Conv→Attention**: Tích chập trích xuất đặc trưng cục bộ trước, attention tổng hợp reasoning toàn cục sau — đúng theo CoAtNet (Dai et al., NeurIPS 2021).
2. **Chess-specific inductive bias**: Directed Ray Stream với GRU học được (learned directional gates) mã hóa trực tiếp line-of-sight của Xe/Tượng/Hậu.
3. **Giản lược để đọc failure**: Single scalar head, hybrid loss có curriculum, không multi-task.

Mục tiêu: kiểm chứng giả thuyết "representation + torso + encode" trước khi mở policy hay RL.

---

## 2. Triết lý thiết kế và cơ sở lý thuyết

### 2.1 Serial Conv→Attention

**Luồng:** Input → Conv trích xuất đặc trưng → Attention reasoning trên refined features

**Bằng chứng lý thuyết.** CoAtNet (Dai et al., NeurIPS 2021) chứng minh thực nghiệm trên ImageNet rằng:

> *"Relative attention after depthwise convolution yields the best accuracy and generalization."*

Tích chập có lợi thế: (a) gom nhóm mẫu cục bộ cực rẻ về FLOPs, (b) tạo translation equivariance. Attention có lợi thế: (a) tổng hợp thông tin toàn cục không bị giới hạn receptive field, (b) lý luận đa bước (multi-hop reasoning). Đặt attention quá sớm buộc nó phải học lại những đặc trưng cơ bản mà convolution đã làm tốt.

**Áp dụng cho cờ vua:** Trên bàn cờ 8×8, 12 ResBlocks với kernel 3×3 cho receptive field 25×25 — đã phủ toàn bộ bàn cờ nhiều lần. Do đó, 12 blocks cuối ResNet gần như không thêm thông tin cục bộ mới. Đẩy capacity từ conv sang attention ở giai đoạn sau là trao đổi (trade) hợp lý: giảm 4 blocks conv nhàm chán để lấy 5 blocks attention tinh vi.

### 2.2 Directed Ray Stream với GRU học được

**Vấn đề với cumsum thuần túy:** Cumsum dọc theo direction $d$ truyền thông tin tuyến tính từ ô này sang ô kia, nhưng **không phân biệt** quân cờ chắn đường quan trọng vs. ô trống. Ví dụ: nếu Hậu ở a1 và Xe ở a8, cumsum dọc cột a sẽ cộng thông tin Hậu vào Xe, nhưng cũng cộng mọi quân trung gian không quan trọng tại a2–a7.

**Giải pháp: Directional GRU (DGRU).** Mỗi direction sử dụng một GRU cell riêng. Reset gate của GRU đóng vai trò "bỏ qua thông tin không quan trọng trên đường ray", update gate đóng vai trò "giữ thông tin từ quân cờ quan trọng khi quét qua ô trống".

**Bằng chứng lý thuyết:**
- *Spatial RNN / Directed Acyclic Graph RNNs* đã được chứng minh hiệu quả trong phân tích ảnh y tế (8-connected grid graphs) và Go (KataGo sử dụng spatial feature propagation).
- GRU (Cho et al., 2014) được thiết kế chính xác cho sequential data ngắn cần learned gating. Với maximum 7 steps per ray trên bàn cờ 8×8, GRU là lựa chọn phù hợp hơn LSTM (ít params hơn ~40%) và phù hợp hơn cumsum thuần (có learned gates).

### 2.3 FiLM Conditioning cho biến vô hướng

Xem chi tiết chứng minh toán học trong Encode Spec, Mục 2.2–2.3. Tóm tắt: constant planes lãng phí 21.7% năng lực tích chập; FiLM modulate toàn bộ channel space bằng 2,304 params.

### 2.4 Không mở multi-task khi value chưa ổn

Nếu failure chính là value calibration / broad fit, thêm policy gradient tạo cạnh tranh gradient (gradient interference) làm chẩn đoán khó hơn.

**Tham chiếu:**
- PCGrad (Yu et al., NeurIPS 2020): gradient projection cho multi-task
- Multi-Task Learning as Multi-Objective Optimization (Sener & Koltun, 2018): nhiệm vụ cạnh tranh
- GradNorm (Chen et al., ICML 2018): điều tiết gradient norm

Kết luận: chưa thêm policy vào v0.

---

## 3. Tổng quan kiến trúc

### 3.1 Sơ đồ luồng dữ liệu

```
X_spatial ∈ R^{B×18×8×8}           s ∈ R^{B×5}
        │                              │
        ▼                              ▼
   ┌─────────┐                  ┌───────────┐
   │  Stem    │                  │ FiLM Gen  │
   │ Conv3×3  │                  │ Linear    │
   │ BN+SiLU  │                  │ 5 → 2C    │
   └────┬─────┘                  └─────┬─────┘
        │                              │ [γ, β]
        ▼                              ▼
   ┌──────────────────────────────────────┐
   │        FiLM Conditioning            │
   │   H₀ = γ ⊙ Stem(X) + β            │
   └──────────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
  ┌──────────────┐  ┌──────────────────┐
  │  GridStream  │  │ DirectedRayStream│
  │ 12 ResBlocks │  │  8× DGRU(24)    │
  │  (parallel)  │  │   (parallel)     │
  └──────┬───────┘  └────────┬─────────┘
         │                    │
         ▼                    ▼
  ┌──────────────────────────────────────┐
   │          RayFusion (Gated)          │
   │  H = H_grid + α·σ(g)⊙f([H_g;H_r]) │
   └──────────────┬───────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────────┐
   │     SerialAttentionStage            │
   │     5 Self-Attention Blocks         │
   │     Pre-LN, RelPosBias, 8 heads     │
   └──────────────┬───────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────────┐
   │       Attention Pooling             │
   │   h = Σᵢ softmax(w·Hᵢ) · Hᵢ       │
   └──────────────┬───────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────────┐
   │          Value Head                 │
   │  Linear(C→C/3) → SiLU → Linear→1   │
   │  → tanh                             │
   └──────────────────────────────────────┘
                  │
                  ▼
           p ∈ [-1, 1]
```

### 3.2 Ký hiệu thống nhất

| Ký hiệu | Ý nghĩa | Giá trị v0 |
|---------|---------|-----------|
| $B$ | Batch size | — |
| $C$ | Channel width cơ bản | 192 |
| $H, W$ | Chiều cao, rộng spatial | 8, 8 |
| $N$ | Số tokens = $H \times W$ | 64 |
| $D_{\text{ray}}$ | GRU hidden size per direction | 24 |
| $L_{\text{attn}}$ | Số attention blocks | 5 |
| $N_{\text{heads}}$ | Số attention heads | 8 |
| $D_{\text{head}}$ | Head dimension = $C / N_{\text{heads}}$ | 24 |
| $L_{\text{grid}}$ | Số ResBlocks | 12 |

---

## 4. Đặc tả từng thành phần

### 4.1 Stem

**Input:** $X_{\text{spatial}} \in \mathbb{R}^{B \times 18 \times 8 \times 8}$

**Công thức:**

$$h_{\text{stem}} = \text{SiLU}(\text{BN}(\text{Conv}_{3 \times 3}(X_{\text{spatial}})))$$

**Chi tiết:** $\text{Conv}_{3 \times 3}$: kernel $3 \times 3$, stride 1, padding 1, input channels = 18, output channels = $C$ = 192.

**Output:** $h_{\text{stem}} \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Tham số:** $18 \times 192 \times 9 + 192 + 192 \times 2 = 31{,}104 + 192 + 384 = 31{,}680$

(Lưu ý: Conv bias=False khi dùng BN là practice phổ biến; nếu bias=False thì params = $18 \times 192 \times 9 + 384 = 31{,}488$)

**Lý do chọn Conv 3×3 làm stem:** Kernel 3×3 là kích thước tối thiểu để capture mối quan hệ giữa ô trung tâm và 8 ô lân cận. Đây là nền tảng của mọi ResNet-style backbone. Kernel lớn hơn (5×5, 7×7) không mang lợi thế trên bàn cờ 8×8 vì receptive field nhanh chóng bao phủ toàn bộ board.

---

### 4.2 FiLM Conditioning

**Input:** $h_{\text{stem}} \in \mathbb{R}^{B \times C \times 8 \times 8}$, $s \in \mathbb{R}^{B \times 5}$

**Công thức:**

$$[\gamma, \beta] = \text{chunk}\!\left(W_{\text{film}} \cdot s + b_{\text{film}},\; 2\right)$$

$$h_0 = \gamma \odot h_{\text{stem}} + \beta$$

**Chi tiết:**
- $W_{\text{film}} \in \mathbb{R}^{2C \times 5}$, $b_{\text{film}} \in \mathbb{R}^{2C}$
- $\gamma, \beta \in \mathbb{R}^{B \times C}$, reshape thành $\mathbb{R}^{B \times C \times 1 \times 1}$ để broadcast
- $\odot$ là nhân theo từng element (Hadamard product)

**Khởi tạo identity:**

$$W_{\text{film}} \leftarrow \mathbf{0}, \quad b_{\text{film}}[0{:}C] \leftarrow \mathbf{1}, \quad b_{\text{film}}[C{:}2C] \leftarrow \mathbf{0}$$

Điều này đảm bảo lúc bắt đầu training: $\gamma = \mathbf{1}$, $\beta = \mathbf{0}$ → FiLM là identity mapping → không disrupt early training.

**Output:** $h_0 \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Tham số:** $2C \times 5 + 2C = 1{,}920 + 384 = 2{,}304$

**Monitoring trong training:** Log $\|\gamma\|_\infty$ và $\|\beta\|_\infty$ mỗi 1000 steps. Nếu $\|\gamma\|_\infty > 10$ hoặc $\|\beta\|_\infty > 10$, FiLM đang quá mạnh → cần điều chỉnh learning rate cho FiLM riêng.

---

### 4.3 GridStream (Residual Backbone)

**Input:** $h_0 \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Công thức:**

$$h_{\text{grid}}^{(i)} = h_0 + \sum_{i=1}^{L_{\text{grid}}} \text{ResBlock}(h^{(i-1)})$$

trong đó mỗi ResBlock:

$$\text{ResBlock}(x) = x + \text{Conv}_{3 \times 3}(\text{SiLU}(\text{BN}(\text{Conv}_{3 \times 3}(\text{SiLU}(\text{BN}(x))))))$$

**Chi tiết:**
- 2 lớp Conv 3×3 + BN + SiLU trong mỗi block (kiến trúc ResNet chuẩn)
- Mọi Conv: stride 1, padding 1, groups = 1
- Activation: SiLU ($\text{SiLU}(x) = x \cdot \sigma(x)$)
- BatchNorm: track_running_stats = True (chế độ train/eval chuẩn)

**Output:** $h_{\text{grid}} \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Tham số mỗi block (Conv bias=False + BN):** $2 \times (C \times C \times 9 + 2C) = 2 \times (331{,}776 + 384) = 2 \times 332{,}160 = 664{,}320$

**Tổng GridStream:** $12 \times 664{,}320 = 7{,}971{,}840$

**Lý do 12 blocks thay vì 16:** Receptive field của $k$ blocks với kernel 3×3 là $2k+1$. Với $k=12$: receptive field = 25×25, đã phủ bàn cờ 8×8 nhiều lần. 4 blocks cuối gần như chỉ thêm capacity mà không thêm thông tin cục bộ mới. Giảm từ 16 xuống 12 nhường capacity cho 5 attention blocks.

**Lý do chọn BatchNorm thay vì GroupNorm:** BatchNorm có tốc độ training nhanh hơn trên GPU nhờ vectorized implementation. Cho teacher network (v0), throughput training quan trọng hơn inference. Student network (v1) có thể dùng GroupNorm để dễ quantize.

**Lý do chọn SiLU thay vì ReLU/Mish:** SiLU ($x \cdot \sigma(x)$) là activation smooth, không có dead region như ReLU, và không có chi phí tính toán phức tạp như Mish ($x \cdot \tanh(\text{softplus}(x))$). SiLU đã được chứng minh thực nghiệm tốt hơn ReLU trong ResNet-family (ResNet-D, ConvNeXt).

---

### 4.4 DirectedRayStream (DGRU)

**Input:** $X_{\text{spatial}} \in \mathbb{R}^{B \times 18 \times 8 \times 8}$ (raw spatial planes, **không** qua FiLM)

**Bước 1 — Projection:**

$$x_{\text{ray}} = \text{Conv}_{1 \times 1}(X_{\text{spatial}}) \in \mathbb{R}^{B \times (8 \cdot D_{\text{ray}}) \times 8 \times 8}$$

Reshape thành 8 nhóm channels, mỗi nhóm $D_{\text{ray}}$ channels:

$$x_{\text{ray}}^{(d)}[\cdot, r, c] \in \mathbb{R}^{D_{\text{ray}}}, \quad d \in \{0, 1, \ldots, 7\}$$

**Bước 2 — Directional GRU scans:**

Cho mỗi direction $d$, chạy GRU cell dọc theo scan order của direction đó:

$$h_d[p] = \text{GRU}_d(x_d[p],\; h_d[p-1]), \quad h_d[0] = \mathbf{0}$$

trong đó GRU cell được định nghĩa:

$$r_p = \sigma(W_{ir}^d \cdot x_d[p] + b_{ir}^d + W_{hr}^d \cdot h_d[p-1] + b_{hr}^d)$$

$$z_p = \sigma(W_{iz}^d \cdot x_d[p] + b_{iz}^d + W_{hz}^d \cdot h_d[p-1] + b_{hz}^d)$$

$$n_p = \tanh(W_{in}^d \cdot x_d[p] + b_{in}^d + r_p \odot (W_{hn}^d \cdot h_d[p-1] + b_{hn}^d))$$

$$h_d[p] = (1 - z_p) \odot n_p + z_p \odot h_d[p-1]$$

**Chi tiết GRU:**
- Input size = $D_{\text{ray}} = 24$
- Hidden size = $D_{\text{ray}} = 24$
- Mỗi direction có bộ tham số GRU riêng (không share giữa các direction)
- Trong cùng một direction, tham số GRU được share across positions (weight sharing theo chiều dọc scan, tương tự RNN chuẩn)

**Bước 3 — Concatenate + Projection:**

$$h_{\text{ray}} = \text{Conv}_{1 \times 1}(\text{Concat}(h_0, h_1, \ldots, h_7)) \in \mathbb{R}^{B \times C \times 8 \times 8}$$

Concat 8 hidden states ($8 \times D_{\text{ray}} = 192$ channels) → Conv1×1(192→192) để project về channel dimension $C$.

**Scan orders cho 8 direction:**

Bàn cờ tọa độ $(r, c)$, $r \in \{0, \ldots, 7\}$, $c \in \{0, \ldots, 7\}$. $r=0$ là phía đối thủ, $r=7$ là phía STM.

| Direction | Nhóm scan line | Thứ tự trong line | Luồng thông tin |
|-----------|---------------|-------------------|-----------------|
| N | Mỗi cột $c$ | $r$: 7→0 | Nam → Bắc |
| S | Mỗi cột $c$ | $r$: 0→7 | Bắc → Nam |
| E | Mỗi hàng $r$ | $c$: 0→7 | Tây → Đông |
| W | Mỗi hàng $r$ | $c$: 7→0 | Đông → Tây |
| NE | $c - r = \text{const}$ | $(r_\text{max}, c_\text{min})$ → $(r_\text{min}, c_\text{max})$ | Tây Nam → Đông Bắc |
| SW | $c - r = \text{const}$ | $(r_\text{min}, c_\text{max})$ → $(r_\text{max}, c_\text{min})$ | Đông Bắc → Tây Nam |
| NW | $c + r = \text{const}$ | $(r_\text{max}, c_\text{max})$ → $(r_\text{min}, c_\text{min})$ | Đông Nam → Tây Bắc |
| SE | $c + r = \text{const}$ | $(r_\text{min}, c_\text{min})$ → $(r_\text{max}, c_\text{max})$ | Tây Bắc → Đông Nam |

Với diagonal scans, chỉ bao gồm các vị trí hợp lệ ($0 \leq r \leq 7$, $0 \leq c \leq 7$). $r_\text{min}$, $r_\text{max}$, $c_\text{min}$, $c_\text{max}$ là giá trị biên của các vị trí hợp lệ trên từng đường chéo.

**Output:** $h_{\text{ray}} \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Tham số:**
- Projection 1×1 Conv(18→192): $18 \times 192 + 192 = 3{,}648$
- 8 × GRU(24, 24): $8 \times 3 \times 2 \times (24 \times 24 + 24) = 8 \times 3{,}600 = 28{,}800$
- Output 1×1 Conv(192→192): $192 \times 192 + 192 = 37{,}056$
- **Tổng: 69{,}504**

**Tại sao GRU nhận raw spatial input thay vì FiLM-conditioned features:**
1. RayStream extract pattern không gian dọc theo đường thẳng — thông tin này là local, không phụ thuộc scalar context.
2. Scalar context (rule50, phase) ảnh hưởng *cách diễn giải* ray features, không ảnh hưởng *việc trích xuất* ray features.
3. Fusion layer (Mục 4.5) kết hợp GridStream (đã conditioned) với RayStream, nên scalar context sẽ ảnh hưởng ray features thông qua gate.

**Tại sao 8 direction riêng biệt thay vì 4 complement pairs:**
- Mỗi direction có ngữ nghĩa riêng: Bắc→Nam khác Nam→Bắc vì thông tin lan truyền khác chiều.
- 8 bộ tham số riêng cho phép mạng học asymmetry (ví dụ: tấn công Bắc→Nam khác phòng thủ Nam→Bắc).
- Chi phí tăng chỉ ~29K params — không đáng kể so với 9.7M tổng.

---

### 4.5 RayFusion (Gated Residual)

**Input:** $h_{\text{grid}}, h_{\text{ray}} \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Công thức:**

$$h_{\text{fused}} = h_{\text{grid}} + \alpha \cdot \sigma(g([h_{\text{grid}}; h_{\text{ray}}])) \odot f([h_{\text{grid}}; h_{\text{ray}}])$$

trong đó:
- $[h_{\text{grid}}; h_{\text{ray}}] \in \mathbb{R}^{B \times 2C \times 8 \times 8}$ là concatenation dọc channel dimension
- $g = \text{Conv}_{1 \times 1}(2C \to C)$: gate generator
- $f = \text{Conv}_{1 \times 1}(2C \to C)$: feature transform
- $\sigma$: hàm Sigmoid
- $\alpha$: residual scale, khởi tạo = 0.1 (xem bên dưới)
- $\odot$: nhân theo từng element

**Khởi tạo gate bias chống collapse:**

$$b_g \leftarrow +1.0$$

Sigmoid(1.0) ≈ 0.73 → gate mở ~73% lúc bắt đầu training, đảm bảo RayStream gradient không bị triệt tiêu sớm.

**Residual scale $\alpha = 0.1$:**

Mục đích: early training, contribution từ ray branch bị nhân với $\alpha = 0.1$, giảm rủi ro noise từ ray branch (chưa học tốt) disrupt grid stream (đã ổn định). Khi ray branch học được features tốt, gate sigmoid mở rộng, hiệu quả $\alpha \cdot \sigma(\cdot)$ có thể tăng lên.

$\alpha$ có thể là:
- (a) Hằng số cố định = 0.1 (khuyến nghị cho v0-sanity)
- (b) Learnable parameter khởi tạo = 0.1 (khuyến nghị cho v0-main)

**Output:** $h_{\text{fused}} \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Tham số:**
- $g$: $2C \times C + C = 384 \times 192 + 192 = 73{,}920$
- $f$: $2C \times C + C = 73{,}920$
- **Tổng: 147{,}840**

**Cơ sở lý thuyết:**
- SE-Net (Hu et al., CVPR 2018): Sigmoid gate cho channel-wise attention
- Highway Networks (Srivastava et al., 2015): Learned gating cho information flow control
- Phép cộng residual đảm bảo gradient đi mượt từ Attention → GridStream, không bị vanish qua gate

---

### 4.6 SerialAttentionStage

**Input:** $h_{\text{fused}} \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Bước 1 — Tokenize:**

$$H_0 = \text{Reshape}(h_{\text{fused}}) \in \mathbb{R}^{B \times N \times C}$$

trong đó $N = H \times W = 64$. Mỗi ô trên bàn cờ trở thành 1 token vector $C$-chiều.

**Bước 2 — $L_{\text{attn}}$ Self-Attention Blocks:**

Cho $i = 1, \ldots, L_{\text{attn}}$:

$$H'_i = H_{i-1} + \text{MHSA}(\text{PreLN}(H_{i-1}))$$

$$H_i = H'_i + \text{FFN}(\text{PreLN}(H'_i))$$

**Bước 3 — Reshape trở lại:**

$$h_{\text{attn}} = \text{Reshape}(H_{L_{\text{attn}}}) \in \mathbb{R}^{B \times C \times 8 \times 8}$$

#### 4.6.1 Multi-Head Self-Attention (MHSA)

**Công thức:**

$$\text{MHSA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{N_{\text{heads}}}) \cdot W_O$$

$$\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{D_{\text{head}}}} + B_{\text{rel}}\right) V_h$$

**Chi tiết:**
- $Q_h = X \cdot W_Q^{(h)} \in \mathbb{R}^{B \times N \times D_{\text{head}}}$
- $K_h = X \cdot W_K^{(h)} \in \mathbb{R}^{B \times N \times D_{\text{head}}}$
- $V_h = X \cdot W_V^{(h)} \in \mathbb{R}^{B \times N \times D_{\text{head}}}$
- $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{C \times D_{\text{head}}}$
- $W_O \in \mathbb{R}^{C \times C}$: output projection
- $D_{\text{head}} = C / N_{\text{heads}} = 192 / 8 = 24$

#### 4.6.2 Relative Position Bias

$$B_{\text{rel}}[i, j] = b[\Delta r(i,j),\; \Delta c(i,j)]$$

trong đó:
- $i, j$ là chỉ số token (0–63), ánh xạ về tọa độ bàn cờ $(r_i, c_i)$
- $\Delta r(i,j) = r_i - r_j \in \{-7, \ldots, +7\}$
- $\Delta c(i,j) = c_i - c_j \in \{-7, \ldots, +7\}$
- $b \in \mathbb{R}^{15 \times 15 \times N_{\text{heads}}}$: bảng bias học được, chia sẻ across tất cả attention blocks

**Tổng số tham số bias:** $15 \times 15 \times 8 = 1{,}800$

**Lý do chọn Relative Position Bias thay vì Absolute PE:** Trong cờ vua, "Mã cách Vua 2 ô dọc, 1 ô ngang" quan trọng hơn "Mã ở tọa độ A1". Relative bias mã hóa trực tiếp quan hệ tương đối, không phụ thuộc vị trí tuyệt đối → tốt cho generalization.

**Tham chiếu:** Swin Transformer (Liu et al., ICCV 2021) thiết kế relative position bias cho không gian 2D grid hiệu quả.

#### 4.6.3 Pre-LayerNorm

$$\text{PreLN}(X) = \text{LayerNorm}(X)$$

LayerNorm chuẩn hóa mỗi token vector độc lập:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

với $\mu, \sigma^2$ tính trên dimension $C$ của từng token, $\gamma, \beta \in \mathbb{R}^{C}$ learnable, $\epsilon = 10^{-5}$.

**Lý do Pre-LN thay vì Post-LN:** Xiong et al. (2020) chứng minh rằng Pre-LN tạo gradient path trực tiếp từ output layer đến mọi attention block qua residual connection, tránh vanishing gradient ở các block sâu. Post-LN buộc gradient đi qua LayerNorm, gây bất ổn định ở depth lớn.

#### 4.6.4 Feed-Forward Network (FFN)

$$\text{FFN}(x) = \text{Linear}_2(\text{SiLU}(\text{Linear}_1(x)))$$

- $\text{Linear}_1$: $C \to 2C$ (expansion factor = 2)
- $\text{Linear}_2$: $2C \to C$
- Activation: SiLU

**Lý do expansion factor = 2 thay vì 4:** Expansion factor 4 (Transformer chuẩn) trên $C = 192$ tạo FFN dimension = 768, làm tăng đáng kể params. Factor 2 là thỏa hiệp giữa capacity và ngân sách ~10M params. Trên bàn cờ 8×8 (64 tokens, D=192), signal complexity thấp hơn NLP (hàng nghìn tokens, D=768+), nên factor 2 là hợp lý.

#### 4.6.5 Tham số Attention Stage

Mỗi block:
- PreLN × 2: $2 \times 2C = 768$
- QKV projections: $3 \times (C \times D_{\text{head}} \times N_{\text{heads}} + D_{\text{head}} \times N_{\text{heads}}) = 3 \times (36{,}864 + 192) = 111{,}168$

  (Hoặc triển khai dưới dạng 1 Linear($C \to 3C$): $C \times 3C + 3C = 111{,}168$)

- Output projection: $C \times C + C = 37{,}056$
- Relative position bias: $1{,}800$
- FFN: $C \times 2C + 2C + 2C \times C + C = 73{,}728 + 384 + 73{,}728 + 192 = 148{,}032$
- **Mỗi block: ~299{,}240**

**5 blocks: ~1{,}496{,}200**

**Lý do 5 blocks thay vì 4:** Reasoning về cờ vua thường cần 4–5 bước logic: nhận diện mối đe dọa → xác định quân phòng thủ → đánh giá ô thoát → đánh giá phản công → kết luận. 5 hops cung cấp tối thiểu reasoning depth cho quy trình này. 4 blocks có thể cắt ngắn bước cuối cùng.

---

### 4.7 Attention Pooling

**Input:** $h_{\text{attn}} \in \mathbb{R}^{B \times C \times 8 \times 8}$

**Công thức:**

$$H_{\text{flat}} = \text{Reshape}(h_{\text{attn}}) \in \mathbb{R}^{B \times N \times C}$$

$$w = \text{softmax}(H_{\text{flat}} \cdot w_p) \in \mathbb{R}^{B \times N \times 1}$$

$$h_{\text{pool}} = \sum_{i=1}^{N} w_i \cdot H_{\text{flat}}[:, i, :] \in \mathbb{R}^{B \times C}$$

trong đó $w_p \in \mathbb{R}^{C \times 1}$ là vector trọng số học được.

**Lý do thay GAP thuần túy:**

Global Average Pooling gán trọng số bằng nhau cho mọi vị trí:

$$h_{\text{GAP}} = \frac{1}{N} \sum_{i=1}^{N} H_{\text{flat}}[:, i, :]$$

GAP mất hoàn toàn thông tin vị trí và ép mọi ô đóng góp ngang nhau. Trên thực tế, vị trí của Vua, quân đang bị tấn công, hay ô đích quan trọng cần được chú ý hơn ô trống ở góc.

Attention Pooling cho phép mạng **tự học** phân bổ trọng số vị trí phụ thuộc vào nội dung (content-dependent), với chi phí chỉ thêm 193 params. Trọng số $w_p$ được share across positions → không phá translation equivariance hoàn toàn.

**Tham số:** $C + 1 = 193$

**Tham chiếu:** Attention pooling được sử dụng trong SET Transformer (Lee et al., ICML 2019) và đã chứng minh vượt trội so với average pooling trong tasks cần selective aggregation.

---

### 4.8 Value Head

**Input:** $h_{\text{pool}} \in \mathbb{R}^{B \times C}$

**Công thức:**

$$z = W_2 \cdot \text{SiLU}(W_1 \cdot h_{\text{pool}} + b_1) + b_2$$

$$p = \tanh(z)$$

**Chi tiết:**
- $W_1 \in \mathbb{R}^{(C/3) \times C}$, $b_1 \in \mathbb{R}^{C/3}$
- $W_2 \in \mathbb{R}^{1 \times (C/3)}$, $b_2 \in \mathbb{R}^{1}$
- $C/3 = 64$ khi $C = 192$
- Output $p \in [-1, 1]$: value estimate từ góc nhìn STM

**Lý do 1 hidden layer thay vì Linear trực tiếp:**

Linear($C \to 1$) trực tiếp sau pooling ép mọi nonlinear combination xảy ra trong backbone. Tuy nhiên, việc quyết định value cuối cùng cần kết hợp nonlinear các channel features (ví dụ: "channel 47 cao VÀ channel 112 thấp → value mạnh"). 1 hidden layer cung cấp khả năng này với chi phí negligible.

So sánh:
- Direct: Linear(192→1) = 193 params, 0 nonlinear
- 1-hidden: Linear(192→64) + Linear(64→1) = 12,417 params, 1 nonlinear (SiLU)

**Tham chiếu:** Lc0 WDL head sử dụng hidden layer (256→1) sau GAP. AlphaZero paper sử dụng 2 hidden layers (256→1) cho value head.

**Tham số:** $192 \times 64 + 64 + 64 \times 1 + 1 = 12{,}352 + 65 = 12{,}417$

---

## 5. Phân tích ngân sách tham số

| Thành phần | Tham số | % Tổng |
|---|---|---|
| Stem | 31,680 | 0.33% |
| FiLM Conditioning | 2,304 | 0.02% |
| GridStream (12 ResBlocks) | 7,971,840 | 81.6% |
| DirectedRayStream (8×DGRU) | 69,504 | 0.71% |
| RayFusion (Gated Residual) | 147,840 | 1.52% |
| SerialAttentionStage (5 blocks) | 1,496,200 | 15.3% |
| Attention Pooling | 193 | ~0% |
| Value Head | 12,417 | 0.13% |
| **Tổng** | **9,729,974** | **~9.73M** |

**Tỷ lệ Grid : Ray : Attention:** 81.6% : 0.71% : 15.3%

**Nhận xét:** GridStream chiếm phần lớn params (đúng vai trò backbone), RayStream cực nhẹ nhưng mang inductive bias mạnh, Attention chiếm ~15% — đủ cho global reasoning mà không quá nặng.

---

## 6. Ước tính FLOPs

Tính cho 1 forward pass, input $18 \times 8 \times 8$, convention 1 MAC = 2 FLOPs:

| Thành phần | Ước tính FLOPs |
|---|---|
| Stem | ~40M |
| FiLM | ~0.004M (negligible) |
| GridStream (12 blocks) | ~1,020M |
| DirectedRayStream | ~5M |
| RayFusion | ~19M |
| SerialAttentionStage (5 blocks) | ~210M |
| Attention Pooling + Value Head | ~0.02M |
| **Tổng** | **~1.29 GFLOPs** |

CRANE-v0 nằm trong cùng hạng cân với Lc0 T1 ResNet (~1–1.5 GFLOPs), nhưng phân bổ capacity khác: ít conv hơn, có attention và ray stream.

---

## 7. Phân tích luồng thông tin

### 7.1 Information Flow qua các stage

```
Raw Board State
       │
       ├─ Spatial (18×8×8) ──────► Stem (local features)
       │                              │
       │                         FiLM (scalar conditioning)
       │                              │
       │                    ┌─── GridStream ───┐
       │                    │  (12 ResBlocks)   │
       │                    │  local patterns   │
       │                    │  translation eq.  │
       │                    └────────┬──────────┘
       │                             │
       ├─ Spatial (18×8×8) ─► DirectedRayStream ──┐
       │                    (8×DGRU)               │
       │                    line-of-sight          │
       │                    directional bias       │
       │                                           │
       │                    ◄── RayFusion (gated) ─►
       │                             │
       │                    SerialAttentionStage
       │                    (5 blocks)
       │                    global reasoning
       │                    multi-hop interaction
       │                             │
       │                    Attention Pooling
       │                    (content-weighted)
       │                             │
       │                    Value Head
       │                    (1 hidden + tanh)
       │                             │
       ▼                             ▼
   p ∈ [-1, 1]  ◄───────────────────┘
```

### 7.2 Gradient Flow

- **Residual connections** trong GridStream: gradient trực tiếp từ block $i$ đến block $i-1$ qua skip connection
- **FiLM**: gradient đến scalar encoder $W_{\text{film}}$ đi qua $\gamma, \beta$ — nếu FiLM identity-initialized, early gradient nhỏ → bảo vệ scalar path từ early disruption
- **RayFusion gate**: gradient đến RayStream đi qua $\alpha \cdot \sigma(g) \cdot f'$ — gate bias=+1.0 đảm bảo gradient không bị triệt tiêu sớm
- **Attention residual**: Pre-LN đảm bảo gradient đi trực tiếp qua residual path, không qua LayerNorm

### 7.3 Diagnosing Failure Modes

Nếu training fail, vị trí probable failure và cách chẩn đoán:

| Triệu chứng | Probable failure | Chẩn đoán |
|---|---|---|
| Validation loss không giảm | Stem/FiLM hoặc data issue | Log $\|\gamma\|$, kiểm tra input stats |
| Center tốt nhưng broad fail | Attention không học global | Visualize attention maps |
| RayStream gate → 0 | Gate collapse | Log mean gate value |
| Tails under-confident | Loss curriculum quá nhanh | Log bucketed MSE theo band |
| Calibration drift cuối run | λ_t tăng quá chậm | Log y-mse vs z-mse |

---

## 8. Loss Function

### 8.1 Loss tổng quát

$$L_t = \lambda_t \cdot (\tanh(z) - y)^2 + (1 - \lambda_t) \cdot (1 - y^2)^\beta \cdot \rho_\delta(z - \text{atanh}(y_{\text{clamp}}))$$

trong đó:
- $z$: logit dự đoán
- $y \in (-1, 1)$: target value
- $y_{\text{clamp}} = \text{clamp}(y, -0.99, 0.99)$: tránh atanh vô hạn
- $\rho_\delta$: Huber loss
- $\beta \approx 1$: trọng số nội suy giữa y-space ($\beta=2$) và z-space ($\beta=0$)
- $\lambda_t$: hệ số curriculum tăng dần

### 8.2 Huber Loss

$$\rho_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{nếu } |r| \leq \delta \\ \delta(|r| - \frac{1}{2}\delta) & \text{nếu } |r| > \delta \end{cases}$$

**Khuyến nghị:** $\delta = 1.0$

**Lý do:** $\text{atanh}(y)$ có range xấp xỉ $[-2.95, 2.95]$ cho $y \in [-0.99, 0.99]$. Residual $r = z - \text{atanh}(y)$ thường trong $[-3, 3]$. $\delta = 1.0$ cho phép MSE behavior trong $\pm 1$, Huber behavior bên ngoài → robust với outlier tail samples.

### 8.3 Curriculum Schedule cho $\lambda_t$

$$\lambda_t = \lambda_{\text{start}} + (\lambda_{\text{end}} - \lambda_{\text{start}}) \cdot \min\!\left(1,\; \frac{t}{T_{\text{total}}}\right)$$

**Khuyến nghị:**
- $\lambda_{\text{start}} = 0.2$: đầu training, 80% weight cho z-space → học dynamic range mạnh
- $\lambda_{\text{end}} = 0.95$: cuối training, 95% weight cho y-space → calibration final
- Không đặt $\lambda_{\text{end}} = 1.0$ vì vẫn cần chút z-space signal để tails không bị quên

**Cơ sở lý thuyết:**
- Đầu run cần học dynamic range, tránh gradient quá yếu ở tails → nhánh z-space phải còn đủ lực
- Cuối run mục tiêu là calibration tốt trong y-space → nhánh y-space phải chiếm ưu thế
- Curriculum biến đổi linear là khởi đầu hợp lý; có thể thử cosine schedule nếu linear cho thấy transition quá đột ngột

### 8.4 Gradient gần nghiệm

Gần nghiệm tối ưu $z \approx \text{atanh}(y)$, với $\beta = 1$:

$$\frac{\partial L_t}{\partial z} \approx 2 \left[ \lambda_t (1 - y^2)^2 + (1 - \lambda_t)(1 - y^2) \right] r$$

Hệ số hiệu dụng nằm giữa hai cực đoan:
- Pure y-space: $(1 - y^2)^2$ (gradient yếu ở tails)
- Pure z-space: $1$ (gradient mạnh nhưng gây drift)

Với $\lambda_t = 0.5$ (giữa run) và $y = 0.96$:
- Pure y-space weight: $0.0061$
- CRANE hybrid weight: $0.5 \times 0.0061 + 0.5 \times 0.0784 = 0.0423$
- Pure z-space weight: $1.0$

CRANE hybrid mạnh hơn pure y-space ~7× nhưng yếu hơn pure z-space ~24× tại tails → cân bằng.

### 8.5 Những gì bị loại khỏi v0

Không dùng làm loss chính:
- Batch hard-bucket calibration loss
- Hard center hinge $\text{ReLU}(|p| - m)^2$
- Oracle auxiliary gradient
- Policy loss
- Uncertainty NLL
- Pseudo-WDL cross-entropy

**Lý do:** v0 cần tối đa hóa khả năng đọc failure. Nếu broad fit vẫn fail, vấn đề còn ở representation/torso/target noise, không bị nhiễu bởi nhiều regularizer cạnh tranh.

---

## 9. Chiến lược Sampling và Checkpoint Selection

### 9.1 Sampling

**Mặc định:** Random theo phân phối train tự nhiên.

**Không dùng mặc định cho v0:**
- Band-balanced
- Sign-stratified

**Lý do:** Phase B đã cho evidence local rằng oversampling proxy có thể làm validation đẹp nhưng broad full-test xấu.

### 9.2 Checkpoint Selection

Checkpoint được chọn theo **broad validation score** trên split validation lớn, không dùng test.

Oracle subset chỉ để continuity/diagnostic, không phải promotion gate chính.

### 9.3 Scale khởi đầu

Hai nấc:

1. **v0-sanity**: 8–10 ResBlocks, $C = 160\text{–}192$, 3 attention blocks
2. **v0-main**: 12 ResBlocks, $C = 192$, 5 attention blocks

Không nhảy thẳng lên v0-main trước khi v0-sanity pass:
- Encode mới qua sanity
- Broad validation không nổ
- Throughput chấp nhận được

---

## 10. Ablation Plan

Chỉ 5 ablation có giá trị thông tin cao, chạy trên v0-sanity scale:

| Run | Config | Giả thuyết kiểm chứng |
|---|---|---|
| A0 | Grid-only + GAP + Linear(192→1), no FiLM, no Ray | Baseline thuần |
| A1 | Grid + FiLM (no Ray) | FiLM có cải thiện so với A0? |
| A2 | Grid + FiLM + Ray(cumsum) + Gate | Cumsum version — có đủ không? |
| A3 | Grid + FiLM + Ray(DGRU) + Gate | DGRU có tốt hơn cumsum? |
| A4 | Full CRANE-v0 (Grid+FiLM+DGRU+Gate+AttnPool+ValueHead) | Full proposal |
| A5 | A4 + 5 attention blocks (thay 3) | Attention depth có quan trọng? |

**Nguyên tắc chống confound:** Không thay đổi nhiều biến cùng lúc giữa các run. Mỗi run chỉ thêm 1 thay đổi so với run trước.

**Metrics chính cho ablation:**
- `overall_mse` (y-space)
- `overall_pearson`
- `test_mse_0.1eq` (decisive band)
- `center_false_decisive_0.1eq`
- `abs_cal_gap` trên các band

---

## 11. Quantization Compatibility (Cho Student v1)

CRANE-v0 là **teacher** — không cần quantize. Tuy nhiên, student (v1 distillation target) phải được thiết kế sẵn cho INT8/INT4. Quy định constraint cho student spec:

| Component | Teacher (v0) | Student (v1) | Lý do |
|---|---|---|---|
| Activation | SiLU | ReLU | ReLU fuse được với Conv trong INT8 |
| Normalization | BatchNorm | BatchNorm (fuse vào Conv trước export) | BN fuse chuẩn |
| Attention | 5 blocks Self-Attention | Không dùng | Softmax+matmul quantize khó, student inference cần nhanh |
| Ray Stream | DGRU (8×GRU) | Cumsum hoặc bỏ | GRU recurrent khó quantize |
| RayFusion Gate | Sigmoid | Hard Sigmoid ($\text{ReLU6}(x+3)/6$) | Hard sigmoid quantize thân thiện hơn |
| Value Head | SiLU + 1 hidden | ReLU + 1 hidden | Fuse-friendly |

**Student mặc định:** ResNet thuần, $128$ channels × 8–10 blocks, giữ input schema và value semantics giống teacher.

**Tham chiếu:**
- PyTorch Quantization Docs
- Stockfish NNUE (int8 inference)
- Lc0 Transformer Progress (quantization experiments)

---

## 12. Điều kiện Pass/Fail của v0

### Pass (mở v1)

v0 chỉ được coi là pass nếu **đồng thời**:
1. Broad validation pass (overall_mse, overall_pearson không regress so với L4 baseline)
2. Offline full-test không regress như B1/B2/C1 ở `overall_mse` và `overall_pearson`
3. `center_false_decisive_0.1eq` không xấu đi rõ
4. Decisive-band không bị under-predict nghiêm trọng

### Fail (không mở v1)

Nếu v0 fail:
- Không mở policy
- Không mở RL
- Không mở distillation
- Phân tích root cause trước khi iterate

---

## 13. Những gì spec này cố ý không làm

Spec này **không**:
1. Định nghĩa policy head → thuộc v1 spec
2. Định nghĩa RL loop → thuộc RL spec
3. Định nghĩa distillation procedure → thuộc v1 spec
4. Định nghĩa search integration → thuộc engine integration spec
5. Mở WDL, uncertainty, hay draw head phụ → giữ v0 giản lược

---

## 14. Quyết định cuối cùng

CRANE-v0 là:

- **Teacher value-only**, tên mạng CRANE = Conv-Ray Attention Network for Evaluation
- **Input**: 18 spatial planes + 5-dim scalar vector (inject qua FiLM)
- **Torso**: Grid (12 ResBlocks) ∥ DirectedRay (8×DGRU) → Gated Fusion → 5 Self-Attention Blocks
- **Head**: Attention Pool → Linear(192→64)→SiLU→Linear(64→1)→tanh
- **Loss**: Hybrid y-space + weighted z-space Huber với curriculum $\lambda_t$
- **Selection**: Broad validation
- **~9.73M params, ~1.29 GFLOPs**

Đây là phiên bản nhỏ nhất nhưng đủ hiện đại để test giả thuyết "representation + torso + encode" một cách sạch, với mọi quyết định thiết kế có nền tảng lý thuyết được trích dẫn cụ thể, không dựa trên trực giác.
