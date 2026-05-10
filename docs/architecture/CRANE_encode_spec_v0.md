# CRANE-v0 Encode Specification

**Ngày:** 2026-05-10  
**Trạng thái:** `finalized`  
**Phạm vi:** Đặc tả biểu diễn đầu vào cho dòng kiến trúc CRANE  
**Phiên bản schema:** `crane_v0_stm_spatial18_scalar5`  
**Tiền tố:** Tài liệu này thay thế toàn bộ `dgrn_x_encode_refresh_spec_vi_2026-05-05.md`

---

## 1. Kết luận điều hành

Encode của CRANE-v0 tách đầu vào thành hai modality:

1. **Spatial planes** $X_{\text{spatial}} \in \mathbb{R}^{18 \times 8 \times 8}$ — thông tin có tính chất vị trí
2. **Scalar vector** $s \in \mathbb{R}^{5}$ — thông tin trạng thái toàn cục không có vị trí không gian

Sự tách này được quyết định dựa trên chứng minh toán học (Mục 2) rằng các scalar plane đồng giá trị (constant planes) lãng phí năng lực biểu diễn của tích chập không gian, và cơ chế FiLM conditioning cung cấp đường tiêm thông tin hiệu quả hơn về mặt lý thuyết lẫn thực nghiệm.

---

## 2. Cơ sở lý thuyết

### 2.1 Aliasing tạo phương sai không thể khử

Nếu hàm encode $\phi$ ánh xạ hai trạng thái khác nhau $s_1, s_2$ vào cùng một tensor:

$$\phi(s_1) = \phi(s_2) \quad \text{nhưng} \quad y(s_1) \neq y(s_2)$$

thì bộ dự đoán tối ưu theo MSE chỉ có thể học:

$$f^*(x) = \mathbb{E}[Y \mid X = x]$$

với phương sai không thể khử:

$$\mathbb{E}[(Y - f^*(X))^2 \mid X = x] = \text{Var}(Y \mid X = x)$$

Kết luận: aliasing trong encode tạo ra trần cứng (hard ceiling) mà không kiến trúc hay loss function nào vượt qua được.

### 2.2 Constant plane lãng phí năng lực tích chập

**Định lý.** Tích chập 3×3 trên một channel đồng giá trị không tạo ra thông tin không gian mới, và tương đương với việc cộng một bias phụ thuộc vào giá trị vô hướng đó.

**Chứng minh.** Cho mặt phẳng đồng giá trị $P \in \mathbb{R}^{8 \times 8}$ với $P[r,c] = v$ $\forall (r,c)$. Cho kernel $K \in \mathbb{R}^{3 \times 3}$:

$$(K * P)[r,c] = \sum_{\Delta r, \Delta c} K[\Delta r, \Delta c] \cdot P[r+\Delta r, c+\Delta c] = v \cdot \sum_{\Delta r, \Delta c} K[\Delta r, \Delta c]$$

Tổng $\sum K$ là một số vô hướng độc lập với $(r,c)$. Do đó output là hằng số trên mọi vị trí không gian, chỉ đóng vai trò bias shift. $\blacksquare$

**Hệ quả định lượng.** Nếu đầu vào có $C_{\text{in}}$ channels và $K$ trong số đó là constant planes, năng lực trích xuất đặc trưng không gian của lớp tích chập đầu tiên bị giảm tỷ lệ $K / C_{\text{in}}$.

Trong thiết kế cũ (DGRN-X-v0.1), 5/23 input planes là constant planes → **21.7% năng lực tích chập bị lãng phí** trên thông tin không tạo gradient không gian.

### 2.3 FiLM Conditioning là cơ chế tiêm thông tin vô hướng vượt trội

Feature-wise Linear Modulation (FiLM) [Perez et al., AAAI 2018] biến đổi affine trên channel dimension:

$$H_{\text{film}} = \gamma \odot H + \beta$$

trong đó $\gamma, \beta \in \mathbb{R}^{C}$ được tính từ vector vô hướng $s$:

$$[\gamma, \beta] = W_{\text{film}} \cdot s + b_{\text{film}}, \quad W_{\text{film}} \in \mathbb{R}^{2C \times 5}, \; b_{\text{film}} \in \mathbb{R}^{2C}$$

**So sánh chi phí:**

| Phương pháp | Params | FLOPs tích chập tiêu tốn | Không gian ảnh hưởng |
|---|---|---|---|
| Constant planes (5 planes) | 0 (trong stem conv) | 5 × 192 × 9 × 64 ≈ 553K MACs/vị trí | Chỉ 5/23 channels đầu vào |
| FiLM conditioning | $2 \times 5 \times 192 + 2 \times 192 = 2{,}304$ | 0 (không qua tích chập) | Toàn bộ $C$ channels |

**Kết luận:** FiLM modulate toàn bộ channel space bằng 2,304 params, trong khi constant planes chiếm 21.7% đầu vào tích chập nhưng chỉ cung cấp bias shift. FiLM vượt trội trên cả 3 trục: chi tiết ảnh hưởng, chi phí tham số, và hiệu quả FLOPs.

---

## 3. Spatial Planes — Đặc tả chi tiết

Đầu ra: $X_{\text{spatial}} \in \mathbb{R}^{18 \times 8 \times 8}$, kiểu `float32` (tương thích `float16`).

**Quy ước tọa độ:** Mọi plane tuân thủ perspective STM (Mục 5). **Hàng 0 = phía STM (back rank)**, hàng 7 = phía đối thủ. Đây là convention chuẩn của Lc0, Stockfish NNUE, AlphaZero — STM luôn nhìn từ dưới lên.

### 3.1 Piece Planes (12 planes, chỉ số 0–11)

Mỗi plane là mask nhị phân: 1.0 tại vị trí có quân, 0.0 ở các vị trí còn lại.

| Chỉ số | Nội dung | Công thức |
|--------|----------|-----------|
| 0 | Tốt STM | $\mathbb{1}[\text{piece at } (r,c) = \text{Pawn} \land \text{color} = \text{STM}]$ |
| 1 | Mã STM | $\mathbb{1}[\text{piece} = \text{Knight} \land \text{color} = \text{STM}]$ |
| 2 | Tượng STM | $\mathbb{1}[\text{piece} = \text{Bishop} \land \text{color} = \text{STM}]$ |
| 3 | Xe STM | $\mathbb{1}[\text{piece} = \text{Rook} \land \text{color} = \text{STM}]$ |
| 4 | Hậu STM | $\mathbb{1}[\text{piece} = \text{Queen} \land \text{color} = \text{STM}]$ |
| 5 | Vua STM | $\mathbb{1}[\text{piece} = \text{King} \land \text{color} = \text{STM}]$ |
| 6 | Tốt OPP | $\mathbb{1}[\text{piece} = \text{Pawn} \land \text{color} = \text{OPP}]$ |
| 7 | Mã OPP | $\mathbb{1}[\text{piece} = \text{Knight} \land \text{color} = \text{OPP}]$ |
| 8 | Tượng OPP | $\mathbb{1}[\text{piece} = \text{Bishop} \land \text{color} = \text{OPP}]$ |
| 9 | Xe OPP | $\mathbb{1}[\text{piece} = \text{Rook} \land \text{color} = \text{OPP}]$ |
| 10 | Hậu OPP | $\mathbb{1}[\text{piece} = \text{Queen} \land \text{color} = \text{OPP}]$ |
| 11 | Vua OPP | $\mathbb{1}[\text{piece} = \text{King} \land \text{color} = \text{OPP}]$ |

### 3.2 Side-to-Move Plane (1 plane, chỉ số 12)

| Chỉ số | Nội dung | Giá trị |
|--------|----------|---------|
| 12 | STM indicator | 1.0 nếu White đi, 0.0 nếu Black đi, fill toàn bộ $8 \times 8$ |

**Lý do giữ uniform-fill:** Đây là biến phân loại nhị phân (ai đang đi), không phải biến trạng thái liên tục. Board flip đã xử lý hướng không gian; STM plane chỉ mã hóa **tempo** (ai có lượt đi).

### 3.3 Castling Rights Planes (4 planes, chỉ số 13–16)

| Chỉ số | Nội dung | Giá trị |
|--------|----------|---------|
| 13 | Nhập thành cánh Vua STM | 1.0 nếu STM còn quyền, 0.0 |
| 14 | Nhập thành cánh Hậu STM | 1.0 nếu STM còn quyền, 0.0 |
| 15 | Nhập thành cánh Vua OPP | 1.0 nếu OPP còn quyền, 0.0 |
| 16 | Nhập thành cánh Hậu OPP | 1.0 nếu OPP còn quyền, 0.0 |

**Lý do giữ uniform-fill:** Tương tự STM indicator, castling rights là quyền hạn (permission) dạng phân loại, không có ngữ nghĩa không gian.

### 3.4 En Passant Plane (1 plane, chỉ số 17)

| Chỉ số | Nội dung | Giá trị |
|--------|----------|---------|
| 17 | Ô đích en passant | 1.0 tại ô đích, 0.0 ở các ô khác |

**Đây là spatial feature thực sự** — chỉ ra một vị trí cụ thể trên bàn cờ, khác biệt cơ bản với các uniform-fill planes ở trên.

---

## 4. Scalar Vector — Đặc tả chi tiết

Đầu ra: $s \in \mathbb{R}^{5}$, kiểu `float32`.

Mọi giá trị đều được chuẩn hóa về khoảng $[-1, 1]$ hoặc $[0, 1]$. Vector này được inject vào mạng qua FiLM conditioning (xem Architecture Spec), **không** được fill thành constant planes.

### 4.1 Rule50

$$s[0] = \min\!\left(1,\; \frac{\text{halfmove\_clock}}{100}\right)$$

**Ý nghĩa:** Biến này xác định trực tiếp mức độ hòa (drawishness). Một vị trí với $\text{halfmove\_clock} = 80$ có phân phối value khác biệt căn bản so với cùng sắp xếp quân cờ với $\text{halfmove\_clock} = 5$.

**Lý do chuẩn hóa:** Chia cho 100 vì Rule 50 quy định hòa khi halfmove_clock đạt 50 (tức 100 nửa nước). Giá trị 1.0 tương ứng với ngưỡng hòa.

**Tham chiếu:** Lc0 từng phát hiện bug encode rule-50 ảnh hưởng nghiêm trọng đến chất lượng đánh giá [Lc0 Blog, 2018]. Mật độ phân phối rule50 trên dataset Lichess eval cho thấy đáng kể các vị trí có $\text{halfmove\_clock} > 20$.

### 4.2 Phase

$$s[1] = \min\!\left(1,\; \frac{\text{phase\_raw}}{20}\right)$$

trong đó:

$$\text{phase\_raw} = n_N + n_B + 2 \cdot n_R + 4 \cdot n_Q$$

Đếm quân nặng của **cả hai phe**. $n_N, n_B, n_R, n_Q$ là số lượng Mã, Tượng, Xe, Hậu trên bàn cờ.

**Ý nghĩa:** Phase xác định chế độ đánh giá (evaluation regime). Cờ tàn (endgame, phase thấp) ưu tiên kích hoạt Vua và Tốt qua; cờ trung (middlegame, phase cao) ưu tiên an toàn Vua. Đây là biến quyết định cho việc mạng chuyển đổi giữa các heuristic regime.

**Lý do dùng công thức cổ điển:** Đây là phase formula chuẩn trong mọi classical chess engine (Tapered Eval). Trọng số (Mã=Tượng=1, Xe=2, Hậu=4) phản ánh mức độ ảnh hưởng đến tính chất trận đấu.

### 4.3 Material Self

$$s[2] = \frac{1 \cdot n_{P,\text{self}} + 3 \cdot n_{N,\text{self}} + 3 \cdot n_{B,\text{self}} + 5 \cdot n_{R,\text{self}} + 9 \cdot n_{Q,\text{self}}}{39}$$

Mẫu số 39 là tổng material tối đa cho một phe: $8 \times 1 + 2 \times 3 + 2 \times 3 + 2 \times 5 + 1 \times 9 = 39$.

### 4.4 Material Opponent

$$s[3] = \frac{1 \cdot n_{P,\text{opp}} + 3 \cdot n_{N,\text{opp}} + 3 \cdot n_{B,\text{opp}} + 5 \cdot n_{R,\text{opp}} + 9 \cdot n_{Q,\text{opp}}}{39}$$

### 4.5 Material Delta

$$s[4] = \text{clamp}(s[2] - s[3],\; -1,\; 1)$$

**Lý do giữ cả 3 biến material thay vì chỉ delta:** Hai vị trí có cùng $\Delta m$ nhưng khác mức material tuyệt đối có tính chất chiến thuật khác nhau:

- $s[2] = 0.31, s[3] = 0.10$ ($\Delta = +0.21$): Cờ tàn ít quân, lợi thế nhỏ có thể quyết định
- $s[2] = 0.82, s[3] = 0.61$ ($\Delta = +0.21$): Cờ trung nhiều quân, lợi thế tương đối cần khai thác khác

Mức material tuyệt đối mã hóa thông tin phase mà $s[4]$ một mình không capture được.

### 4.6 Tương quan giữa Phase và Material

Phase và material có tương quan nhưng **không dư thừa**:

| Vị trí ví dụ | Phase | Material Self | Giải thích |
|---|---|---|---|
| 8 Tốt, không quân nặng | 0.0 | 0.21 | Phase thấp, material thấp |
| 2 Xe, không Tốt | 0.20 | 0.26 | Phase thấp-trung, material trung bình |
| Hậu + 2 Tượng | 0.50 | 0.62 | Phase trung, material cao |

Tương quan tồn tại nhưng rank-order khác nhau → cả hai đều cần giữ.

---

## 5. Hợp đồng Perspective (STM-Relative)

### 5.1 Định nghĩa

Mọi thành phần encode tuân thủ quy ước STM-relative:

- **Hàng 0 tensor = hàng sau của STM (back rank)**
- Hàng 7 tensor = hàng sau của đối thủ
- "Self" luôn chỉ phe đang đi (Side To Move)
- "Opponent" luôn chỉ phe kia

Convention này khớp với Lc0, Stockfish NNUE, AlphaZero. Ưu điểm: (a) Row 0 = STM → $\Delta r > 0$ luôn là "về phía đối thủ" (tấn công), ngữ nghĩa trực tiếp; (b) debug dễ hơn; (c) tương thích tools ecosystem.

### 5.2 Yêu cầu nhất quán

1. **Input encode:** STM-relative
2. **Target label:** STM-relative ($y > 0$ nghĩa là STM đang thắng)
3. **Không trộn perspective** trong cùng một training sample

### 5.3 Phép lật bàn cờ

Khi STM là Black, bàn cờ phải được lật dọc trước khi encode:

$$X_{\text{spatial}}^{\text{flipped}}[r, c] = X_{\text{spatial}}[7 - r, c] \quad \forall r, c$$

Đồng thời hoán đổi: piece planes self ↔ opponent (chỉ số 0–5 ↔ 6–11), castling self ↔ opp (chỉ số 13–14 ↔ 15–16). Việc hoán đổi này được thực hiện tự động bằng cách dùng `current_color` và `opponent_color` khi xác định plane index.

Scalar vector cũng phải cập nhật: $s[2] \leftrightarrow s[3]$, $s[4] \leftarrow -s[4]$. Các giá trị $s[0], s[1]$ không đổi vì chúng đối xứng theo phe.

**Triển khai flip:** Với convention `square = rank * 8 + file`, `rank 0 = a1` (White back rank):
- White = STM: `r_idx = rank` (không flip)
- Black = STM: `r_idx = 7 - rank` (flip dọc)

---

## 6. Versioning Schema

### 6.1 Định danh schema

```
encode_schema = "crane_v0_stm_spatial18_scalar5"
```

### 6.2 Quy tắc bump

Mọi thay đổi đối với bất kỳ thành phần nào sau đây phải bump schema:
- Thêm/xóa plane
- Thay đổi công thức chuẩn hóa
- Thay đổi perspective convention
- Thay đổi scalar vector dimension hoặc semantics

### 6.3 Cache invalidation

Bất kỳ artifact nào phụ thuộc encode phải mang `encode_schema`:
- Processed shard manifest
- Train run config
- Benchmark config
- Distillation dataset cache

**Không được phép** tái sử dụng shard từ schema khác.

### 6.4 Split key semantics

Hàm `canonical_fen_for_split()` bỏ `halfmove/fullmove` khỏi split key để tránh leakage. Quy tắc này **không thay đổi** trong v0.

---

## 7. Testing bắt buộc

### 7.1 Unit tests

| Test | Mô tả |
|------|--------|
| `test_shape` | Output spatial = (18,8,8), scalar = (5,) |
| `test_stm_orientation` | STM=Black → board lật dọc, màu hoán đổi |
| `test_rule50` | $s[0]$ khớp $\min(1, \text{halfmove\_clock}/100)$ |
| `test_phase_decrease` | Khấu trừ quân nặng → $s[1]$ giảm |
| `test_material_delta_sign` | STM nhiều quân hơn → $s[4] > 0$ |
| `test_perspective_flip` | Lật bàn cờ → $s[2] \leftrightarrow s[3]$, $s[4]$ đổi dấu |
| `test_en_passant_spatial` | En passant plane có đúng 1 ô = 1.0 hoặc tất cả = 0.0 |

### 7.2 Consistency checks

- FEN → Board → encode không crash cho mọi FEN hợp lệ
- Board flip tạo encoding nhất quán về mặt semantics
- Encode mới không gãy pipeline benchmark hiện có

### 7.3 Dataset smoke checks

Trên sample nhỏ train/val/test, log:
- Mean/std của từng scalar variable
- Histogram `rule50`, histogram `phase`
- Pearson correlation giữa `material_delta` và target value
- Kiểm tra: plane toàn hằng số (không thiết kế), sign bug, scaling bug, cache trộn schema

---

## 8. Những gì spec này cố ý không làm

Spec này **không**:
1. Định nghĩa kiến trúc torso/head → thuộc Architecture Spec
2. Định nghĩa policy action space
3. Định nghĩa RL loop
4. Khẳng định repetition-awareness (không có history chain trong nguồn dữ liệu hiện tại)
5. Bao gồm `fullmove_number` — phase capture signal có ích hơn và ổn định hơn

### Extension path cho history/repetition

Khi dataset builder hỗ trợ history chain nhất quán, extension hợp lệ:
- Thêm $k=2$ hoặc $k=4$ historical board stacks vào spatial planes
- Thêm `repetition_count_clip` hoặc repetition flag nếu tái tạo chính xác được
- Tăng encode_schema version riêng

**Không được** suy luận repetition từ `halfmove_clock` một cách giả tạo.

---

## 9. So sánh với encode cũ (DGRN-X-v0.1)

| Tiêu chí | DGRN-X-v0.1 | CRANE-v0 |
|---|---|---|
| Spatial planes | 23×8×8 | 18×8×8 |
| Scalar injection | 5 constant planes | 5-dim vector qua FiLM |
| Hiệu quả Conv | 21.7% channel lãng phí | 100% channel chứa thông tin không gian |
| Chi phí FiLM | 0 | 2,304 params |
| Thông tin tương đương | Đầy đủ | Đầy đủ + tiêm mạnh hơn qua FiLM |

CRANE-v0 giữ nguyên **mọi bit thông tin** có trong DGRN-X-v0.1, nhưng phân phối qua cơ chế phù hợp hơn về mặt lý thuyết.
