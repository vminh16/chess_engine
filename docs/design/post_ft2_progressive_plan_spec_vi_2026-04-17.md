# Đặc Tả Chương Trình Thực Nghiệm Sau FT2 (Theo Pha, Ưu Tiên Chi Phí)

**Ngày:** 2026-04-17  
**Ngôn ngữ:** Tiếng Việt  
**Trạng thái:** Đề xuất thực thi  
**Phạm vi:** Teacher network `architecture_v2`, nhánh nghiên cứu hậu `FT2`

---

## 1) Quyết định điều hành

### 1.1 Kết luận chốt

Kế hoạch theo thứ tự:

1. chạy line rẻ theo **report 1** trước,
2. nếu không pass thì chuyển sang **head redesign**,
3. cuối cùng mới **re-encode dataset**,

là **hợp lý**, **khoa học**, và **phù hợp với chi phí thực tế** của repo hiện tại.

Tuy nhiên, cần một điều chỉnh quan trọng:

- pha `re-encode` không nên bắt đầu bằng rebuild toàn bộ dataset ngay lập tức,
- mà nên có một **pilot rebuild nhỏ** trước khi commit vào full rebuild.

Lý do: với dữ liệu hiện tại, `encode refresh` là nhánh có **động cơ lý thuyết mạnh**, nhưng cũng là nhánh có **chi phí vận hành cao nhất** và **thời gian phản hồi chậm nhất**. Nếu bỏ qua bước pilot, rủi ro đốt compute/I/O/storage quá lớn trong khi causal attribution vẫn mờ.

### 1.2 Thứ tự can thiệp được chốt

Thứ tự chính thức đề xuất:

1. **Pha A - Report 1 core line**
   - `8b/128d`
   - giữ data hiện tại
   - giữ head baseline, chưa mở head redesign
   - mục tiêu: kiểm tra giả thuyết `shrink-first` và `coherence objective-data` với chi phí thấp nhất

2. **Pha B - Head program**
   - chỉ mở khi Pha A không pass
   - ưu tiên đầu tiên: head đơn giản hơn hoặc tách chế độ
   - mục tiêu: đánh đúng vào nơi xung đột gradient đang đo được

3. **Pha C - Encode refresh program**
   - chỉ mở khi Pha B vẫn không pass
   - bắt đầu bằng pilot rebuild
   - mục tiêu: xử lý representation aliasing / missing-state bằng thay đổi dữ liệu có chủ đích

### 1.3 Vì sao đây là thứ tự tốt nhất hiện tại

Thứ tự này tối ưu đồng thời theo bốn tiêu chí:

1. **chi phí compute**
2. **độ sạch khoa học của attribution**
3. **xác suất sửa đúng failure thật**
4. **thời gian phản hồi để ra quyết định tiếp**

Về mặt thực nghiệm nội bộ:

- FT2 hiện tại đã chứng minh rằng `full rerun center-aware` không tự động tạo ra checkpoint vừa tốt ở A vừa tốt ở B.
- FT2 hiện tại cũng cho thấy **trade-off thật**, không phải đơn thuần “train thêm sẽ tốt”.
- Vì vậy, bước tiếp theo phải là **can thiệp ít confound hơn**, không phải gom thêm nhiều thay đổi vào một mega-run.

Về mặt lý thuyết:

- Khi nhiều mục tiêu cạnh tranh, bài toán trở thành một dạng **multi-objective optimization** và việc không thể đồng thời tối ưu tất cả metric là hiện tượng bình thường chứ không phải bất thường; xem [Sener & Koltun 2018](https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization).
- Vì vậy, chương trình tốt nhất sau FT2 là chương trình **theo pha**, mỗi pha kiểm tra một giả thuyết cụ thể với số biến thay đổi tối thiểu.

---

## 2) Cơ sở bằng chứng bắt buộc

## 2.1 FT2 chưa đạt mục tiêu

Run hiện tại là `dgrn_5m_ft2_t4_run1`.

Từ [`runs/dgrn_5m_ft2_t4_run1/reports/history.csv`](../../runs/dgrn_5m_ft2_t4_run1/reports/history.csv) và [`runs/dgrn_5m_ft2_t4_run1/reports/decision_summary.json`](../../runs/dgrn_5m_ft2_t4_run1/reports/decision_summary.json):

- `has_best_gate = false`
- best A nằm ở `epoch 16`:
  - `oracle_midband_mae_sum_stable = 0.6059230709963943`
  - `oracle_stable_0.7_slope = 0.5614989093319399`
- best B nằm ở `epoch 11`:
  - `center_score = 0.33054353284792715`

Trong khi gate chấp nhận từ `report 1` là:

1. `M <= 0.5997118252`
2. `S >= 0.5977681498`

Nghĩa là:

- FT2 đã **tiến gần gate A**, nhưng **vẫn trượt**,
- FT2 đã **đưa B xuống ngang FT1-best**, nhưng **không thắng rõ**,
- quan trọng nhất: **best A và best B nằm ở hai epoch khác nhau**.

Đây là bằng chứng trực tiếp của một **Pareto trade-off**.

## 2.2 Report 1 đáng tin ở phần chẩn đoán

Từ [`docs/reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md`](../reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md):

- `shrink-first` được đề xuất như nhánh rẻ nhất và ít confound nhất
- xung đột gradient mạnh hơn ở **head** so với backbone
- active pipeline hiện tại không còn bug `white-label vs stm-input`
- data hiện tại có `exact-zero mass` thật, không phải lỗi rounding giả

Đánh giá của tôi:

- phần **diagnosis** của report 1 có độ tin cậy **cao**
- phần **prescription** của report 1 có độ tin cậy **trung bình đến cao**
- vì nó đúng về thứ tự kiểm định, nhưng chưa có run nội bộ nào chứng minh trước rằng `8b/128d` chắc chắn sẽ pass

## 2.3 Encode hiện tại không sai POV, nhưng thiếu state

Từ [`representation/encode.py`](../../representation/encode.py) và [`core/board.py`](../../core/board.py):

- `Board` có `halfmove_clock` và `fullmove_number`
- nhưng `encode_board(...)` hiện chỉ encode:
  - piece planes
  - side-to-move
  - castling rights
  - en passant

Nghĩa là input hiện tại **không sai về perspective**, nhưng **under-specified** về state.

Đây là điểm rất quan trọng, vì với supervised regression:

$$
f^*(x) = \mathbb{E}[y \mid x]
$$

Nếu hai trạng thái cờ khác nhau bị map vào cùng một tensor `x` nhưng target khác nhau, thì lỗi không thể loại bỏ hoàn toàn là:

$$
\mathbb{E}\big[(y - f^*(x))^2 \mid x\big] = \mathrm{Var}(y \mid x) > 0
$$

Nói ngắn gọn:

- nếu representation làm **mất thông tin trạng thái**, mô hình tốt nhất cũng chỉ học được **trung bình có điều kiện**,
- phần phương sai còn lại trở thành **irreducible error** do aliasing của input.

Với chess, `halfmove_clock` ảnh hưởng trực tiếp tới drawishness qua luật `50-move`, nên đây là một missing-state có ý nghĩa domain thật.

## 2.4 Head hiện tại là điểm nghi ngờ mạnh hơn backbone

Từ [`experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv`](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv) và [`docs/reports/training_audit_report.md`](../reports/training_audit_report.md):

- cặp `center_raw_0_005` vs `mid_05_07` có:
  - `cosine_backbone = -0.2792`
  - `cosine_head = -0.7239`

Điều này cho thấy:

- xung đột có ở backbone,
- nhưng mạnh hơn nhiều ở **head**.

Ngoài ra, với cấu hình `20b/256d`, head hiện tại có quy mô lớn:

- `ResidualGainValueHead(256, 128)` = **1,272,578** tham số
- `SimplifiedGlobalHead(256, 128)` = **73,985** tham số
- `RegimeSeparatedHead(256, 128)` = **82,306** tham số

Ở cấu hình `8b/128d`:

- `ResidualGainValueHead(128, 64)` = **318,850** tham số
- `SimplifiedGlobalHead(128, 64)` = **18,561** tham số
- `RegimeSeparatedHead(128, 64)` = **20,674** tham số

Nghĩa là:

- chỉ riêng đổi head đã có thể giảm mạnh capacity của readout,
- trong khi vẫn giữ backbone gần như nguyên để causal attribution rõ ràng.

---

## 3) Nguyên tắc thiết kế chương trình mới

## 3.1 Không gom nhiều thay đổi vào một run

Mỗi pha chỉ nên thay đổi **một trục chính**:

1. trục `capacity / optimization geometry`
2. trục `head readout structure`
3. trục `representation`

Không nên chạy ngay một run kiểu:

- shrink
- đổi head
- đổi normalization
- đổi sampling
- đổi encode

trong cùng một lần.

Lý do:

- nếu run thắng, bạn không biết yếu tố nào là nguyên nhân,
- nếu run thua, bạn cũng không biết nên rollback phần nào.

## 3.2 Ưu tiên biến rẻ trước biến đắt

Chi phí tăng theo thứ tự:

1. `shrink-first` run
2. head redesign
3. pilot re-encode
4. full dataset re-encode

Vì vậy, thứ tự can thiệp cũng nên đi theo thứ tự này.

## 3.3 Ưu tiên giả thuyết đã có bằng chứng nội bộ

Mức ưu tiên phải theo:

1. **head conflict**: đã đo trực tiếp
2. **trade-off A/B**: đã đo trực tiếp qua FT2
3. **encode missing-state**: có cơ sở lý thuyết mạnh + bằng chứng code-path
4. **backbone normalization / GN / transformer / graph readout**: chỉ nên mở sau khi các giả thuyết trên không đủ

---

## 4) Kế hoạch thực nghiệm theo pha

## 4.1 Pha A - Report 1 core line

### Mục tiêu

Kiểm tra xem failure hiện tại có được giải đáng kể chỉ bằng:

1. giảm capacity từ `20b/256d` xuống `8b/128d`,
2. làm objective-data coherence tốt hơn,

mà **không cần đổi representation** và **chưa cần head sáng tạo** hay không.

### Lý do chọn trước

Đây là nhánh:

- rẻ nhất,
- sạch nhất,
- bám đúng diagnosis mạnh nhất của report 1,
- và phù hợp với tình trạng hiện tại: FT2 đã cho thấy line lớn hiện tại chưa ra checkpoint chấp nhận được.

### Các run trong Pha A

**A1. Shrink-first baseline**

1. Model: `8b/128d`
2. Head: giữ `ResidualGainValueHead`
3. Objective: `FT1-style objective` như report 1 đề xuất
4. Thời lượng: `12-15` epoch
5. Mục tiêu: xem A và B có cùng đi đúng hướng không

**A2. Coherence run**

1. Giữ `8b/128d`
2. Giữ head như A1
3. Thêm `sign-stratified sampling` hoặc cơ chế tương đương
4. Thời lượng: `10-12` epoch
5. Mục tiêu: cải thiện slope/directional consistency mà không làm center nổ lại

### Công thức khái niệm cho sign-stratified sampling

Giả sử chia dữ liệu thành các bucket theo dấu và biên độ:

$$
\mathcal{B} = \{B_1, B_2, \ldots, B_K\}
$$

với xác suất lấy mẫu bucket:

$$
P(B_k) \propto w_k
$$

và xác suất chọn mẫu trong bucket:

$$
P(i \mid i \in B_k) = \frac{1}{|B_k|}
$$

thì xác suất lấy mẫu tổng là:

$$
P(i) = \frac{w_k}{\sum_j w_j} \cdot \frac{1}{|B_k|}
$$

Ý nghĩa:

- ép optimizer nhìn thấy đủ cả vùng dương/âm và các dải biên độ quan trọng,
- giảm nguy cơ gradient bị thống trị bởi vùng center đông nhưng ít thông tin direction.

Đây là can thiệp **ở phân phối huấn luyện**, không phải thay đổi label.

### Tiêu chí stop/go của Pha A

Điều kiện pass:

1. `M <= 0.5997118252`
2. `S >= 0.5977681498`
3. `center_score <= 0.3409` hoặc ít nhất không xấu hơn FT1-best quá `+0.01`

Điều kiện chuyển sang Pha B:

1. sau `epoch 8` vẫn không có xu hướng đóng A gap rõ ràng,
2. hoặc slope tiếp tục trượt dù center tốt,
3. hoặc best A và best B vẫn tách xa nhau theo epoch.

### Ghi chú vận hành

Tôi **không** khuyến nghị đưa `RegimeSeparatedHead` vào ngay Pha A mặc dù report 1 có nhắc tới.

Lý do:

- bạn đã chốt chiến lược “report 1 trước, head sau”,
- và về mặt khoa học, tách `shrink/coherence` ra khỏi `head redesign` làm attribution sạch hơn.

---

## 4.2 Pha B - Head program

Pha này chỉ mở nếu Pha A không pass.

## 4.2.1 Head H1 - SimplifiedGlobalHead

### Định nghĩa

Head này đã có sẵn ở [`model/architecture_v2/head.py`](../../model/architecture_v2/head.py).

Với feature map backbone $X \in \mathbb{R}^{C \times 8 \times 8}$:

1. average pooling:
$$
g_{\text{avg}} = \mathrm{AvgPool}(X) \in \mathbb{R}^{C}
$$

2. max pooling:
$$
g_{\text{max}} = \mathrm{MaxPool}(X) \in \mathbb{R}^{C}
$$

3. ghép vector:
$$
p = [g_{\text{avg}} ; g_{\text{max}}] \in \mathbb{R}^{2C}
$$

4. MLP ra logit:
$$
z = f_{\theta}(p)
$$

5. output:
$$
\hat{y} = \tanh(z)
$$

### Lý do hợp lý

Head hiện tại có một nhánh flatten lớn:

$$
\mathbb{R}^{64 \cdot h} \to \mathbb{R}^{h}
$$

với `h = 128` ở line lớn, tức là phép chiếu `8192 -> 128`.

Điều này tạo ra:

1. coupling dày đặc giữa toàn bộ 64 ô,
2. một readout bottleneck có capacity lớn,
3. một nơi dễ tích tụ gradient conflict.

`SimplifiedGlobalHead` loại bỏ hoàn toàn nhánh flatten này.

### Vì sao hợp chess

Nếu backbone đã đủ receptive field trên bàn cờ `8x8`, thì:

- phần “nhìn quan hệ không gian cục bộ” nên do backbone làm,
- head chỉ nên làm nhiệm vụ **readout toàn cục**.

Theo đúng logic đó, global head là một baseline rất sạch.

### Độ tin cậy

**Cao** như một bước thử nghiệm sau Pha A.  
Lý do: vừa rẻ, vừa bám đúng failure đo được ở head.

## 4.2.2 Head H2 - RegimeSeparatedHead

### Định nghĩa

Head này cũng đã có sẵn ở [`model/architecture_v2/head.py`](../../model/architecture_v2/head.py).

Với pooled vector $p$:

1. shared trunk:
$$
r = \phi(W_s p + b_s)
$$

2. magnitude branch:
$$
z_m = f_m(r)
$$

3. sign branch:
$$
z_s = f_s(r)
$$

4. combined logit:
$$
z = z_m \cdot \tanh(z_s)
$$

5. output:
$$
\hat{y} = \tanh(z)
$$

### Ý nghĩa toán học

Một scalar value đang đồng thời phải encode:

1. **direction**: phe nào tốt hơn
2. **magnitude**: tốt hơn bao nhiêu

Nếu hai vai trò này dùng cùng một affine path, gradient có thể xung đột.

Tách `sign` và `magnitude` ra cho phép:

- gradient về hướng ưu tiên branch `sign`,
- gradient về độ lớn ưu tiên branch `magnitude`,
- trong khi vẫn dùng chung một trunk nhỏ phía trước.

### Vì sao hợp chess

Chess value thực tế đúng là sự kết hợp của:

1. “bên nào đang tốt hơn”
2. “mức chênh bao nhiêu”

Ở vùng center, magnitude nhỏ nhưng sign vẫn quan trọng.  
Ở vùng mid/tails, magnitude quan trọng hơn.

Head này vì vậy có inductive bias phù hợp với bài toán hơn một scalar affine path đơn.

### Độ tin cậy

**Trung bình đến cao**.  
Mạnh hơn ở mặt lý thuyết so với global head, nhưng confound cũng lớn hơn vì thay đổi semantics của readout.

## 4.2.3 Head H3 - Phase-conditioned tapered head

### Trạng thái

Đây là **ý tưởng sáng tạo có grounding domain**, nhưng **chưa có thực nghiệm nội bộ**.

### Định nghĩa

Từ chính tensor hiện tại, ta có thể tính một scalar phase $p \in [0,1]$ từ số quân và trọng số quân:

$$
\mathrm{phase\_raw} = n_N + n_B + 2n_R + 4n_Q
$$

và chuẩn hóa:

$$
p = \min\left(1, \frac{\mathrm{phase\_raw}}{20}\right)
$$

trong đó `20` bám theo logic phase đang dùng trong preprocessing.

Sau đó dùng hai nhánh readout:

$$
z_{\text{mid}} = f_{\text{mid}}(h), \quad z_{\text{end}} = f_{\text{end}}(h)
$$

và một gate:

$$
g = \sigma(a p + b)
$$

hoặc gate học được:

$$
g = \sigma(f_g(h, p))
$$

Output:

$$
z = g \cdot z_{\text{mid}} + (1-g) \cdot z_{\text{end}}
$$

$$
\hat{y} = \tanh(z)
$$

### Vì sao hợp lý

Trong chess, hàm value không đồng nhất theo phase:

- opening thiên về development, king safety, initiative,
- endgame thiên về opposition, passed pawn, king activity, 50-move drawishness.

Một head duy nhất cho toàn bộ phase buộc mô hình học một hàm quá “toàn năng”.

`Phase-conditioned tapered head` là cách mềm để xấp xỉ:

$$
f(x) \approx g(p) f_{\text{mid}}(x) + (1-g(p)) f_{\text{end}}(x)
$$

tức là một **mixture-of-experts nhẹ** theo phase.

### Vì sao phù hợp với encode hiện tại

Điểm quan trọng: phase có thể suy ra từ chính piece planes hiện có.  
Nghĩa là ý tưởng này **không bắt buộc** rebuild dataset.

### Độ tin cậy

**Trung bình**.  
Đây là ý tưởng tốt về mặt chess bias, nhưng vẫn là hypothesis.

## 4.2.4 Head H4 - Value + uncertainty / confidence head

### Trạng thái

Đây là ý tưởng hợp lý về mặt production/search, nhưng **không nên là head can thiệp đầu tiên** nếu mục tiêu trước mắt là pass gate FT1-style.

### Định nghĩa

Head ra hai scalar:

1. mean/value:
$$
\hat{y} = \tanh(z_{\mu})
$$

2. log-variance:
$$
s = z_{\sigma}
$$

Loss heteroscedastic regression:

$$
\mathcal{L}(x, y) = \exp(-s)(y - \hat{y})^2 + s
$$

Theo [Kendall & Gal 2017](https://arxiv.org/abs/1703.04977), biểu thức này tương đương việc học **aleatoric uncertainty** trong regression.

### Ý nghĩa

Nếu target search-derived có noise không đồng nhất theo trạng thái, thì một scalar duy nhất là chưa đủ.

Head này cho phép mô hình học:

1. giá trị dự đoán,
2. độ bất định nội tại của chính target.

### Vì sao hợp chess

Có những thế cờ:

- stable,
- tactical,
- depth-sensitive,
- gần draw rule,

với cùng độ lớn `|y|` nhưng độ tin cậy của target rất khác nhau.

Nếu sau này đưa teacher vào search, confidence head có thể dùng để điều tiết trust giữa:

1. neural evaluator
2. search/classical evaluator

### Hạn chế

- không đảm bảo giúp pass gate ngay,
- thêm một output head đồng nghĩa thêm một lớp phức tạp objective.

### Độ tin cậy

**Trung bình** cho mục tiêu calibration/search.  
**Thấp hơn** cho mục tiêu “pass gate càng sớm càng tốt”.

## 4.2.5 Head H5 - Piece-relational readout

### Trạng thái

Đây là ý tưởng **research-grade**, chưa nên đưa vào ngay sau Pha A nếu mục tiêu là phản hồi nhanh.

### Định nghĩa khái niệm

Từ position hiện tại, tạo tập token quân:

$$
\{h_i\}_{i=1}^{P}, \quad P \le 32
$$

với mỗi token biểu diễn:

1. loại quân,
2. màu,
3. vị trí,
4. feature cục bộ từ backbone.

Sau đó dùng message passing / self-attention:

$$
h_i^{(\ell+1)} = \psi\left(h_i^{(\ell)}, \sum_{j \ne i} \alpha_{ij} \, \phi(h_i^{(\ell)}, h_j^{(\ell)}, e_{ij})\right)
$$

với:

- $e_{ij}$: đặc trưng quan hệ giữa quân `i` và `j`
- $\alpha_{ij}$: trọng số attention hay edge weight

Cuối cùng pool toàn cục ra scalar value.

### Vì sao hợp chess

Chess là bài toán quan hệ:

1. pin
2. skewers
3. overloaded defender
4. battery
5. discovered attack
6. king shelter

Những mẫu này là quan hệ giữa quân, không chỉ là texture trên lưới.

Theo [Battaglia et al. 2018](https://arxiv.org/abs/1806.01261), graph/relational inductive bias phù hợp với các bài toán có cấu trúc tương tác giữa thực thể.

### Hạn chế

- engineering cost cao hơn rõ rệt,
- thêm nhiều biến mới,
- khó làm attribution hơn các head nhẹ ở trên.

### Độ tin cậy

**Trung bình về mặt lý thuyết**, **thấp hơn về mặt ưu tiên thực thi**.

---

## 4.3 Pha C - Encode refresh program

Pha này chỉ mở nếu:

1. Pha A không pass,
2. Pha B cũng không pass,
3. hoặc evidence từ Pha B cho thấy head đã bớt conflict nhưng gate vẫn không đóng.

## 4.3.1 Mục tiêu

Kiểm tra giả thuyết rằng một phần failure hiện tại đến từ **representation aliasing** chứ không chỉ do kiến trúc và objective.

## 4.3.2 Nguyên tắc

Không rebuild toàn bộ dataset ngay.  
Chia làm 2 bước:

1. **C1 - pilot rebuild**
2. **C2 - full rebuild**

### C1 - Pilot rebuild

Rebuild một tập con đủ lớn để so sánh có nghĩa, nhưng vẫn rẻ hơn full line.

Mục tiêu:

- xác minh encode patch có tạo tín hiệu đúng không,
- trước khi commit vào rebuild lớn.

### C2 - Full rebuild

Chỉ chạy nếu pilot cho tín hiệu thắng rõ.

## 4.3.3 Encode patch tối thiểu đề xuất

Ưu tiên thêm đúng những state có ý nghĩa nhất:

1. **halfmove / no-progress plane**
2. **phase plane**
3. **fullmove plane** là tùy chọn thấp ưu tiên hơn

### Plane 1 - No-progress / halfmove clock

Định nghĩa:

$$
r_{50} = \min\left(1, \frac{\mathrm{halfmove\_clock}}{100}\right)
$$

và fill toàn plane bằng giá trị $r_{50}$.

Lý do:

- liên quan trực tiếp tới luật 50 nước,
- ảnh hưởng đúng vùng drawish / center,
- tồn tại sẵn trong `Board` nhưng chưa được encode.

### Plane 2 - Phase

Định nghĩa:

$$
\mathrm{phase\_raw} = n_N + n_B + 2n_R + 4n_Q
$$

$$
p = \min\left(1, \frac{\mathrm{phase\_raw}}{20}\right)
$$

Lý do:

- value function của chess thay đổi theo phase,
- phase đang có mặt trong logic preprocessing,
- chi phí encode thêm thấp.

### Plane 3 - Fullmove number

Định nghĩa ví dụ:

$$
m = \min\left(1, \frac{\mathrm{fullmove\_number}}{M}\right)
$$

với `M` là hằng số cắt ngưỡng vận hành.

Lý do:

- có thể mang thông tin tiến trình game,
- nhưng rủi ro mang bias dataset cao hơn `halfmove`.

Vì vậy đây là plane **ưu tiên thấp** hơn.

## 4.3.4 Vì sao encode refresh có cơ sở mạnh nhưng đặt sau

Theo [AlphaZero supplementary](https://schachklub.ws/wp-content/uploads/2018/12/alfazero_supplementary_data.pdf) và [Lc0 Alphazero Primer](https://lczero.org/dev/lc0/search/alphazero/), các chess network mạnh thường dùng input giàu state hơn, bao gồm:

1. side to move
2. special rules
3. repetition / history
4. no-progress count
5. move count

Repo hiện tại chưa encode đủ các thành phần này.

Tuy nhiên, hai điểm phải nói thẳng:

1. pipeline hiện tại **không lưu history stack**, nên không thể thêm repetition/history đúng nghĩa nếu upstream raw data không có chuỗi trạng thái;
2. rebuild dataset là can thiệp đắt nhất.

Vì vậy:

- encode refresh có **động cơ lý thuyết mạnh**,
- nhưng vẫn nên đến **sau** head line vì cost lớn hơn nhiều.

## 4.3.5 Chi phí tính toán thêm là nhỏ ở model, lớn ở data

Nếu thêm `k` plane vào stem hiện tại:

$$
\Delta P_{\text{stem}} = k \cdot C \cdot 3 \cdot 3
$$

với `C = 256`, `k = 2`:

$$
\Delta P_{\text{stem}} = 2 \cdot 256 \cdot 9 = 4608
$$

Đây là rất nhỏ so với tổng `9,047,682` tham số.

Nghĩa là:

- chi phí **model** tăng không đáng kể,
- chi phí chính nằm ở **rebuild dataset và I/O**, không phải số tham số.

---

## 5) Backbone có nên là pha riêng không

### Kết luận

**Không nên** là pha đầu tiên.

### Lý do

1. evidence nội bộ hiện tại nghiêng về **head** hơn backbone
2. FT2 đã chứng minh trade-off nhưng chưa chỉ ra backbone là root cause số 1
3. đổi backbone sớm sẽ làm causal attribution bẩn

### Những gì chỉ nên thử sau này

1. `GroupNorm` A/B
2. backbone shrink sâu hơn
3. adapter nhỏ theo phase/regime

### Cơ sở lý thuyết cho GN

Theo [Wu & He 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf), GroupNorm ổn định hơn BatchNorm khi batch nhỏ vì không phụ thuộc batch statistics.

Theo [PyTorch BatchNorm2d docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d), `BatchNorm2d` dùng running statistics khi `track_running_stats=True`.

Trong repo này, GN là một nhánh **hợp lý để thử**, nhưng hiện **chưa có bằng chứng nội bộ** để nâng nó lên ưu tiên số 1.

---

## 6) Ma trận quyết định chính thức

## 6.1 Pha A

1. `A1`: `8b/128d + ResidualGainValueHead + baseline objective`
2. `A2`: `8b/128d + ResidualGainValueHead + sign-stratified sampling`

Luật ra quyết định:

1. nếu `A1` pass: dừng, scale line này
2. nếu `A1` fail nhưng `A2` đóng A gap rõ: tiếp tục line sampling
3. nếu `A1/A2` đều fail: mở Pha B

## 6.2 Pha B

Ưu tiên:

1. `H1 - SimplifiedGlobalHead`
2. `H2 - RegimeSeparatedHead`
3. `H3 - Phase-conditioned tapered head`
4. `H4 - Value + uncertainty head`
5. `H5 - Piece-relational readout`

Luật ra quyết định:

1. nếu `H1` đã cải thiện cả A lẫn B: giữ `H1`, không cần mở phức tạp hơn
2. nếu `H1` giữ center tốt nhưng slope chưa đủ: thử `H2`
3. nếu `H2` vẫn không giải được trade-off: xét `H3`
4. `H4/H5` chỉ mở nếu mục tiêu chuyển từ “pass gate nhanh” sang “teacher/search robust hơn”

## 6.3 Pha C

1. `C1`: pilot rebuild với encode patch tối thiểu
2. `C2`: full rebuild chỉ nếu `C1` thắng rõ

Luật ra quyết định:

1. nếu pilot không cho directional win rõ ràng: dừng encode line
2. nếu pilot tốt nhưng margin nhỏ: chỉ scale lên full rebuild nếu head line đã bế tắc
3. nếu pilot tốt rõ ràng trên cả gate và center: mở full rebuild

---

## 7) Mức tự tin của từng nhánh

### 7.1 Mức tự tin hiện tại

1. **Cao**
   - bắt đầu bằng line rẻ theo report 1
   - giữ backbone gần như nguyên trong pha đầu

2. **Trung bình đến cao**
   - head simplification là nhánh tiếp theo hợp lý nhất nếu Pha A fail
   - `RegimeSeparatedHead` có cơ sở tốt hơn một số ý tưởng sáng tạo khác

3. **Trung bình**
   - encode refresh sẽ sửa một phần failure nếu đúng là representation aliasing đang đáng kể
   - nhưng chi phí rebuild lớn nên cần pilot trước

4. **Thấp đến trung bình**
   - uncertainty head sẽ giúp pass gate nhanh hơn
   - piece-relational head là đường nhanh nhất đến checkpoint pass

### 7.2 Điều tôi không thể cam kết

Tôi **không thể** cam kết rằng:

1. line `8b/128d` sẽ chắc chắn pass
2. head sáng tạo sẽ chắc chắn tốt hơn head có sẵn
3. encode refresh sẽ chắc chắn sửa hết failure

Những điều đó hiện **chưa được chứng minh** bằng artifact nội bộ.

Điều tôi có thể nói với độ tin cậy cao là:

1. thứ tự `report 1 -> head -> encode` là thứ tự **đúng về mặt ra quyết định**
2. đây là thứ tự tối ưu nhất hiện tại theo cả `science` lẫn `cost`

---

## 8) Những điều không nên làm

1. Không mở full rebuild dataset ngay từ đầu.
2. Không chạy một mega-run trộn nhiều thay đổi.
3. Không chuyển sang backbone rewrite lớn khi head/encode chưa được kiểm tra đủ.
4. Không thêm history/repetition vào encode nếu upstream raw pipeline không có dữ liệu lịch sử thật.
5. Không dùng “cảm giác hợp lý” thay cho gate và artifact.

---

## 9) Khuyến nghị cuối cùng

### 9.1 Khuyến nghị chính thức

Tôi chấp thuận kế hoạch của bạn, với cách triển khai như sau:

1. **chạy Pha A trước** theo hướng rẻ của report 1
2. **nếu fail thì chuyển sang Pha B** với head program
3. **nếu vẫn fail thì mới mở Pha C** để encode refresh

### 9.2 Lý do lựa chọn

Đây là phương án:

1. **ít tốn chi phí nhất** trên mỗi giả thuyết được kiểm tra,
2. **giữ causal attribution rõ nhất**,
3. **bám evidence nội bộ tốt nhất**,
4. **không đẩy bạn vào full dataset rebuild quá sớm**.

### 9.3 Một câu chốt

Nếu phải diễn đạt rất ngắn:

- **report 1** là nơi nên bắt đầu,
- **head** là nơi nên tấn công tiếp theo nếu report 1 không đủ,
- **encode** là nơi có động cơ lý thuyết mạnh nhất nhưng chi phí cao nhất, nên đặt cuối và phải có pilot.

---

## 10) Nguồn

### 10.1 Nguồn nội bộ repo

1. [`docs/reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md`](../reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md)
2. [`runs/dgrn_5m_ft2_t4_run1/reports/history.csv`](../../runs/dgrn_5m_ft2_t4_run1/reports/history.csv)
3. [`runs/dgrn_5m_ft2_t4_run1/reports/decision_summary.json`](../../runs/dgrn_5m_ft2_t4_run1/reports/decision_summary.json)
4. [`docs/reports/training_audit_report.md`](../reports/training_audit_report.md)
5. [`representation/encode.py`](../../representation/encode.py)
6. [`core/board.py`](../../core/board.py)
7. [`model/architecture_v2/head.py`](../../model/architecture_v2/head.py)
8. [`model/architecture_v2/model.py`](../../model/architecture_v2/model.py)
9. [`data/process_data.ipynb`](../../data/process_data.ipynb)
10. [`docs/reports/project_report_vi.md`](../reports/project_report_vi.md)

### 10.2 Nguồn ngoài

1. [Sener & Koltun 2018 - Multi-Task Learning as Multi-Objective Optimization](https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization)
2. [Wu & He 2018 - Group Normalization](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)
3. [PyTorch BatchNorm2d Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d)
4. [Kendall & Gal 2017 - What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
5. [Battaglia et al. 2018 - Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
6. [AlphaZero Supplementary Data](https://schachklub.ws/wp-content/uploads/2018/12/alfazero_supplementary_data.pdf)
7. [Lc0 Alphazero Primer](https://lczero.org/dev/lc0/search/alphazero/)
