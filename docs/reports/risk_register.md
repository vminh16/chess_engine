# DGRNChessNet Risk Report (Corrected)

**Model:** DGRNChessNet / PhantomChessNet  
**Architecture in repo:** 20 `DFGBlock`, `hidden_dim=256`, `ContextGatedHead`  
**Target:** validation MSE `< 0.05`  
**Evidence used:** bucketed validation analysis supplied by user, [history (1).csv](C:/Users/USER/Downloads/history%20(1).csv), [history.json](C:/Users/USER/Downloads/history.json), and the current code in `model/architecture`.

## 1. Executive Summary

Kết luận chính của báo cáo này là:

- **Backbone không phải bottleneck chính.**
- **Head hiện tại là bottleneck kiến trúc quan trọng nhất.**
- Cụ thể, `ContextGatedHead` tạo ra một **low-gain attractor**: mô hình học đúng dấu nhưng co biên độ dự đoán về gần `0`.
- `tanh` làm nghiệm này ổn định hơn, nhưng **không phải nguyên nhân gốc duy nhất**.
- Khi bỏ `tanh` mà giữ nguyên head và recipe train, mô hình không thoát khỏi bottleneck đó mà trở nên **mất ổn định**.

Nếu giữ nguyên phân phối dữ liệu hiện tại, target `< 0.05` vẫn có thể đạt, nhưng gần như chắc chắn **không bền vững với head hiện tại và recipe train hiện tại**.

## 2. Facts That Are Actually Confirmed

### 2.1 Training histories

Từ [history (1).csv](C:/Users/USER/Downloads/history%20(1).csv):

- `tanh` head train được 42 epoch.
- `val_loss` giảm từ `0.10658739` xuống `0.07848357`.
- LR giảm từ `1e-4` xuống `1.5625e-6`.
- 15 epoch cuối gần như chỉ mài rất chậm.

Từ [history.json](C:/Users/USER/Downloads/history.json):

- run `linear` chỉ có 12 epoch.
- best `val_loss` là ngay epoch `0`: `0.1121699453`.
- sau đó `val_loss` xấu dần, có lúc lên `0.1566415920`.
- `train_loss` giảm đến khoảng epoch `4`, rồi tăng trở lại đến `0.1320509076`.

Điều này không giống một mô hình “thiếu capacity” đơn thuần. Nó giống một hệ **optimization bị kéo vào quỹ đạo sai**.

### 2.2 Bucket analysis

Bucketed validation analysis của bạn cho:

- `n_used = 200704`
- `mse_total = 0.079060`

Ba vùng chính:

| Region | Data share | Avg MSE | Loss share |
|---|---:|---:|---:|
| `|y| <= 0.1` | `43.1%` | `0.0302` | `16.5%` |
| `0.1 < |y| <= 0.5` | `30.1%` | `0.0510` | `19.4%` |
| `|y| > 0.5` | `26.7%` | `0.1894` | `64.1%` |

Tức là vùng decisive chỉ chiếm `26.7%` data nhưng gánh `64.1%` loss.

Các extreme buckets cho thấy co biên độ rất đều:

- `+0.964 -> +0.528`
- `+0.844 -> +0.463`
- `+0.748 -> +0.395`
- `-0.844 -> -0.486`
- `-0.964 -> -0.531`

Tỉ lệ `|pred| / |target|` chỉ khoảng `0.53 - 0.58`, gần như hằng theo toàn decisive region. Đây là dấu hiệu rất mạnh của **low-gain readout**, không phải chỉ là saturation ở cực trị cao nhất.

### 2.3 Parameter allocation

Trên kiến trúc hiện tại trong repo:

- total params: `7,917,825`
- stem: `41,984`
- backbone blocks: `7,733,120`
- head: `142,721`

Tức là:

- backbone chiếm `97.67%` tham số
- head chỉ chiếm `1.80%`

Nếu score bị nghẹt ở head, việc tăng backbone tiếp tục gần như tăng tham số sai chỗ.

## 3. Mathematical Diagnosis Of The Current Head

Head hiện tại trong [model/architecture/head.py](../../model/architecture/head.py) có dạng:

```text
p(x)      = [GAP(x), GMP(x)]
g(x)      = sigmoid(A p(x))             in (0,1)^C
x_gated   = x ⊙ g(x)
s(x)      = spatial_conv(x_gated)
y_hat     = w^T vec(s(x)) + b
output    = tanh(y_hat) or y_hat
```

Trong đó nhánh global chỉ sinh ra một channel gate `g(x)`.

### 3.1 The key inequality

Vì từng phần tử của `g(x)` nằm trong `(0,1)`, với mọi `p-norm` ta có:

```text
|x ⊙ g(x)| <= |x|    elementwise
=> ||x ⊙ g(x)||_p <= ||x||_p
```

Đây là mệnh đề toán học rất quan trọng:

- nhánh global **chỉ có thể suppress**
- nó **không thể amplify**
- nó cũng **không có đường additive trực tiếp** đi từ pooled global signal ra scalar score

Nói cách khác, pooled context chỉ có thể nói:

- “kênh nào nên giảm”

nhưng không thể nói trực tiếp:

- “vị trí này cần decisively thắng hơn”
- “hãy cộng thêm một scalar toàn cục vào value”

### 3.2 Why this creates a low-gain attractor

Ở mức tuyến tính hóa cục bộ, head hành xử như một readout có gain hiệu dụng thấp:

```text
y_hat ≈ b + J_x x + J_g g(x)
```

Nhưng `g(x)` không đi thẳng ra score; nó chỉ đi qua phép nhân `x ⊙ g(x)`. Vì `g(x)` bị chặn trong `(0,1)`, nhánh global không thể tự nó tạo ra một score lớn hơn bằng cách khuếch đại magnitude của tensor hiện có.

Hệ quả:

- để tạo ra decisive outputs, mô hình phải ép **backbone** hoặc **fc_out** tự tăng norm đủ lớn
- đó lại là hướng bị cản bởi regularization và dynamics train

Với recipe hiện tại trong notebook:

- `Adam(..., weight_decay=1e-2)`
- `ReduceLROnPlateau`
- `AUTO_RESUME=True`
- `RESUME_STRICT_CONFIG=False`

thì nghiệm “giữ gain thấp, dự đoán gần 0 hơn” là nghiệm **rất hấp dẫn về mặt tối ưu hóa**.

Đây là điểm gốc rễ hơn việc chỉ nói “tanh bão hòa”.

### 3.3 Why tanh is not the sole root cause

Nếu output là `tanh(z)`, với MSE:

```text
dL/dz = 2 (tanh(z) - y) (1 - tanh(z)^2)
```

Tại các decisive buckets mà bạn đưa ra, model đang dự đoán khoảng `0.395 - 0.528`, nên hệ số:

```text
1 - y_hat^2 ≈ 0.72 - 0.84
```

Nghĩa là tại đúng trạng thái plateau đang quan sát, `tanh` mới chỉ làm gradient yếu đi vừa phải, chưa đủ để một mình giải thích chuyện:

- `0.964 -> 0.528`
- `0.844 -> 0.463`

Nếu `tanh` là nguyên nhân chính duy nhất, ta kỳ vọng shrink sẽ tăng rất mạnh khi target tiến sát `1`. Nhưng bucket analysis cho thấy shrink gần như đồng đều trên cả decisive region. Điều đó khớp với **low-gain solution**, không khớp với “pure tanh saturation only”.

### 3.4 Why the linear run still failed

Run `linear` bỏ `tanh`, nhưng giữ nguyên tinh thần head và recipe train. Khi đó:

- bottleneck suppress-only của head vẫn còn
- không có đường global additive ra score
- muốn tăng decisiveness, mô hình buộc phải tăng norm của backbone/readout

Trên recipe hiện tại, hướng đó dễ mất ổn định hơn:

- decay mạnh kéo tham số về nhỏ
- scheduler hạ LR sớm
- kết quả là run `linear` không tạo được gain tốt, rồi train loss đảo chiều đi lên

Vì vậy:

- `linear` fail **không chứng minh backbone yếu**
- nó chứng minh rằng **bỏ tanh mà không sửa head/readout geometry là chưa đủ**

## 4. Why Bigger Backbone Does Not Solve The Right Problem

Backbone hiện tại chiếm gần toàn bộ tham số của mô hình, nhưng scalar value vẫn phải chui qua một head rất nhỏ và rất bảo thủ.

Điều này tạo ra mismatch:

- biểu diễn bên trong có thể đủ giàu
- nhưng cơ chế đọc ra `1` scalar lại quá nghèo

Đó là lý do “mạng lớn” vẫn có thể plateau ở một mức không tốt.

Nói chính xác hơn:

- **expressivity của toàn class mô hình chưa bị bác bỏ**
- nhưng **optimization geometry của head hiện tại làm target `< 0.05` trở nên khó và không ổn định**

Tôi không khẳng định có một chứng minh tuyệt đối rằng kiến trúc hiện tại là bất khả thi. Nhưng với dữ liệu thực nghiệm hiện có, tôi đánh giá:

- **backbone không phải blocker**
- **head hiện tại là blocker kiến trúc thật sự**

## 5. What Is Primary, What Is Secondary

### Primary bottleneck

**Current `ContextGatedHead`**

- global branch chỉ suppress
- không thể amplify
- không có direct global signed score path
- dễ tạo ra nghiệm under-confident, low-gain

### Co-primary optimization issue

**Train recipe trong notebook**

- `weight_decay = 1e-2` trên toàn bộ params là rất mạnh cho value regression
- `ReduceLROnPlateau` giảm LR sớm trong khi decisive region còn học chưa xong
- run `linear` không phải một ablation sạch nếu reuse cùng setup resume

### Secondary issues

**Tanh**

- có hại
- là amplifier của low-gain solution
- nhưng không phải nguyên nhân gốc duy nhất

**Positive/negative asymmetry**

- có thật
- nếu sửa hết asymmetry bằng cách mirror MSE âm sang dương thì tổng MSE mới chỉ giảm khoảng `0.0069`
- đây là vấn đề phụ, chưa phải lý do chính khiến không xuống `< 0.05`

**P4/P5 trong `blocks.py`**

- hiện tại local repo đã để `CoordinateAttention(..., reduction=8)`
- và đã đổi sang attend-before-fuse
- chúng có thể có ích, nhưng không còn là nghi phạm số 1

## 6. Can This Architecture Reach `< 0.05`?

Câu trả lời chính xác nhất là:

- **backbone này có đủ sức biểu diễn**
- nhưng **toàn kiến trúc với head hiện tại, dưới recipe train hiện tại, khó đạt `< 0.05` một cách đáng tin cậy**

Lý do định lượng:

- hiện tại decisive region có average MSE `0.1894`
- nếu center và mid giữ nguyên, để tổng MSE từ `0.07906` xuống `< 0.05`, decisive region phải giảm về khoảng `0.0808`
- tức cần giảm thêm khoảng `57%` ở vùng decisive

Bucket analysis hiện tại cho thấy decisive region không chỉ noisy mà đang bị **global amplitude collapse**. Đây không phải kiểu lỗi thường được giải bằng cách “thêm block”.

## 7. Proposed Head Redesign

Đây là phương án được vẽ trong [docs/dgrn_value_head_redesign.drawio](../../docs/dgrn_value_head_redesign.drawio).

### 7.1 Proposed equations

Thay head hiện tại bằng:

```text
p(x)       = [GAP(x), GMP(x)]
m(x)       = 1 + α tanh(B p(x))         with 0 < α <= 1
x_mod      = x ⊙ m(x)
s_spatial  = w_s^T vec(psi(x_mod))
s_global   = h(p(x))
y_hat      = s_spatial + s_global
```

Nếu vẫn muốn output bị chặn trong `[-1,1]`, chỉ áp `tanh` ở inference hoặc train trong logit-space.

### 7.2 Why this is mathematically better

**Residual gain instead of suppress-only gate**

Với `m(x) = 1 + α tanh(.)`, ta có:

```text
m(x) in [1-α, 1+α]
=> ||x ⊙ m(x)||_p <= (1+α) ||x||_p
```

Khác với head cũ:

- nhánh global giờ có thể **amplify có kiểm soát**
- vẫn bị chặn trên, nên không gây nổ vô hạn

**Direct global score path**

`h(p(x))` cho phép pooled global information đi **thẳng** ra scalar:

- material imbalance
- king safety
- phase-like global state

không còn bị bắt buộc đi vòng qua phép nhân với tensor không gian.

**Decouple feature selection from score magnitude**

Head cũ trộn hai việc vào cùng một gate:

- chọn feature nào quan trọng
- và vô tình áp đặt gain bảo thủ

Head mới tách chúng ra:

- `x ⊙ m(x)` xử lý modulation của feature tensor
- `h(p(x))` xử lý scalar global bias/gain trực tiếp

Đây là sửa đúng vào hình học của bài toán value head.

## 8. Architecture V2

Tôi đã lưu một bản mới ở package `model/architecture_v2` với mục tiêu sửa đúng bottleneck đã xác định, nhưng không động vào backbone:

- `model/architecture_v2/blocks.py`
- `model/architecture_v2/head.py`
- `model/architecture_v2/model.py`

### 8.1 What architecture_v2 changes

Backbone vẫn là DGRN như cũ:

- stem 3x3 + BN + Mish
- 20 `DFGBlock`
- `CoordinateAttention(reduction=8)`
- attend-before-fuse

Thay đổi nằm ở head:

```text
p(x)       = [GAP(x), GMP(x)]
m(x)       = 1 + α tanh(B p(x))
x_mod      = x ⊙ m(x)
s_spatial  = w_s^T vec(psi(x_mod))
s_global   = h(p(x))
y_hat      = s_spatial + s_global
```

Điểm kỹ thuật chính trong `architecture_v2`:

- gain branch khởi tạo đúng identity bằng cách zero-init lớp cuối của `gain_mlp`
- head có `forward_logits()` riêng
- `forward()` vẫn hỗ trợ cả `tanh` và `linear`

### 8.2 Why this version is safer

`architecture_v2` sửa đúng ba lỗi lớn của head cũ:

- pooled context có thể **amplify có kiểm soát**
- pooled context có **đường additive trực tiếp ra scalar**
- train có thể dùng **raw logits** thay vì bị khóa vào output đã squash

Nói ngắn gọn:

- head cũ chủ yếu là một **selector**
- head mới vừa là **selector**, vừa là **gain controller**, vừa là **global scorer**

## 9. Tanh Vs Linear In The New Head

Đây là điểm bạn hỏi trực tiếp, và tôi sẽ chốt rõ.

### 9.1 Rủi ro của `tanh`

Ưu điểm:

- output nằm đúng miền `[-1, 1]`
- khớp trực tiếp với nhãn hiện tại của bạn
- ổn định hơn trong early training
- khó bị runaway ở vùng decisive

Rủi ro:

- nếu train trực tiếp với `loss(y_hat, y)` sau khi đã `tanh`, gradient ở extremes bị nén
- càng gần `±1`, học càng chậm hơn nếu không đổi loss

### 9.2 Rủi ro của `linear`

Ưu điểm:

- không có hệ số co gradient ở output layer
- về lý thuyết dễ fit tails hơn nếu train dynamics sạch

Rủi ro:

- output không bị chặn, trong khi nhãn lại bị chặn
- dễ overshoot hoặc drift nếu recipe train chưa ổn
- nhạy hơn với `weight_decay`, scheduler và calibration
- run `linear` bạn đã có là bằng chứng nó không tự cứu được mô hình khi head/readout vẫn sai

### 9.3 Quan điểm của tôi

Nếu buộc chọn trực tiếp giữa:

- `tanh` head train trực tiếp trên nhãn đã squash
- `linear` head train trực tiếp trên chính nhãn đó

thì với repo và data của bạn, tôi **nghiêng về `tanh`**.

Lý do:

- nhãn của bạn đã được encode về `[-1, 1]`
- bạn muốn value ổn định, không runaway
- run `linear` thực tế đã cho thấy recipe hiện tại không phù hợp

Nhưng lựa chọn tốt nhất về mặt kỹ thuật không phải “tanh thuần” hay “linear thuần”.

### 9.4 Lựa chọn tôi khuyên dùng thật sự

**Internal logits + tanh view**

Tức là:

- head sinh ra `logits`
- train bằng `forward_logits()` nếu có thể
- chỉ áp `tanh` ở output hoặc ở lúc inference/reporting

Nếu giữ đúng triết lý nhãn hiện tại của bạn, công thức train tốt nhất là:

```text
z_target = atanh(clamp(y, -1+eps, 1-eps))
loss = MSE(z_pred, z_target)
y_pred = tanh(z_pred)
```

Đây là phương án dung hòa được cả hai điều bạn muốn:

- giữ semantic của nhãn `tanh`
- tránh bottleneck gradient của việc train trực tiếp trên output đã squash

### 9.5 Practical recommendation

Nếu bạn muốn đổi ít nhất có thể:

- dùng `architecture_v2`
- giữ `output_mode="tanh"`
- nhưng thêm lựa chọn train bằng `forward_logits()`

Nếu bạn chưa muốn đổi loss ngay:

- vẫn nên ưu tiên `tanh` hơn `linear` trong `architecture_v2`

Nếu bạn sẵn sàng sửa loss:

- `architecture_v2 + forward_logits + logit-space target` là lựa chọn tôi đánh giá tốt nhất

## 10. Practical Solutions

### Solution A: Replace the head first

Giữ nguyên backbone. Sửa head theo `architecture_v2`:

- residual gain branch
- spatial score branch
- direct global scalar branch

Đây là thay đổi có tỷ lệ lợi ích/chi phí tốt nhất.

### Solution B: If you keep bounded outputs, train in logit-space

Thay vì train trực tiếp `tanh(z)` với MSE trên `y`, dùng:

```text
z_target = atanh(clamp(y, -1+eps, 1-eps))
loss = MSE(z, z_target)
y_pred = tanh(z)   only for inference / reporting
```

Lợi ích:

- bỏ hệ số `(1 - y_hat^2)` khỏi gradient train
- vẫn giữ được output bị chặn khi suy luận
- khớp với trực giác của bạn về việc nhãn đã encode bằng `tanh`

### Solution C: Fix the train recipe together with the head

Nếu không sửa recipe, head mới vẫn có thể bị kéo theo hướng bảo thủ.

Các thay đổi hợp lý:

- dùng `AdamW` thay cho `Adam`
- giảm mạnh weight decay tổng thể
- không decay `BatchNorm`, bias, và head bias
- bỏ `ReduceLROnPlateau`, chuyển sang warmup + cosine decay
- chạy ablation sạch, không resume lẫn giữa các kiến trúc

### Solution D: Keep the current data distribution

Bạn muốn giữ phân phối center-heavy, và với mục tiêu engine value ổn định thì điều đó hợp lý.

Từ bằng chứng hiện có:

- **không cần đổi data distribution để chẩn đoán đúng vấn đề**
- vấn đề chính nằm ở **head/readout geometry và train dynamics**

## 11. Confidence

Các kết luận tôi tự tin nhất:

| Claim | Confidence |
|---|---:|
| Backbone không phải bottleneck chính | `> 90%` |
| Head hiện tại là bottleneck kiến trúc chính | `> 90%` |
| `tanh` không phải root cause duy nhất | `> 90%` |
| Recipe train hiện tại đang củng cố nghiệm low-gain | `85 - 90%` |
| Chỉ tăng backbone sẽ không giải đúng chỗ | `> 90%` |
| Head mới theo hướng residual gain + additive global score là hướng sửa đúng nhất | `80 - 85%` |

## 12. Final Takeaway

Vấn đề thật sự không phải là:

- data của bạn center-heavy
- hay backbone quá yếu
- hay chỉ đơn giản là `tanh`

Vấn đề thật sự là:

- **scalar value đang bị đọc ra qua một head có hình học quá bảo thủ**
- global context chỉ biết suppress chứ không biết amplify hoặc cộng trực tiếp vào score
- optimization hiện tại khiến nghiệm low-gain đó trở thành nghiệm rất hấp dẫn

Muốn xuống `< 0.05`, hướng có cơ sở toán học tốt nhất là:

1. **đổi head trước**
2. **đổi train recipe đi kèm**
3. **không cần đổi phân phối dữ liệu**

Nếu chỉ bỏ `tanh` hoặc chỉ tăng số block, khả năng cao bạn sẽ vẫn ở sai hướng.
