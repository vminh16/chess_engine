# Đặc Tả Thiết Kế Loss Hàm Đánh Giá

**Tệp:** `docs/disigner_loss_func.md`  
**Mục tiêu:** xác định một loss function phù hợp cho value network hiện tại, giải quyết đồng thời hai vấn đề đã quan sát được:

1. `y-space loss` học tails yếu, dễ under-confident
2. `z-space loss` học tails mạnh nhưng gây calibration drift ở validation/test

Bản đặc tả này tổng hợp nền tảng lý thuyết, lập luận toán học và kết luận thực dụng để quyết định xem hướng loss mới đã đủ cơ sở để bắt đầu thực nghiệm hay chưa.

---

## 1. Bối cảnh bài toán

Mạng hiện tại sinh ra một **logit** `z = f_theta(x)` và output triển khai là:

```text
p = tanh(z)
```

Trong pipeline hiện tại:

- dữ liệu target được lưu ở miền `y in [-1, 1]`
- validation và checkpoint selection dùng MSE trong `y-space`
- nhưng train objective hiện tại lại dùng MSE trong `z-space` với target `atanh(y)`

Cụ thể:

```text
L_y = (tanh(z) - y)^2
L_z = (z - atanh(y_clamped))^2
```

Ta cần một loss mới sao cho:

- vẫn học được decisive tails tốt
- nhưng không gây drift ở `y-space`
- phù hợp với cách đánh giá và triển khai thực tế của engine

---

## 2. Ký hiệu

Gọi:

- `x`: input board tensor
- `theta`: tham số mô hình
- `z = f_theta(x)`: logit model dự đoán
- `p = tanh(z)`: output bounded thật sự dùng trong engine
- `y in (-1, 1)`: target hiện tại
- `a = atanh(y)`: target mapped sang logit-space
- `r = z - a`: sai số trong logit-space

Ta sẽ phân tích ba họ loss:

```text
L_y      = (p - y)^2
L_z      = (z - a)^2
L_hybrid = λ L_y + μ L_z
```

và sau đó đi tới dạng loss đề xuất cuối cùng.

---

## 3. Phân tích toán học của hai loss cơ bản

## 3.1. Loss trong `y-space`

Loss trực tiếp trên output bounded:

```text
L_y = (tanh(z) - y)^2
```

Gradient theo `z` là:

```text
∂L_y/∂z = 2 (tanh(z) - y) (1 - tanh(z)^2)
```

Hệ số `(1 - tanh(z)^2)` chính là đạo hàm của `tanh`.

### Ý nghĩa

- Khi `|z|` lớn, `tanh(z)` tiến sát `±1`
- Lúc đó `1 - tanh(z)^2` rất nhỏ
- Gradient bị co lại mạnh ở các mẫu decisive

Đây là nguyên nhân cơ học khiến pure `y-space` dễ học yếu ở tails.

---

## 3.2. Loss trong `z-space`

Loss trên logit với target đảo ngược bởi `atanh`:

```text
L_z = (z - a)^2
```

trong đó:

```text
a = atanh(y)
```

Gradient theo `z` là:

```text
∂L_z/∂z = 2 (z - a) = 2r
```

### Ý nghĩa

- Không có hệ số co do `tanh`
- Gradient mạnh hơn ở tails
- Dễ học decisive region hơn pure `y-space`

Nhưng `atanh(y)` có đạo hàm:

```text
d/dy atanh(y) = 1 / (1 - y^2)
```

Nghĩa là nhiễu nhỏ trong `y` sẽ bị phóng đại ở vùng gần biên.

Nếu `delta y` là nhiễu nhỏ thì xấp xỉ bậc một cho ta:

```text
delta a ≈ delta y / (1 - y^2)
```

Ví dụ:

- `y = 0.8`  -> hệ số khuếch đại khoảng `2.78x`
- `y = 0.9`  -> khoảng `5.26x`
- `y = 0.96` -> khoảng `12.82x`

Đây là gốc rễ lý thuyết của calibration drift nếu mô hình tiếp tục tối ưu mạnh trong `z-space`.

---

## 4. Vì sao `L_y` và `L_z` không tương đương khi train mạng thật

Đây là điểm quan trọng nhất.

Nhiều người nhìn thấy `p = tanh(z)` và `a = atanh(y)` rồi nghĩ rằng:

```text
MSE(tanh(z), y)  <=>  MSE(z, atanh(y))
```

Nhưng điều này **không đúng** khi train một mạng có tham số dùng chung cho nhiều mẫu.

## 4.1. Chúng giống nhau ở mức nghiệm từng mẫu

Với một mẫu đơn lẻ, cả hai loss có cùng nghiệm tối ưu:

```text
z* = a = atanh(y)
```

vì khi đó:

```text
tanh(z*) = y
```

## 4.2. Nhưng chúng khác nhau ở hình học gradient

Khai triển `tanh(z)` quanh nghiệm `z = a`:

```text
tanh(z) ≈ y + (1 - y^2)(z - a)
```

Suy ra:

```text
L_y = (tanh(z) - y)^2
    ≈ (1 - y^2)^2 (z - a)^2
    = (1 - y^2)^2 r^2
```

Tức là gần nghiệm, `L_y` tương đương với **logit MSE có trọng số**:

```text
L_y ≈ w(y) r^2
w(y) = (1 - y^2)^2
```

Trong khi đó:

```text
L_z = r^2
```

không có trọng số phụ thuộc `y`.

### Hệ quả định lượng

Ví dụ với vài giá trị `y`:

```text
y = 0.0   -> w(y) = 1.0000
y = 0.5   -> w(y) = 0.5625
y = 0.8   -> w(y) = 0.1296
y = 0.9   -> w(y) = 0.0361
y = 0.96  -> w(y) ≈ 0.0061
```

Tức là ở `y = 0.96`, pure `y-space` xem một logit error cùng độ lớn nhẹ hơn khoảng **160 lần** so với quanh `y = 0`.

### Kết luận của mục này

- `L_y` và `L_z` có cùng nghiệm theo từng mẫu
- nhưng không cùng điều kiện tối ưu, không cùng trọng số gradient, không cùng hình học Hessian
- vì vậy chúng **không tương đương** khi train một network thực

---

## 5. Hai failure mode chính

## 5.1. Failure mode của pure `y-space`

Nếu dùng:

```text
L_y = (tanh(z) - y)^2
```

thì các mẫu decisive có gradient rất yếu ở vùng muộn của training.

### Hệ quả

- tails học chậm
- mô hình dễ under-confident
- với dữ liệu center-heavy, mô hình càng bị kéo về vùng dự đoán bảo thủ gần `0`

### Diễn giải chính xác

Không phải “loss train không thể giảm”, mà là:

- bài toán có condition number xấu hơn ở tails
- mô hình vẫn giảm loss được, nhưng rất khó mở dynamic range đúng mức ở decisive region

## 5.2. Failure mode của pure `z-space`

Nếu dùng:

```text
L_z = (z - atanh(y))^2
```

thì tails được học mạnh hơn, nhưng nhiễu và sai số tại vùng gần `|y|=1` bị phóng đại.

### Hệ quả

- logit tiếp tục tăng độ quyết liệt ở late stage
- `z-space loss` giảm
- nhưng `y-space calibration` có thể xấu đi

Đây chính là pattern đã được quan sát thực tế trong checkpoint thật:

- `ckpt_latest` tốt hơn ở `z_mse`
- nhưng tệ hơn ở `y_mse`

---

## 6. Hybrid loss tĩnh có đủ không?

Loss lai đơn giản:

```text
L_hybrid = λ L_y + μ L_z
```

Gradient theo `z`:

```text
∂L_hybrid/∂z = 2λ (tanh(z) - y)(1 - tanh(z)^2) + 2μ (z - a)
```

Gần nghiệm tối ưu:

```text
∂L_hybrid/∂z ≈ 2 [ λ(1 - y^2)^2 + μ ] r
```

### Ý nghĩa

Nếu `μ` là hằng số dương và không rất nhỏ, thì tails vẫn bị thành phần `z-space` kéo mạnh đến cuối training.

Nói cách khác:

- hybrid tĩnh có thể giảm bớt vấn đề
- nhưng chưa giải quyết triệt để calibration drift ở giai đoạn cuối

### Kết luận

**Hybrid loss tĩnh là chưa đủ tốt**.

Ta cần thêm hai cơ chế:

1. trọng số theo `y`
2. curriculum theo thời gian train

---

## 7. Thiết kế loss phù hợp hơn về mặt lý thuyết

Từ các phân tích trên, loss hợp lý hơn nên có dạng:

```text
L_t = λ_t L_y + (1 - λ_t) L_z^w
```

trong đó:

```text
L_y   = (tanh(z) - y)^2
L_z^w = w_beta(y) * ρ_delta(z - atanh(y))
```

với:

- `λ_t`: hệ số curriculum phụ thuộc thời gian train
- `w_beta(y) = (1 - y^2)^beta`
- `ρ_delta`: Huber loss trên residual `z - atanh(y)`

Ta sẽ phân tích từng thành phần.

---

## 8. Vì sao cần trọng số `w_beta(y)`

Ta đã biết:

- pure `z-space` tương đương với `beta = 0`
- pure `y-space` gần nghiệm tương đương với `beta = 2`

Vậy dạng:

```text
w_beta(y) = (1 - y^2)^beta
```

là một họ loss liên tục nội suy giữa hai thế giới.

### Ý nghĩa thống kê

- `beta = 0`
  - giả định nhiễu gần homoscedastic trong latent `z`
- `beta = 2`
  - giả định nhiễu gần homoscedastic trong output `y`
- `beta` ở giữa
  - là một thỏa hiệp giữa hai giả định

### Vì sao `beta = 1` là điểm bắt đầu hợp lý nhất

Nếu chọn `beta = 1`, ta chỉ giữ lại **một nửa** mức phóng đại của `atanh`.

Ví dụ tại `y = 0.96`:

```text
pure y-space   -> weight ≈ 0.0061
beta = 1       -> weight = 0.0784
pure z-space   -> weight = 1.0
```

Nghĩa là `beta = 1`:

- mạnh hơn pure `y-space` khoảng `12.8x`
- nhưng yếu hơn pure `z-space` khoảng `12.8x`

Đây là một điểm cân bằng rất tự nhiên về mặt toán học.

---

## 9. Vì sao nhánh `z-space` nên dùng Huber thay vì MSE thuần

Nếu ta dùng trực tiếp:

```text
(z - atanh(y))^2
```

thì một số mẫu tail nhiễu hoặc outlier sẽ tạo gradient rất mạnh.

Huber loss trên residual `r = z - a`:

```text
ρ_delta(r) = 0.5 r^2                    nếu |r| <= delta
ρ_delta(r) = delta (|r| - 0.5 delta)    nếu |r| > delta
```

Gradient của Huber:

```text
ρ'_delta(r) = r                         nếu |r| <= delta
ρ'_delta(r) = delta * sign(r)           nếu |r| > delta
```

### Ý nghĩa

- gần nghiệm, nó vẫn giống MSE
- xa nghiệm, gradient bị chặn tuyến tính
- giảm nguy cơ vài sample decisive nhưng nhiễu chi phối hướng cập nhật

### Kết luận

Nếu giữ một thành phần `z-space`, **Huber hợp lý hơn MSE thuần**.

---

## 10. Vì sao curriculum là bắt buộc, không phải tùy chọn

Nếu giữ thành phần `z-space` mạnh suốt quá trình train, calibration drift vẫn có thể quay lại ở cuối run.

Ta cần một hệ số `λ_t` tăng dần theo thời gian:

```text
L_t = λ_t (tanh(z) - y)^2 + (1 - λ_t) (1 - y^2)^beta ρ_delta(z - atanh(y))
```

với:

- đầu run: `λ_t` nhỏ hơn
- cuối run: `λ_t` tiến gần `1`

## 10.1. Lý do ở đầu run

Đầu training cần:

- học được dynamic range
- tránh gradient quá yếu ở tails
- nhanh thoát vùng dự đoán quá bảo thủ

Nên nhánh `z-space` phải còn đủ lực.

## 10.2. Lý do ở cuối run

Cuối training mục tiêu thực sự là:

- calibration tốt trong `y-space`
- metric validation tốt
- checkpoint tốt theo đúng đại lượng triển khai

Lúc này `y-space` phải chiếm ưu thế.

### Kết luận

Không nên giữ trọng số hai nhánh cố định từ đầu đến cuối.  
**Curriculum là một phần bản chất của thiết kế loss này.**

---

## 11. Loss đề xuất cuối cùng

Loss được đề xuất cho repo hiện tại là:

```text
L_t = λ_t (tanh(z) - y)^2 + (1 - λ_t) (1 - y^2)^beta ρ_delta(z - atanh(y_clamped))
```

với:

- `beta ≈ 1` là điểm khởi đầu lý thuyết hợp lý nhất
- `ρ_delta` là Huber loss
- `λ_t` tăng dần theo epoch hoặc global step

## 11.1. Gradient gần nghiệm

Gần nghiệm tối ưu `z ≈ atanh(y)`, do Huber gần giống bình phương khi residual nhỏ, ta có:

```text
∂L_t/∂z ≈ 2 [ λ_t (1 - y^2)^2 + (1 - λ_t) (1 - y^2)^beta ] r
```

Nếu `beta = 1`:

```text
∂L_t/∂z ≈ 2 [ λ_t (1 - y^2)^2 + (1 - λ_t)(1 - y^2) ] r
```

Hệ số hiệu dụng này nằm đúng giữa hai cực đoan:

- pure `y-space`: `(1 - y^2)^2`
- pure `z-space`: `1`

Nghĩa là loss mới:

- vẫn giữ gradient đủ cho tails
- nhưng không đối xử tails “quá mạnh” như pure `z-space`

---

## 12. Ưu và nhược điểm của loss đề xuất

## 12.1. Ưu điểm

- giải đúng hai failure mode đã quan sát
- có nền tảng toán học rõ ràng
- không quay lại pure `y-space`
- không giữ nhược điểm drift của pure `z-space`
- phù hợp với mục tiêu cuối là metric trong `y-space`

## 12.2. Nhược điểm

- tăng số hyperparameter:
  - `beta`
  - `delta` của Huber
  - lịch `λ_t`
- tuning khó hơn MSE thuần
- cần log thêm metric để đánh giá đúng hiệu quả:
  - `y_mse`
  - `z_mse`
  - bucketed MSE
  - signed bias
  - mean absolute prediction

---

## 13. Loss này đã đủ nền tảng lý thuyết để đem đi thực nghiệm chưa?

### Câu trả lời ngắn

**Có, đã đủ cơ sở lý thuyết để bắt đầu thực nghiệm có kiểm soát.**

### Vì sao có thể bắt đầu thử ngay

1. Nó không phải ý tưởng tùy hứng.
   - Nó được suy ra trực tiếp từ việc so sánh hình học của `L_y` và `L_z`.

2. Nó giải đúng failure mode đã được quan sát thực nghiệm.
   - `y-space` yếu ở tails
   - `z-space` gây drift ở val/test

3. Nó có tham số điều chỉnh mang ý nghĩa rõ ràng.
   - `beta`: mức độ nội suy giữa `y-space` và `z-space`
   - `delta`: mức robust với outlier tail
   - `λ_t`: mức ưu tiên calibration theo thời gian

4. Nó không mâu thuẫn với pipeline hiện tại.
   - vẫn dùng target `y`
   - vẫn giữ output `tanh`
   - vẫn có thể checkpoint theo `y-space`

### Điều cần nhấn mạnh

“Đủ nền tảng lý thuyết để thử” **không có nghĩa là đã được chứng minh chắc chắn sẽ thắng**.

Lý thuyết hiện tại chứng minh rằng loss này:

- hợp lý hơn hai cực đoan cũ
- phù hợp hơn với failure mode hiện tại
- đáng để thử trước các hướng khác

Nhưng chiến thắng cuối cùng vẫn phải được xác nhận bằng thực nghiệm.

---

## 14. Điều kiện để một thực nghiệm đầu tiên được coi là tốt

Một run thử loss mới nên được xem là thành công nếu đồng thời có các dấu hiệu sau:

1. `val_loss` theo `y-space` không drift sớm như pure `z-space`
2. bucketed MSE ở decisive region không bị yếu đi quá mạnh như pure `y-space`
3. `mean_abs_pred` không tăng quá nhanh ở late stage
4. signed bias không drift rõ về một phía
5. `ckpt_latest` không còn pattern:
   - `z_mse` tốt hơn nhưng `y_mse` tệ hơn `ckpt_best`

Nếu đạt các điều kiện này, ta mới có cơ sở nói rằng loss mới thật sự giải được bài toán.

---

## 15. Kết luận cuối cùng

Loss hiện tại của pipeline đang mắc đúng mâu thuẫn sau:

- muốn học tốt tails -> đi về `z-space`
- muốn calibrated tốt theo metric triển khai -> phải quay lại `y-space`

Pure `y-space` và pure `z-space` đều là hai cực đoan không phù hợp hoàn toàn với repo này.

Thiết kế loss hợp lý nhất về mặt lý thuyết hiện tại là:

```text
L_t = λ_t (tanh(z) - y)^2 + (1 - λ_t) (1 - y^2)^beta ρ_delta(z - atanh(y_clamped))
```

với:

- `beta ≈ 1`
- `ρ_delta` là Huber
- `λ_t` tăng dần theo thời gian train

Đây là thiết kế có nền tảng lý thuyết đủ mạnh để **bắt đầu thực nghiệm một cách nghiêm túc**, và là ứng viên hợp lý nhất để kiểm tra tiếp theo trên repo hiện tại.
