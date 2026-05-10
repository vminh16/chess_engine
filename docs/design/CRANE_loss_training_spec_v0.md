# CRANE-v0 Loss Function & Training Strategy Specification

**Ngày:** 2026-05-10  
**Trạng thái:** `finalized`  
**Phạm vi:** Đặc tả loss function, chiến lược training, và chứng minh toán học cho dòng kiến trúc CRANE  
**Tiền tố:** Tài liệu này thay thế Mục 8–9 của `CRANE_architecture_spec_v0.md`  
**Phụ thuộc:** [`CRANE_architecture_spec_v0.md`](./CRANE_architecture_spec_v0.md), [`CRANE_encode_spec_v0.md`](./CRANE_encode_spec_v0.md)

---

## 1. Kết luận điều hành

Loss function của CRANE-v0 là **z-space Huber** — một loss đơn giản, đúng hình học, không cần curriculum, không cần hybrid:

$$\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} \rho_\delta\!\left(z_i - \operatorname{atanh}(\operatorname{clip}(y_i,\, -c,\, c))\right)$$

trong đó $\rho_\delta$ là Huber loss với $\delta = 1.0$, $c = 0.999$.

Chiến lược training bao gồm 3 kỹ thuật bổ trợ:

1. **Gradient noise injection** — thoát anti-aligned gradient trap (giảm Failure B)
2. **Cosine annealing với warmup** — điều chỉnh gradient magnitude ổn định
3. **Mixed-band batching** — giảm phương sai gradient estimator

Thiết kế này được suy ra từ phân tích root cause toán học của 2 failure mode đã chứng minh thực nghiệm, không dựa trên trực giác hay "chữa cháy".

---

## 2. Phân tích root cause: vì sao loss cũ thất bại

### 2.1. Failure A: nén biên độ dải giữa

**Mô hình output:** $p = \tanh(z)$, target $y \in (-1, 1)$.

**Loss cũ (y-MSE):** $\mathcal{L}_y = (\tanh(z) - y)^2$

**Định lý 1 (Curvature Attenuation).** *Gần nghiệm $z^* = \operatorname{atanh}(y)$, y-MSE tương đương với z-space loss với trọng số $(1 - y^2)^2$:*

$$\mathcal{L}_y \approx (1 - y^2)^2 \cdot (z - z^*)^2$$

**Chứng minh.** Khai triển Taylor bậc 1 của $\tanh(z)$ quanh $z^*$:

$$\tanh(z) = \tanh(z^*) + \tanh'(z^*)(z - z^*) + O((z-z^*)^2)$$

Vì $\tanh(z^*) = y$ và $\tanh'(z^*) = 1 - y^2$:

$$\tanh(z) \approx y + (1 - y^2)(z - z^*)$$

Thay vào $\mathcal{L}_y$:

$$(\tanh(z) - y)^2 \approx (1 - y^2)^2 (z - z^*)^2 \qquad \blacksquare$$

**Hệ quả định lượng.** Trọng số hiệu dụng $(1-y^2)^2$ triệt tiêu gradient ở mid-band và tail:

| $y$ | $(1-y^2)^2$ | Gradient còn lại |
|-----|-------------|-----------------|
| 0.0 | 1.0000 | 100% |
| 0.3 | 0.8281 | 83% |
| 0.5 | 0.5625 | 56% |
| 0.7 | 0.2601 | **26%** |
| 0.9 | 0.0361 | **3.6%** |

### 2.2. Tương tác với phân phối dữ liệu: Double Whammy

Histogram dataset cho thấy phân phối **chuông** (bell-shaped) với peak tại $y \approx 0$:

- Khoảng 43% samples nằm tại $|y| < 0.15$
- Khoảng 30% tại $0.15 < |y| < 0.5$
- Khoảng 27% tại $|y| > 0.5$

Gradient allocation thực tế là tích của data density và curvature weight:

$$G_{\text{band}} \propto \underbrace{P(y \in \text{band})}_{\text{data density}} \times \underbrace{(1 - y^2)^2}_{\text{curvature attenuation}}$$

**Kết quả đo được thực nghiệm (gradient mass audit):**

| Band | Data % | y-MSE gradient % | z-MSE gradient % |
|------|--------|------------------|------------------|
| Center ($|y| \leq 0.2$) | ~50% | **72%** | **~50%** |
| Mid ($0.2 < |y| \leq 0.7$) | ~30% | **18%** | **~30%** |
| Tail ($|y| > 0.7$) | ~20% | **9%** | **~20%** |

Hai yếu tố cộng hưởng: data density + curvature cùng dồn gradient về center → **double whammy**.

### 2.3. Failure B: over-confidence tại ultra-center

Hai sub-mechanism đã được chứng minh:

**B1 — Label impurity.** Raw labels $|y| \leq 0.05$ chỉ có precision 15–22% so với oracle-center. Raw-center penalty nhắm sai tập mẫu.

**B2 — Gradient interference.** Gradient center và mid-band anti-aligned:

$$\cos(\nabla_\theta \mathcal{L}_{\text{center}},\; \nabla_\theta \mathcal{L}_{\text{mid}}) = -0.64 \text{ (baseline)}$$

Mỗi tăng gradient cho mid-band (sửa A) → tăng can nhiễu lên center → B xấu hơn. Đây là lý do mọi "chữa cháy" cũ thất bại.

### 2.4. Tại sao các khắc phục cũ thất bại

| Khắc phục | Kết quả | Nguyên nhân thất bại |
|---|---|---|
| A1 (curvature compensation) | Sửa A, phá B | Tăng mid-band gradient → tăng interference lên center |
| A2 (band-balanced sampling) | Giúp nhẹ center | Không sửa gốc curvature, chỉ giảm density dominance |
| S1 (center penalty) | Không cứu center | Raw labels không đáng tin (B1) |
| OC1/OC2 (late polish) | Thất bại | B là training dynamics problem, không phải late calibration |

**Nguyên lý sâu:** Hai failure mode **anti-aligned** trong gradient space. Bất kỳ increase gradient cho mid-band nào cũng làm center xấu đi, và ngược lại. Loss function cũ trở thành Frankenstein vì cố gắng vá từng failure mode riêng lẻ.

---

## 3. Giải pháp: z-space Huber Loss

### 3.1. Định nghĩa

$$\boxed{\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B} \rho_\delta(z_i - z_i^*)}$$

trong đó:

- $z_i = f_\theta(x_i)$: logit đầu ra từ Value Head (trước tanh)
- $z_i^* = \operatorname{atanh}(\operatorname{clip}(y_i,\, -c,\, c))$: target trong logit space
- $c = 0.999$: clipping threshold, tránh $\operatorname{atanh}(\pm 1) = \pm\infty$
- $\rho_\delta$: Huber loss

### 3.2. Huber Loss

$$\rho_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{nếu } |r| \leq \delta \\ \delta\left(|r| - \frac{1}{2}\delta\right) & \text{nếu } |r| > \delta \end{cases}$$

**Đạo hàm (gradient w.r.t. $z$):**

$$\frac{\partial \rho_\delta}{\partial z} = \begin{cases} r & \text{nếu } |r| \leq \delta \\ \delta \cdot \operatorname{sign}(r) & \text{nếu } |r| > \delta \end{cases}$$

**Khuyến nghị:** $\delta = 1.0$

### 3.3. Định lý 2: z-space Huber có curvature độc lập với y

**Định lý.** *Hessian của $\rho_\delta(z - z^*)$ w.r.t. $z$ không phụ thuộc vào $y$:*

$$\frac{\partial^2 \rho_\delta}{\partial z^2} = \begin{cases} 1 & \text{nếu } |r| \leq \delta \\ 0 & \text{nếu } |r| > \delta \end{cases}$$

*So sánh với y-MSE tại $z = z^*$: $\frac{\partial^2 \mathcal{L}_y}{\partial z^2} = 2(1 - y^2)^2$ — phụ thuộc mạnh vào $y$.*

**Chứng minh.** Residual $r = z - z^*$ không chứa $y$ ngoài định nghĩa của $z^*$, nhưng $z^*$ là hằng số w.r.t. $z$. Do đó:

$$\frac{\partial r}{\partial z} = 1, \quad \frac{\partial^2 r}{\partial z^2} = 0$$

Với $|r| \leq \delta$: $\rho_\delta = \frac{1}{2}r^2$, suy ra $\frac{\partial^2 \rho_\delta}{\partial z^2} = \frac{\partial^2}{\partial z^2}\frac{1}{2}(z-z^*)^2 = 1$.

Với $|r| > \delta$: $\rho_\delta = \delta(|r| - \frac{1}{2}\delta)$, suy ra $\frac{\partial \rho_\delta}{\partial z} = \delta \cdot \operatorname{sign}(r)$, $\frac{\partial^2 \rho_\delta}{\partial z^2} = 0$.

Không có term nào chứa $y$. $\blacksquare$

### 3.4. Định lý 3: z-space Huber giải quyết Failure A

**Định lý.** *Gradient allocation của z-space Huber trên toàn bộ target range chỉ phụ thuộc vào residual distribution và data density, không bị curvature attenuation.*

**Chứng minh.** Xét band $\mathcal{B}$ trong target space. Gradient contribution từ band này:

$$G_{\text{Huber}}(\mathcal{B}) = \sum_{i \in \mathcal{B}} \left|\frac{\partial \rho_\delta}{\partial z_i}\right| = \sum_{i \in \mathcal{B}} \min(|r_i|, \delta)$$

Giả sử mô hình hội tụ đồng đều (residual magnitude tương đương giữa các band — hợp lý vì curvature bằng nhau), thì:

$$G_{\text{Huber}}(\mathcal{B}) \propto |\mathcal{B}| \cdot \overline{|r|}_{\mathcal{B}} \propto P(y \in \mathcal{B})$$

Gradient allocation tỉ lệ với data density, **không** bị nhân thêm factor $(1-y^2)^2$.

So sánh với y-MSE:

$$G_{\text{y-MSE}}(\mathcal{B}) = \sum_{i \in \mathcal{B}} (1-y_i^2)^2 |r_i|$$

Factor $(1-y_i^2)^2$ giảm mạnh khi $|y_i|$ tăng → under-update ở mid-band và tail. $\blacksquare$

### 3.5. Tại sao KHÔNG cần y-space term

**Nhận định quan trọng:** Curvature-compensated y-MSE tương đương z-MSE.

Nếu bù curvature bằng trọng số $w(y) = \frac{1}{(1-y^2)^2}$:

$$w(y) \cdot \mathcal{L}_y = \frac{1}{(1-y^2)^2} \cdot (1-y^2)^2 \cdot (z-z^*)^2 = (z-z^*)^2$$

**Curvature compensation chính là z-space regression.** Không cần weight phức tạp — làm việc trong không gian đúng từ đầu.

Thực nghiệm cũ cũng xác nhận: L1 (z-strong hybrid, chỉ tăng z-term) cải thiện nhẹ, trong khi L2 (curvature-compensated y-only) cải thiện mạnh. **Driver chính là sửa curvature, không phải thêm z-term.** z-space Huber làm đúng việc này bằng construction.

### 3.6. Tại sao Huber thay vì MSE trong z-space

**Vấn đề:** $\operatorname{atanh}$ phóng đại nhiễu label gần $|y|=1$:

| $y_1$ | $y_2$ | $\|y_1 - y_2\|$ | $\|z_1^* - z_2^*\|$ | Amplification |
|-------|-------|-----------------|---------------------|--------------|
| 0.50 | 0.51 | 0.01 | 0.013 | 1.3× |
| 0.90 | 0.91 | 0.01 | 0.045 | 4.5× |
| 0.98 | 0.99 | 0.01 | 0.201 | **20.1×** |

z-MSE cho gradient $\propto r = z - z^*$. Nếu outlier label tạo $|r| = 3.8$ (khi $y = 0.999$), gradient gấp 3.8× so với sample bình thường tại center.

**Huber cap gradient tại $\delta = 1.0$:**

- Sample bình thường ($|r| \leq 1$): gradient $\propto r$ (giống MSE)
- Outlier ($|r| > 1$): gradient $= \delta \cdot \operatorname{sign}(r) = \pm 1$ (không tăng thêm)

Điều này ngăn 1% outlier labels chi phối training dynamics.

### 3.7. Phân tích Failure B dưới z-space Huber

z-space Huber không giải B hoàn toàn (anti-alignment là vấn đề shared parameters), nhưng **giảm tự nhiên**:

1. **Center gradient giảm từ 72% → ~50%**: không còn curvature amplification → ít "áp lực" đẩy center predictions sai
2. **Mid-band gradient tăng từ 18% → ~30%**: sửa A mà không cần tăng tổng gradient → ít interference hơn A1
3. **Huber cap gradient**: outlier labels không tạo gradient burst → giảm variance → giảm can nhiễu ngẫu nhiên lên center

Kết hợp với **gradient noise injection** (Mục 5), giúp thoát local minimum do anti-alignment, và **kiến trúc mới CRANE** có feature separation tốt hơn (RayStream tách biệt, FiLM tách scalar path), B được dự báo cải thiện đáng kể.

---

## 4. Xử lý edge cases

### 4.1. Clipping target trước atanh

$$z^* = \operatorname{atanh}(\operatorname{clip}(y, -c, c)), \quad c = 0.999$$

**Giới hạn $z^*$:**

| $y$ (raw) | $z^* = \operatorname{atanh}(\operatorname{clip}(y))$ |
|-----------|------------------------------------------------------|
| 0.0 | 0.000 |
| 0.5 | 0.549 |
| 0.9 | 1.472 |
| 0.99 | 2.647 |
| 0.999 | 3.800 |
| 1.0 (raw) | 3.800 (clipped) |

Tổng range $z^* \in [-3.800, 3.800]$. Huber $\delta = 1.0$ cap gradient tại 1.0 — ngay cả khi residual = 3.8, gradient vẫn chỉ là 1.0.

**Tại sao $c = 0.999$ thay vì 0.99:**
- $c = 0.99$ cắt mất thông tin ở $0.99 < |y| < 1.0$, tương ứng với $|z^*| > 2.65$
- $c = 0.999$ bảo toàn thông tin đến $|z^*| = 3.80$, chỉ cắt đúng 0.1% tail
- Khoảng cách giữa $z^*_{\max}$ và giá trị thực tế rất lớn ($y = 0.9999 \Rightarrow z^* \to \infty$) — cần cắt ở đâu đó

### 4.2. Tanh saturation không ảnh hưởng gradient

Loss được tính trên $z$ (logit), **không** trên $p = \tanh(z)$:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \rho_\delta}{\partial z} \cdot \frac{\partial z}{\partial \theta}$$

Không có $\tanh'(z) = 1 - \tanh^2(z)$ trong gradient path. Tanh chỉ applied ở inference: $p = \tanh(z)$.

### 4.3. Giai đoạn đầu training: dự đoán ngẫu nhiên

Khi khởi tạo, $z \sim \mathcal{N}(0, \sigma_{\text{init}})$, $z^*$ phân bố trong $[-3.8, 3.8]$.

Residual $|r| = |z - z^*|$ có thể lớn. Với z-MSE, gradient $= r$ không bị giới hạn → rủi ro explosion. Với z-Huber, gradient bị cap tại $\delta = 1.0$ → **tự động ổn định early training**.

### 4.4. Dải giá trị z và quantization compatibility

Khi distill sang student, teacher logits $z$ là target. Student cũng học bằng z-space Huber.

Khi quantize student sang INT8:
- Range $z \in [-3.8, 3.8]$
- INT8 scale: $7.6 / 256 \approx 0.030$ per step
- Tại center: $0.030$ trong z-space $\approx 0.030$ trong y-space (vì $\tanh'(0) = 1$)
- Tại tail ($y = 0.9$): $0.030$ trong z-space $\approx 0.003$ trong y-space (vì $\tanh'(1.47) \approx 0.10$)
- Resolution đủ cho cả center lẫn tail

---

## 5. Gradient Noise Injection

### 5.1. Động lực

Gradient interference (B2) tạo **anti-aligned gradient trap**: khi $\cos(\nabla_{\text{center}}, \nabla_{\text{mid}}) < 0$, optimizer có thể bị kẹt ở trạng thái mà center over-confident VÀ mid-band under-confident đồng thời, vì bất kỳ bước nào sửa một bên đều làm bên kia xấu đi.

Gradient noise injection giúp **thoát trap** bằng cách thêm nhiễu stochastic, cho phép optimizer "nhảy" qua các vùng loss landscape mà gradient thuần không thể thoát.

### 5.2. Định nghĩa

Sau khi tính gradient $\nabla_\theta \mathcal{L}$, trước khi apply optimizer step:

$$\tilde{\nabla}_\theta = \nabla_\theta \mathcal{L} + \sigma_t \cdot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \mathbf{I})$$

Noise scale $\sigma_t$ giảm dần theo training:

$$\sigma_t = \sigma_0 \cdot \max\!\left(0,\; 1 - \frac{t}{T_{\text{noise}}}\right)$$

### 5.3. Định lý 4: Noise injection tương đương Langevin dynamics có ủ

**Định lý.** *SGD với gradient noise injection và learning rate $\eta_t$ tương đương Langevin dynamics với nhiệt độ hiệu dụng $T_t = \frac{\sigma_t^2}{2\eta_t}$:*

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L} + \sqrt{2\eta_t T_t} \cdot \boldsymbol{\varepsilon} = \theta_t - \eta_t \nabla_\theta \mathcal{L} + \sigma_t \boldsymbol{\varepsilon}$$

**Chứng minh.** Từ định nghĩa:

$$\theta_{t+1} = \theta_t - \eta_t \tilde{\nabla}_\theta = \theta_t - \eta_t(\nabla_\theta \mathcal{L} + \sigma_t \boldsymbol{\varepsilon})$$

So sánh với Langevin update:

$$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \mathcal{L} + \sqrt{2\eta_t T_t} \cdot \boldsymbol{\varepsilon}$$

Suy ra: $\sigma_t = \sqrt{2\eta_t T_t}$, tức $T_t = \frac{\sigma_t^2}{2\eta_t}$. $\blacksquare$

**Ý nghĩa:** Khi $\sigma_t \to 0$ (cuối training), $T_t \to 0$ → dynamics "đóng băng" ở gần nghiệm, giống simulated annealing. Khi $\sigma_t$ lớn (đầu training), $T_t$ cao → exploration mạnh.

### 5.4. Ảnh hưởng lên gradient interference

**Mô hình.** Xét hai band gradient $g_A$ (center) và $g_B$ (mid-band) với $\cos(g_A, g_B) = \alpha < 0$.

Không có noise, update: $\Delta\theta = -\eta(g_A + g_B)$.

Với noise, update: $\Delta\theta = -\eta(g_A + g_B) + \sigma \boldsymbol{\varepsilon}$.

**Xác suất thoát trap.** Tại critical point nơi $g_A + g_B \approx 0$ (gradient triệt tiêu), noise $\sigma \boldsymbol{\varepsilon}$ cung cấp "động năng" để thoát. Xác suất thoát trong 1 bước tỉ lệ với:

$$P_{\text{escape}} \propto \exp\!\left(-\frac{\Delta E}{T_t}\right) = \exp\!\left(-\frac{2\eta_t \Delta E}{\sigma_t^2}\right)$$

trong đó $\Delta E$ là "barrier height" của gradient trap. Khi $T_t$ cao (early training), $P_{\text{escape}}$ lớn → dễ thoát. Khi $T_t \to 0$ (late training), mô hình đã ổn định ở nghiệm tốt hơn.

### 5.5. Chọn siêu tham số

**$\sigma_0$ (noise scale ban đầu):**

$$\sigma_0 = \eta_{\max} \cdot \kappa, \quad \kappa \in [0.5, 1.0]$$

Lý do: noise scale nên tỉ lệ với learning rate. $\kappa = 0.5$ là khởi đầu bảo thủ; $\kappa = 1.0$ nếu cần exploration mạnh hơn.

**$T_{\text{noise}}$ (thời gian dừng noise):**

$$T_{\text{noise}} = 0.8 \cdot T_{\text{total}}$$

Noise linearly decay về 0 ở 80% training. 20% cuối không noise → fine-tune calibration.

**Khuyến nghị mặc định:** $\kappa = 0.5$, $T_{\text{noise}} = 0.8 \cdot T_{\text{total}}$.

### 5.6. Tương tác với BatchNorm

Khi thêm noise vào gradient, running statistics của BatchNorm bị nhiễu. Tuy nhiên:
- Noise scale $\sigma_t$ giảm dần → nhiễu BN giảm theo
- BN running average (momentum = 0.1) tự smoothing → ít nhạy cảm với noise
- 20% cuối training không noise → BN statistics ổn định

Nếu vẫn lo ngại, có thể freeze BN running stats khi noise > 0, nhưng **không khuyến nghị cho v0** vì thêm complexity.

---

## 6. Cosine Annealing với Warmup

### 6.1. Định nghĩa

Learning rate schedule 2 pha:

**Pha 1 — Linear warmup (bước $0 \to W$):**

$$\eta_t = \eta_{\max} \cdot \frac{t}{W}$$

**Pha 2 — Cosine annealing (bước $W \to T_{\text{total}}$):**

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\pi \cdot \frac{t - W}{T_{\text{total}} - W}\right)\right)$$

### 6.2. Tại sao cosine thay vì step decay

**Step decay** tạo nhảy discontinuous trong learning rate. Mỗi nhảy gây:
- Gradient magnitude đột ngột thay đổi
- Tạm thời phá balance giữa center và mid-band gradient
- Có thể "quên" điều đã học nếu step quá lớn

**Cosine annealing** giảm mượt, liên tục:
- Gradient magnitude giảm dần, dễ dự đoán
- Balance giữa A và B không bị disrupt đột ngột
- Tự nhiên điều chỉnh exploration/exploitation: đầu training LR cao → exploration, cuối LR thấp → fine-tune

### 6.3. Tại sao cần warmup

Đầu training:
1. **BatchNorm chưa ổn**: running stats bắt đầu từ zero, cần vài bước để hội tụ
2. **Gradient noise lớn**: random initialization tạo gradient lớn, dễ divergence
3. **Mixed-band batching**: batch đầu có variance cao, cần LR nhỏ để không overshoot

Warmup cho phép BN ổn định và gradient settle trước khi tăng LR lên giá trị đầy đủ.

### 6.4. Chọn siêu tham số

**$\eta_{\max}$ (peak learning rate):**

Phụ thuộc vào batch size và model size. Với AdamW và batch size $B = 1024$:

$$\eta_{\max} = 3 \times 10^{-4}$$

Quy tắc linear scaling: nếu $B$ tăng $k$ lần, $\eta_{\max}$ tăng $k$ lần (tối đa $1 \times 10^{-3}$). Với $B = 2048$: $\eta_{\max} = 6 \times 10^{-4}$.

**$\eta_{\min}$ (minimum learning rate):**

$$\eta_{\min} = \frac{\eta_{\max}}{100} = 3 \times 10^{-6}$$

**$W$ (warmup steps):**

$$W = \min(2000,\; 0.02 \cdot T_{\text{total}})$$

2% tổng steps hoặc 2000 steps, lấy giá trị nhỏ hơn.

### 6.5. Tương tác với gradient noise

Nhiệt độ hiệu dụng của Langevin dynamics:

$$T_t = \frac{\sigma_t^2}{2\eta_t}$$

Khi $\eta_t$ giảm (cosine decay), nếu $\sigma_t$ không đổi, $T_t$ tăng → "ủ nóng lại" không mong muốn.

Do đó, $\sigma_t$ phải decay nhanh hơn hoặc đồng bộ với $\eta_t$. Với $\sigma_t$ linearly decaying về 0 ở 80% training:

- Đầu training: $\eta_t$ tăng (warmup), $\sigma_t$ cao → exploration
- Giữa training: $\eta_t$ giảm (cosine), $\sigma_t$ giảm → ổn định
- Cuối training: $\eta_t \to \eta_{\min}$, $\sigma_t = 0$ → fine-tune

Schedule đồng bộ: cả LR và noise cùng giảm → $T_t$ giảm đơn điệu → **ủ lạnh đúng**.

---

## 7. Mixed-Band Batching

### 7.1. Động lực

Random sampling từ phân phối chuông → mỗi batch over-represents center. Variance của gradient estimator cao vì:
1. Center và mid-band gradient anti-aligned ($\cos = -0.64$)
2. Random batch có thể toàn center samples → gradient chỉ phản ánh center
3. Batch tiếp theo có thể nhiều mid-band → gradient chỉ phản ánh mid-band

High variance → slow convergence, unstable training.

### 7.2. Định nghĩa

Chia target space thành $K$ bands. Mỗi batch lấy $B/K$ samples từ mỗi band.

**Band definition (dựa trên $|y|$):**

| Band | Range | Ký hiệu | Xấp xỉ % data |
|------|-------|---------|---------------|
| $B_0$ | $0 \leq |y| < 0.10$ | Ultra-center | ~35% |
| $B_1$ | $0.10 \leq |y| < 0.30$ | Near-center | ~22% |
| $B_2$ | $0.30 \leq |y| < 0.55$ | Mid | ~20% |
| $B_3$ | $0.55 \leq |y| < 0.75$ | Upper-mid | ~13% |
| $B_4$ | $0.75 \leq |y| \leq 1.0$ | Tail | ~10% |

Mỗi band lấy $B/K = B/5$ samples. Within band, samples drawn uniformly.

### 7.3. Định lý 5: Mixed-band batching giảm phương sai gradient estimator

**Định lý (Stratified Sampling Variance Reduction).** *Cho gradient estimator $\hat{g} = \frac{1}{n}\sum_{i=1}^{n} \nabla_\theta \mathcal{L}_i$. Stratified sampling với $K$ strata giảm phương sai so với random sampling:*

$$\operatorname{Var}_{\text{strat}}(\hat{g}) = \sum_{k=1}^{K} \frac{p_k^2}{n_k} \operatorname{Var}(g_k)$$

$$\operatorname{Var}_{\text{random}}(\hat{g}) = \sum_{k=1}^{K} \frac{p_k}{n} \operatorname{Var}(g_k) + \frac{1}{n} \sum_{k=1}^{K} p_k \|\mathbb{E}[g_k] - \mathbb{E}[g]\|^2$$

*trong đó $p_k$ là proportion của stratum $k$, $n_k$ là samples từ stratum $k$, $g_k$ là gradient từ stratum $k$.*

**Chứng minh.** Đây là kết quả chuẩn của sampling theory (Cochran, 1977). Xem *Sampling Techniques*, Chapter 5. $\blacksquare$

**Hiệu quả giảm phương sai:**

Term $\frac{1}{n}\sum_k p_k \|\mathbb{E}[g_k] - \mathbb{E}[g]\|^2$ là **between-strata variance** — bị loại bỏ hoàn toàn bởi stratified sampling.

Với chess data, between-strata variance **lớn** vì gradient direction khác biệt giữa các band:

$$\|\mathbb{E}[g_{B_0}] - \mathbb{E}[g_{B_3}]\|^2 \gg 0 \quad (\cos = -0.64)$$

Do đó, mixed-band batching cho giảm phương sai đáng kể.

### 7.4. Lợi ích phụ: đảm bảo coverage

Với random sampling và batch size 1024, xác suất có ít hơn 10 samples từ tail ($|y| > 0.75$, chiếm 10% data):

$$P(n_{\text{tail}} < 10) = \sum_{k=0}^{9} \binom{1024}{k} (0.10)^k (0.90)^{1024-k} \approx 0.004\%$$

Xác suất thấp nhưng không为零. Với mixed-band batching, mỗi band có đúng $B/5$ samples → **đảm bảo 100% coverage**.

### 7.5. Triển khai thực tế

**Cách triển khai với sharded data:**

1. **Pre-compute band index:** Khi encode mỗi sample, tính $|y|$ và gán band index $k \in \{0,1,2,3,4\}$.
2. **Shard structure:** Mỗi shard lưu samples grouped by band. Hoặc đơn giản hơn: lưu band index cùng sample, sampler chọn theo band.
3. **Sampler logic:**
   ```python
   for each band k:
       draw B/K samples uniformly from band k
   concatenate → batch of size B
   shuffle within batch
   ```
4. **Không cần re-encode:** Chỉ cần thêm 1 byte band index per sample.

**Lưu ý về perspective:** $y$ phải là STM-relative (đã có trong encode spec). Band index tính trên $|y|$ nên đối xứng.

### 7.6. Tương tác với data distribution tự nhiên

Phân phối chuông là **đúng thực tế** — phần lớn thế cờ cờ vua là cân bằng. Mixed-band batching **không thay đổi loss weighting**, chỉ đảm bảo gradient estimator có variance thấp hơn.

So sánh với band-balanced sampling (A2 cũ): A2 oversampling tail trong loss → thay đổi gradient allocation. Mixed-band batching chỉ thay đổi **sampling** → gradient allocation vẫn tỉ lệ với data density, nhưng variance thấp hơn.

---

## 8. Spec tổng hợp: Loss Function

### 8.1. Loss function hoàn chỉnh

$$\mathcal{L}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \rho_\delta(z_i - z_i^*)$$

trong đó:
- $z_i = f_\theta(x_i)$: raw logit output từ Value Head
- $z_i^* = \operatorname{atanh}(\operatorname{clip}(y_i, -0.999, 0.999))$
- $\rho_\delta(r)$: Huber loss, $\delta = 1.0$
- $B$: batch size

### 8.2. Gradient computation

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{B} \cdot \begin{cases} (z_i - z_i^*) & \text{nếu } |z_i - z_i^*| \leq 1.0 \\ \operatorname{sign}(z_i - z_i^*) & \text{nếu } |z_i - z_i^*| > 1.0 \end{cases}$$

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{i=1}^{B} \frac{\partial \mathcal{L}}{\partial z_i} \cdot \frac{\partial z_i}{\partial \theta}$$

### 8.3. Siêu tham số loss

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| $\delta$ | 1.0 | Huber threshold |
| $c$ | 0.999 | Clipping target trước atanh |

**Chỉ 2 siêu tham số.** So với loss cũ: $\lambda_y, \lambda_z, \beta, \delta, \lambda_{\text{center}}, \tau_{\text{center}}, \lambda_{\text{start}}, \lambda_{\text{end}}$ — 8 siêu tham số.

---

## 9. Spec tổng hợp: Training Strategy

### 9.1. Optimizer

**AdamW** (Loshchilov & Hutter, 2019):

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| $\beta_1$ | 0.9 | Standard |
| $\beta_2$ | 0.95 | Thấp hơn 0.999 (Transformer chuẩn) vì dataset chess ít noisy hơn NLP |
| $\epsilon$ | $10^{-8}$ | Standard |
| Weight decay | 0.01 | Standard cho model ~10M params |

### 9.2. Learning Rate Schedule

| Tham số | Giá trị |
|---------|---------|
| $\eta_{\max}$ | $3 \times 10^{-4}$ (batch 1024) |
| $\eta_{\min}$ | $3 \times 10^{-6}$ |
| Warmup steps $W$ | $\min(2000, 0.02 \cdot T_{\text{total}})$ |
| Schedule | Cosine annealing sau warmup |

### 9.3. Gradient Noise Injection

| Tham số | Giá trị |
|---------|---------|
| $\sigma_0$ | $0.5 \cdot \eta_{\max}$ |
| $T_{\text{noise}}$ | $0.8 \cdot T_{\text{total}}$ |
| Decay | Linear về 0 |

### 9.4. Mixed-Band Batching

| Tham số | Giá trị |
|---------|---------|
| Số bands $K$ | 5 |
| Samples per band | $B / 5$ |
| Band boundaries | $[0, 0.10), [0.10, 0.30), [0.30, 0.55), [0.55, 0.75), [0.75, 1.0]$ |

### 9.5. Batch Size

| Config | Batch Size | $\eta_{\max}$ |
|--------|-----------|---------------|
| v0-sanity | 512 | $1.5 \times 10^{-4}$ |
| v0-main | 1024 | $3 \times 10^{-4}$ |

### 9.6. Gradient Clipping

Clip gradient norm toàn cục:

$$\|\nabla_\theta\|_2 \leq 1.0$$

Áp dụng **sau** noise injection, **trước** optimizer step. Lý do: noise có thể tạo gradient norm rất lớn trong early training.

### 9.7. Training Duration

Ước lượng dựa trên dataset Lichess evals (~50M positions):

| Config | Epochs | Steps (B=1024) | Thời gian ước lượng (1×L4 GPU) |
|--------|--------|----------------|---------------------------------|
| v0-sanity | 10 | ~500K | ~8 giờ |
| v0-main | 20 | ~1M | ~16 giờ |

---

## 10. So sánh: Loss cũ vs Loss mới

| Khía cạnh | Loss cũ (hybrid y+z + curriculum) | Loss mới (z-space Huber) |
|---|---|---|
| Số terms | 2–3 (y-loss, z-loss, center penalty) | **1** |
| Số siêu tham số | 8+ | **2** |
| Failure A | Cần curvature compensation | **Fix bằng construction** |
| Failure B | Chữa cháy thất bại | **Giảm tự nhiên** (50% vs 72% center gradient) |
| Outlier robustness | Kém (MSE nhạy cảm) | **Tốt** (Huber cap) |
| Curriculum | Cần | **Không cần** |
| Dễ debug | Khó (nhiều terms tương tác) | **Dễ** (1 term, gradient rõ ràng) |
| Tương thích distill | Cần cùng curriculum | **Trực tiếp** (teacher z → student z) |
| Tương thích quantize | Phức tạp | **Đơn giản** (z range cố định) |

---

## 11. Evaluation Protocol

### 11.1. Oracle-based metrics (kế thừa từ thí nghiệm cũ)

**Failure A metrics:**

| Metric | Ý nghĩa | Mục tiêu |
|--------|---------|----------|
| `oracle_stable_0.7_slope` | Hiệu chỉnh biên độ mid-band | $\geq 0.85$ |
| `oracle_midband_mae_sum_stable` | Tổng MAE mid-band | Giảm ≥30% vs baseline |

**Failure B metrics:**

| Metric | Ý nghĩa | Mục tiêu |
|--------|---------|----------|
| `oracle_center_amp_ratio` | Biên độ center so với oracle | $\leq 2.0$ |
| `oracle_center_false_0.1eq` | Tỉ lệ center false decisive | $\leq 0.20$ |

**Mục tiêu thiết kế:**
- Baseline cũ: slope = 0.575, amp_ratio = 5.85
- L4 (best old): slope = 0.618, amp_ratio = 6.38
- **CRANE-v0 target**: slope ≥ 0.85, amp_ratio ≤ 2.0

### 11.2. Checkpoint selection gates

Checkpoint được chọn theo **gate thác**:

1. **Gate 1 (broad fit):** `oracle_stable_0.7_slope` ≥ 0.80 → nếu không pass, loại
2. **Gate 2 (center):** `oracle_center_amp_ratio` ≤ 3.0 → nếu không pass, loại
3. **Gate 3 (fine):** Tối thiểu `oracle_midband_mae_sum_stable` trong các checkpoint pass Gate 1+2

**Không dùng global MSE** làm metric ra quyết định (đã chứng minh không đáng tin).

### 11.3. Monitoring trong training

Log mỗi 1000 steps:

| Metric | Mục đích |
|--------|----------|
| Loss (overall + per-band) | Theo dõi hội tụ |
| $\|z\|_{\text{mean}}$ theo band | Chẩn đoán saturation |
| Gradient norm (overall + per-component) | Phát hiện gradient issues |
| Gate value mean (RayFusion) | Phát hiện gate collapse |
| $\|\gamma\|_\infty$, $\|\beta\|_\infty$ (FiLM) | Phát hiện FiLM instability |
| Learning rate, noise scale | Verify schedule |
| Band composition của batch | Verify mixed-band batching |

---

## 12. Ablation Plan cho Loss/Training

Chạy trên **v0-sanity** scale (8 ResBlocks, C=160, 3 attention blocks):

| Run | Config | Giả thuyết kiểm chứng |
|-----|--------|----------------------|
| L0 | z-MSE (no Huber, no noise, no mixed-band, cosine) | Baseline: z-space có đủ sửa A không? |
| L1 | z-Huber (δ=1.0, no noise, no mixed-band, cosine) | Huber cap có ích không? |
| L2 | z-Huber + gradient noise | Noise có giúp B không? |
| L3 | z-Huber + mixed-band batching | Stratified sampling có giảm variance không? |
| L4 | z-Huber + noise + mixed-band + cosine | Full proposal |
| L5 | z-Huber + noise + mixed-band + cosine + warmup 5% | Warmup duration có nhạy cảm không? |

**Metrics chính:** oracle_stable_0.7_slope, oracle_center_amp_ratio, oracle_midband_mae_sum_stable.

**Nguyên tắc:** Mỗi run chỉ thêm 1 thay đổi so với run trước → có thể attribute improvement.

---

## 13. Điều kiện Pass/Fail

### 13.1. Pass (mở v1)

v0 được coi là pass nếu **đồng thời**:

1. `oracle_stable_0.7_slope` ≥ 0.80 (sửa A)
2. `oracle_center_amp_ratio` ≤ 3.0 (cải thiện B)
3. Broad validation metrics không regress so với L4 baseline
4. Offline full-test confirm không có failure mode mới

### 13.2. Fail

Nếu v0 fail:
- **Nếu A chưa sửa:** Vấn đề không phải loss → kiểm tra architecture, encode
- **Nếu B chưa sửa:** Thêm oracle-center auxiliary set hoặc data-level filtering, không thêm loss complexity
- Không mở policy, RL, hay distillation cho đến khi value-only pass

---

## 14. Quyết định cuối cùng

**Loss function:** z-space Huber $\mathcal{L} = \frac{1}{B}\sum_i \rho_\delta(z_i - z_i^*)$, $\delta = 1.0$, $c = 0.999$

**Training strategy:** AdamW + Cosine Annealing + Warmup + Gradient Noise Injection + Mixed-Band Batching

**Siêu tham số tổng hợp:**

| Tham số | Giá trị | Loại |
|---------|---------|------|
| $\delta$ | 1.0 | Loss |
| $c$ | 0.999 | Loss |
| $\eta_{\max}$ | $3 \times 10^{-4}$ | Optimizer |
| $\eta_{\min}$ | $3 \times 10^{-6}$ | Optimizer |
| $\beta_1, \beta_2$ | 0.9, 0.95 | Optimizer |
| Weight decay | 0.01 | Optimizer |
| Warmup | $\min(2000, 0.02 \cdot T)$ | Schedule |
| $\sigma_0$ | $0.5 \cdot \eta_{\max}$ | Noise |
| $T_{\text{noise}}$ | $0.8 \cdot T$ | Noise |
| $K$ (bands) | 5 | Batching |
| Grad clip | 1.0 | Stability |

**Tổng: 11 siêu tham số.** Trong đó chỉ 2 thuộc loss, 6 thuộc optimizer/schedule, 2 thuộc noise, 1 thuộc batching. Mỗi cái đều có lý do lý thuyết rõ ràng, không phải điều chỉnh theo "cảm giác".

Đây là loss function đơn giản nhất giải quyết đúng root cause, với chiến lược training được chứng minh toán học, không dựa trên trực giác, không phải "chữa cháy".
