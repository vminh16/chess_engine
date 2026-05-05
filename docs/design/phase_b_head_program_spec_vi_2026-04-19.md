# Spec Pha B: Head Program Sau Pha A

Ngày: `2026-04-19`  
Trạng thái: `active design spec`  
Phạm vi: teacher-network line `architecture_v2`, sau khi `Pha A` (`R1`, `R2`) đã cho đủ evidence để chuyển pha

---

## 1) Mục tiêu

Pha B được mở để trả lời câu hỏi hẹp nhưng rất quan trọng:

> Nếu giữ backbone đủ mạnh để không mất biểu diễn, rồi thay head theo hướng giảm xung đột gradient ở readout, liệu có thể đóng `A-gate` mà không làm hỏng `B-side` hay không?

Nói ngắn gọn:

1. `Pha A` đã kiểm tra `capacity-only` và `sampling-only`.
2. `Pha B` kiểm tra `head-first intervention`.
3. `Pha C` (`encode refresh`) chỉ mở nếu `Pha B` vẫn không pass.

Spec này chốt **4 thực nghiệm**:

1. `B1`: `16/256 + SimplifiedGlobalHead + baseline objective`
2. `B2`: `16/256 + SimplifiedGlobalHead + sign-stratified sampling`
3. `B3`: `16/256 + RegimeSeparatedHead + best sampling from B1/B2`
4. `B4`: `20/256 + winning head recipe from B1-B3` để xác nhận line lớn

---

## 2) Bằng chứng buộc phải mở Pha B

### 2.1 R1 bác bỏ giả thuyết `shrink-only là đủ`

Nguồn:  
[`runs/dgrn_5m_report1_r1_shrink_run1/reports/history.csv`](../../runs/dgrn_5m_report1_r1_shrink_run1/reports/history.csv)  
[`runs/dgrn_5m_report1_r1_shrink_run1/reports/decision_summary.json`](../../runs/dgrn_5m_report1_r1_shrink_run1/reports/decision_summary.json)  
[`runs/dgrn_5m_report1_r1_shrink_run1/reports/l4_reference.json`](../../runs/dgrn_5m_report1_r1_shrink_run1/reports/l4_reference.json)

`R1 = 8b/128d + ResidualGainValueHead + baseline objective`

Best observed:

- `best_midband = 0.6682138422`
- `best_slope = 0.5318312020`
- `best_center = 0.5507412479`

Hard A-gate từ L4:

- `M_gate = 0.5997277169`
- `S_gate = 0.5978235041`

Khoảng cách còn lại:

- `ΔM = 0.6682138422 - 0.5997277169 = 0.0684861253`
- `ΔS = 0.5978235041 - 0.5318312020 = 0.0659923021`

Kết luận:

1. `R1` **không gần pass** theo nghĩa kỹ thuật.
2. `R1` cũng **không thắng B** vì `center_score = 0.5507` vẫn kém L4 (`0.5252`).
3. Vì vậy, giảm mạnh backbone xuống `8b/128d` là đủ để tạo tín hiệu chẩn đoán, nhưng không đủ để ra recipe cuối.

### 2.2 R2 cho tín hiệu thật, nhưng tín hiệu đó nằm ở B chứ không nằm ở A

Nguồn dùng để kết luận:

- [`runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/history.csv`](../../runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/history.csv)
- [`runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/decision_summary.json`](../../runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/decision_summary.json)

Lưu ý: folder `dgrn_5m_report1_r2_sign_run1` thiếu `history.csv` và `decision_summary.json`, nên **không được dùng** làm nguồn chính cho kết luận khoa học hiện tại.

`R2 = 8b/128d + ResidualGainValueHead + sign-stratified sampling`

Best observed tại snapshot hiện tại:

- `best_midband = 0.6785189647`
- `best_slope = 0.4828366405`
- `best_center = 0.4495319558`

So với `R1`:

- `center` tốt hơn `0.1012092921`
- `midband` xấu hơn `0.0103051225`
- `slope` xấu hơn `0.0489945614`

Kết luận:

1. `sign-stratified` **không vô ích**; nó thực sự kéo `B-side` đi đúng hướng.
2. Nhưng `sign-stratified` **không cứu được A**, thậm chí còn làm A tệ hơn `R1`.
3. Như vậy, bottleneck không còn trông giống `capacity-only` hay `sampling-only`.

### 2.3 Evidence mạnh nhất hiện tại vẫn chỉ vào head

Nguồn:

- [`experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv`](../../experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv)
- [`docs/reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md`](../reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md)

Cặp objective quan trọng:

- `center_raw_0_005` vs `mid_05_07`

Cosine đã đo:

- `backbone cosine = -0.2792`
- `head cosine = -0.7239`

Kết luận:

1. Xung đột gradient tập trung ở `head` mạnh hơn nhiều so với backbone.
2. Điều này là cơ sở trực tiếp để mở `Pha B`.

---

## 3) Vì sao không quay lại `20/256` ngay, và vì sao chọn `16/256`

### 3.1 Không quay lại `20/256` ngay

Nguồn:

- [`runs/dgrn_5m_ft2_t4_run1/reports/history.csv`](../../runs/dgrn_5m_ft2_t4_run1/reports/history.csv)
- [`docs/reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md`](../reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md)

Line `20/256` đã có rất nhiều evidence, nhưng:

1. đã tốn nhiều epoch,
2. vẫn không qua `A-gate`,
3. và không tạo được checkpoint vừa tốt A vừa tốt B.

Nếu quay về `20/256` ngay trong `B1`, chi phí sẽ cao hơn trong khi causal attribution lại bẩn hơn: nếu run thắng, vẫn chưa rõ thắng vì head hay vì backbone lớn hơn.

### 3.2 Không giữ `8/128` cho Pha B

`8/128` đã hoàn thành vai trò diagnostic của nó.  
Giữ nguyên `8/128` cho Pha B sẽ làm câu hỏi “head có sửa được không?” bị lẫn với “backbone có quá yếu không?”.

### 3.3 `16/256` là điểm cân bằng hợp lý nhất

Tính trực tiếp từ code hiện tại:

- `20/256`: `9,047,682` params
- `16/256`: `7,501,058` params
- `12/256`: `5,954,434` params
- `8/128`: `1,116,418` params

`16/256` hợp lý vì:

1. giữ nguyên `width = 256`, tức là vẫn gần line lớn hơn nhiều so với `8/128`;
2. giảm khoảng `17%` params so với `20/256`, nên rẻ hơn và ít confound hơn;
3. đủ để kiểm tra giả thuyết “head là bottleneck chính” mà không bị shrink quá mạnh.

---

## 4) Nguyên tắc cố định của Pha B

Những gì **giữ nguyên** qua cả `B1-B4`:

1. cùng split dữ liệu hiện tại
2. cùng target convention `STM`
3. cùng protocol eval/gate với `FT1/L4`
4. cùng objective family đang active trong notebook report1 line
5. cùng cách log `history.csv`, `decision_summary.json`, `selected_checkpoint_eval.json`

Những gì **không làm** trong Pha B:

1. không đổi `encode`
2. không thêm phase planes / halfmove planes
3. không đổi normalization toàn backbone
4. không bật thêm một “mega-run” trộn nhiều thay đổi đồng thời

Lý do: Pha B là pha kiểm định `head-first`, không phải pha đổi nhiều trục cùng lúc.

---

## 5) Chỉ tiêu đánh giá

### 5.1 Hard A-gate

Từ L4 reference:

$$
M \equiv \text{oracle\_midband\_mae\_sum\_stable} \le 0.5997277169
$$

$$
S \equiv \text{oracle\_stable\_0.7\_slope} \ge 0.5978235041
$$

### 5.2 B-side acceptance

Pha B không được phép “pass A bằng cách phá hỏng center”.

Đặt 2 mức:

**B-pass tối thiểu**

$$
\text{center\_score} \le 0.5251979644
$$

tức là không tệ hơn L4.

**B-pass mạnh**

$$
\text{center\_score} \le 0.46
$$

Mốc `0.46` được chọn vì:

1. `R2` đã đi xuống `0.4495`,
2. nên đây là vùng B đã có bằng chứng repo nội bộ rằng hoàn toàn đạt được.

### 5.3 Run được xem là “pass”

Một run trong Pha B được công nhận là **pass** nếu thỏa đồng thời:

1. `M <= 0.5997277169`
2. `S >= 0.5978235041`
3. `center_score <= 0.5251979644`

Run được xem là **pass mạnh** nếu thỏa thêm:

4. `center_score <= 0.46`

---

## 6) Thiết kế 4 thực nghiệm

## 6.1 B1 - `16/256 + SimplifiedGlobalHead + baseline objective`

### Cấu hình

- backbone: `num_blocks=16`, `hidden_dim=256`
- head: `SimplifiedGlobalHead`
- sampling: baseline / band-balanced như `R1`
- objective: baseline objective line hiện tại
- thời lượng: `10-12 epoch`

### Giả thuyết

Nếu bottleneck chính nằm ở nhánh flatten lớn của current head, thì bỏ flatten bottleneck sẽ:

1. cải thiện `A-side` rõ trước,
2. giữ hoặc ít nhất không phá `B-side`,
3. giảm conflict readout mà không cần đổi dữ liệu.

### Vì sao hợp lý

Từ [`model/architecture_v2/head.py`](../../model/architecture_v2/head.py):

- `ResidualGainValueHead_256`: `1,272,578` params
- `SimplifiedGlobalHead_256`: `73,985` params

Đây là giảm rất mạnh đúng vào readout bottleneck, trong khi backbone vẫn được giữ đủ lớn.

### Tiêu chí pass cho B1

1. vượt `R1` trên ít nhất một trong hai trục A:
   - `midband < 0.6682` hoặc
   - `slope > 0.5318`
2. đồng thời `center_score <= 0.60`

### Nếu B1 pass mạnh

Nếu B1 qua hẳn hard gate và `center <= L4`, dừng Pha B sớm và chuyển sang `B4` để xác nhận trên line lớn.

### Nếu B1 fail

Fail của B1 có ý nghĩa:

1. current head không phải chỉ bị vấn đề “flatten quá lớn” ở mức đơn giản nhất,
2. hoặc conflict còn mang tính semantic hơn, không chỉ là vấn đề capacity/readout size.

Khi đó chuyển `B2`.

---

## 6.2 B2 - `16/256 + SimplifiedGlobalHead + sign-stratified`

### Cấu hình

- backbone: `16/256`
- head: `SimplifiedGlobalHead`
- sampling: `sign-stratified`
- objective: giữ như B1
- thời lượng: `10-12 epoch`

### Giả thuyết

`R2` đã chứng minh `sign-stratified` giúp B.  
Nếu current problem là:

1. head đang xung đột quá mạnh,
2. còn sampling lại đang giúp đúng subset B,

thì ghép `head sạch hơn + sign-stratified` có thể giữ được lợi ích B mà không làm A sụp như `R2`.

### Tiêu chí pass cho B2

1. `center_score < B1_best_center`
2. `midband <= B1_best_midband + 0.01`
3. `slope >= B1_best_slope - 0.01`

Nếu đạt hard gate và `center <= L4`, đây là line thắng rõ.

### Nếu B2 pass

Kết luận khi đó là:

1. `sampling` không nên bị bỏ,
2. vấn đề của `R2` trước đây chủ yếu do head cũ,
3. recipe tốt hơn là `better head + sign-stratified`, không phải `sampling-only`.

### Nếu B2 fail

Khi đó inference mạnh nhất là:

1. conflict không chỉ do flatten bottleneck,
2. mà còn do một scalar readout đang phải encode đồng thời `direction` và `magnitude`.

Khi đó chuyển `B3`.

---

## 6.3 B3 - `16/256 + RegimeSeparatedHead + best sampling from B1/B2`

### Cấu hình

- backbone: `16/256`
- head: `RegimeSeparatedHead`
- sampling:
  - nếu `B1` tốt hơn: dùng baseline sampling
  - nếu `B2` tốt hơn: dùng `sign-stratified`
- objective: giữ như B1/B2
- thời lượng: `10-12 epoch`

### Giả thuyết

`RegimeSeparatedHead` kiểm tra trực diện giả thuyết:

> Một scalar value head đang buộc cùng một affine path phải học cả `sign` lẫn `magnitude`, và chính việc đó tạo ra interference ở head.

Head này tách:

$$
z = z_m \cdot \tanh(z_s)
$$

với:

- `z_m`: branch magnitude
- `z_s`: branch sign

### Tiêu chí pass cho B3

1. vượt `B1` và `B2` trên ít nhất một trục A,
2. không để `center_score` regress quá `+0.03` so với best line trước đó,
3. ưu tiên line có `selection_score_v2` tốt nhất trong số các line head mới

### Nếu B3 pass

Kết luận:

1. bottleneck nằm ở semantics của value readout, không chỉ do head quá to,
2. `sign/magnitude decoupling` là inductive bias hợp với bài toán,
3. line này trở thành candidate mạnh nhất cho scale-up xác nhận.

### Nếu B3 fail

Kết luận:

1. cả `simplify head` lẫn `regime-separated head` đều không đủ để đóng A/B frontier,
2. lúc đó head conflict có thể chỉ là phần nổi,
3. root cause còn lại nhiều khả năng nằm ở representation/state aliasing,
4. khi đó mở `Pha C` (`encode refresh`) là hợp lý.

---

## 6.4 B4 - `20/256 + winning head recipe from B1-B3`

### Cấu hình

- backbone: `20/256`
- head: recipe thắng nhất trong `B1-B3`
- sampling: theo recipe thắng
- objective: giữ nguyên
- thời lượng: `8-10 epoch`

### Mục tiêu

`B4` **không phải run khám phá**.  
`B4` chỉ làm một việc:

> xác nhận rằng recipe head thắng ở `16/256` không bị mất khi scale lại lên torso lớn hơn.

### Vì sao cần B4

Nếu một head mới thắng ở `16/256`, vẫn còn hai khả năng:

1. thắng vì head đúng thật,
2. hoặc thắng vì `16/256` tình cờ là điểm regularization tốt hơn.

`B4` giúp tách hai khả năng đó.

### Nếu B4 pass

Kết luận mạnh:

1. head recipe mới là line chính thức để đi tiếp,
2. có thể cân nhắc scale run dài hơn hoặc tích hợp vào line kế tiếp,
3. chưa cần chạm `encode`.

### Nếu B4 fail nhưng B1/B2/B3 pass

Kết luận:

1. head recipe mới là đúng,
2. nhưng `20/256` hiện không còn là operating point tốt,
3. line phù hợp hơn trong ngắn hạn là `16/256`, không phải `20/256`.

---

## 7) Luật stop/go

## 7.1 Early stop chung cho từng run

Dừng sớm nếu:

1. có `non-finite`
2. metric A và B cùng regress rõ trong `3` epoch liên tiếp
3. sau `epoch 6` không vượt được baseline trực tiếp của run trước đó trên trục mà run đó được thiết kế để cải thiện

## 7.2 Luật chuyển giữa các run

1. Nếu `B1` pass mạnh:
   - bỏ `B2/B3`
   - chạy `B4`
2. Nếu `B1` fail:
   - chạy `B2`
3. Nếu `B2` fail:
   - chạy `B3`
4. Nếu `B3` fail:
   - đóng `Pha B`
   - chuyển `Pha C`

## 7.3 Luật kết thúc Pha B

Pha B được xem là:

**thành công**

nếu có ít nhất một run:

1. pass hard A-gate,
2. đồng thời `center_score <= L4 center`,
3. và checkpoint đó reproducible đủ để justify một run xác nhận tiếp theo.

**thất bại**

nếu:

1. `B1/B2/B3` đều fail,
2. hoặc chỉ thắng B nhưng không đóng được A,
3. hoặc chỉ thắng A nhưng B regress vượt L4 rõ rệt.

---

## 8) Ý nghĩa khoa học nếu Pha B pass hoặc fail

### 8.1 Nếu Pha B pass

Điều đó sẽ cho phép kết luận khá mạnh:

1. `head` là root cause chi phối hơn `encode`,
2. `Pha A` thất bại chủ yếu vì head cũ làm méo frontier,
3. chưa cần mở `encode refresh`,
4. line tiếp theo nên là xác nhận / scale-up chứ không phải rebuild data.

### 8.2 Nếu Pha B fail

Khi đó, kết luận hợp lý nhất là:

1. `head` đúng là một phần vấn đề, nhưng không phải phần đủ để giải toàn bộ failure,
2. `capacity-only`, `sampling-only`, và `head-only` đều đã bị kiểm đủ,
3. bước tiếp theo hợp lý nhất là `Pha C - encode refresh`,
4. đặc biệt là patch các state hiện đang có trong `Board` nhưng chưa được encode, như `halfmove_clock` và `phase`.

---

## 9) Những gì spec này cố ý không làm

1. không mở block sweep rộng (`12/256`, `14/256`, `16/256`, `18/256`, ...)
2. không đổi BN -> GN trong cùng pha
3. không thêm uncertainty head ở bước đầu
4. không rebuild dataset trong Pha B

Lý do:

1. block sweep rộng sẽ làm pha này biến thành `capacity study`, không còn là `head program`
2. GN là hypothesis khác
3. uncertainty head hợp production/search hơn là hard-gate repair trước mắt
4. rebuild dataset thuộc `Pha C`

---

## 10) Khuyến nghị thực thi

Thứ tự thực tế khuyến nghị:

1. chạy `B1`
2. nếu `B1` fail, chạy `B2`
3. nếu `B2` fail, chạy `B3`
4. chỉ chạy `B4` khi đã có winner rõ từ `B1-B3`

Nếu cần tiết kiệm compute hơn nữa:

1. giữ `B1`
2. chọn `B2` hoặc `B3` tùy metric sau `B1`

nhưng trong điều kiện bình thường, matrix `B1-B4` là đầy đủ và đúng tinh thần thí nghiệm nhất hiện nay.

---

## 11) Nguồn

### Nội bộ repo

1. [`runs/dgrn_5m_report1_r1_shrink_run1/reports/history.csv`](../../runs/dgrn_5m_report1_r1_shrink_run1/reports/history.csv)
2. [`runs/dgrn_5m_report1_r1_shrink_run1/reports/decision_summary.json`](../../runs/dgrn_5m_report1_r1_shrink_run1/reports/decision_summary.json)
3. [`runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/history.csv`](../../runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/history.csv)
4. [`runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/decision_summary.json`](../../runs/dgrn_5m_report1_r2_sign_run1_local_opt/reports/decision_summary.json)
5. [`docs/reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md`](../reports/ft2_ab_failure_report_and_next_spec_vi_2026-04-16.md)
6. [`docs/design/post_ft2_progressive_plan_spec_vi_2026-04-17.md`](./post_ft2_progressive_plan_spec_vi_2026-04-17.md)
7. [`model/architecture_v2/model.py`](../../model/architecture_v2/model.py)
8. [`model/architecture_v2/head.py`](../../model/architecture_v2/head.py)

### Nguồn ngoài

1. He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016):  
   [https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
2. Yu et al., *Gradient Surgery for Multi-Task Learning* (OpenReview / ICLR 2020):  
   [https://openreview.net/forum?id=HJewiCVFPB](https://openreview.net/forum?id=HJewiCVFPB)
3. Chen et al., *GradNorm* (PMLR 2018):  
   [https://proceedings.mlr.press/v80/chen18a.html](https://proceedings.mlr.press/v80/chen18a.html)
4. AlphaZero supplementary architecture notes:  
   [https://schachklub.ws/wp-content/uploads/2018/12/alfazero_supplementary_data.pdf](https://schachklub.ws/wp-content/uploads/2018/12/alfazero_supplementary_data.pdf)
5. Lc0 AlphaZero Primer:  
   [https://lczero.org/dev/lc0/search/alphazero/](https://lczero.org/dev/lc0/search/alphazero/)

