# Spec Training Cho DGRN-X

Ngày gốc: `2026-04-30`  
Ngày cập nhật: `2026-05-05`  
Trạng thái: `overview / superseded in detail`

---

## 1) Trạng thái hiện tại

Tài liệu này không còn là spec training chi tiết duy nhất cho DGRN-X.

Nó được cập nhật thành một bản overview, vì thiết kế training hiện tại đã được tách thành nhiều contract độc lập hơn:

- encode,
- value-only `v0`,
- policy/distillation `v1`,
- RL riêng,
- evaluation riêng.

---

## 2) Những nguyên tắc từ bản cũ vẫn còn đúng

1. `Value first`: chưa mở policy/RL trước khi value ổn.
2. `Test frozen`: test set không dùng để chọn checkpoint.
3. `Broad validation`: selector chính phải bám deployment-facing metrics.
4. `Oracle subset secondary`: không dùng bundle hẹp làm promotion gate chính.

---

## 3) Những phần đã được thay thế

Các phần sau của bản cũ không còn được dùng như guidance chi tiết chính:

- calibration bucket loss batch-wise như thành phần mặc định của objective,
- center hinge cứng như regularizer trung tâm mặc định,
- mô tả policy/teacher/student/RL trong cùng một flow liền khối,
- assumption rằng value + WDL + draw + sigma nên mở sớm ở vòng đầu.

Những phần này đã được chia nhỏ và làm chặt hơn trong bộ spec mới.

---

## 4) Bộ spec training mới cần dùng

1. Tổng quan suite  
   [`dgrn_x_program_suite_spec_vi_2026-05-05.md`](./dgrn_x_program_suite_spec_vi_2026-05-05.md)

2. Encode / data contract  
   [`dgrn_x_encode_refresh_spec_vi_2026-05-05.md`](./dgrn_x_encode_refresh_spec_vi_2026-05-05.md)

3. `v0` Value Teacher  
   [`dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md`](./dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md)

4. `v1` Policy + Distillation  
   [`dgrn_x_v1_policy_and_distillation_spec_vi_2026-05-05.md`](./dgrn_x_v1_policy_and_distillation_spec_vi_2026-05-05.md)

5. RL Extension  
   [`dgrn_x_rl_extension_spec_vi_2026-05-05.md`](./dgrn_x_rl_extension_spec_vi_2026-05-05.md)

6. Evaluation / Promotion  
   [`dgrn_x_evaluation_protocol_spec_vi_2026-05-05.md`](./dgrn_x_evaluation_protocol_spec_vi_2026-05-05.md)

---

## 5) Trình tự đúng

Training program hiện tại phải được hiểu theo thứ tự:

1. cập nhật encode và version data,
2. chạy `v0` value-only teacher,
3. nếu `v0` pass thì mở `v1`,
4. nếu `v1` pass thì mới cân nhắc RL,
5. eval spec áp dụng ở mọi stage.

---

## 6) Quyết định cuối cùng

Tài liệu này chỉ còn là overview của training program.

Chi tiết implementation mới phải đọc từ bộ spec `2026-05-05`, không dùng lại trực tiếp bản cũ `2026-04-30` như nguồn đơn lẻ.

