# Spec Kiến Trúc DGRN-X

Ngày gốc: `2026-04-30`  
Ngày cập nhật: `2026-05-05`  
Trạng thái: `overview / superseded in detail`

---

## 1) Trạng thái hiện tại

Tài liệu này không còn là nguồn chi tiết chính cho implementation DGRN-X.

Nó được giữ lại để:

- bảo toàn lịch sử thiết kế,
- giải thích vì sao hướng DGRN-X vẫn đúng,
- điều hướng sang bộ spec mới chặt hơn.

Sau khi đọc artifact Phase B/C1 và rà lại phản biện lý thuyết, thiết kế DGRN-X đã được tái cấu trúc thành bộ spec nhỏ hơn, rõ contract hơn.

---

## 2) Kết luận còn hiệu lực từ spec gốc

Các nhận định sau vẫn giữ nguyên:

1. `head-only` không đủ sau Phase B.
2. Encode hiện tại thiếu state có ích như `halfmove` và `phase`.
3. Teacher/runtime/student không nên bị ép thành một kiến trúc duy nhất quá sớm.
4. Policy và RL không nên được mở trước khi value teacher đủ ổn.

---

## 3) Những phần của spec gốc đã bị thay thế

Các phần sau trong bản cũ không còn là guidance chi tiết chính:

- `PieceSummary(T)` rồi broadcast về grid
- piece stream chỉ `1` relational block
- legal-move scorer như main policy path
- scalar + WDL + draw + sigma được mở đồng thời từ vòng đầu

Những phần này đã được sửa hoặc tách pha trong bộ spec mới.

---

## 4) Bộ spec thay thế

Nguồn chi tiết chính hiện tại là:

1. Tổng quan suite  
   [`dgrn_x_program_suite_spec_vi_2026-05-05.md`](./dgrn_x_program_suite_spec_vi_2026-05-05.md)

2. Encode  
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

## 5) Cách đọc đúng

Nếu mục tiêu là implementation mới:

- bắt đầu ở `program suite`,
- sau đó đọc `encode`,
- rồi `v0`,
- chỉ đọc `v1/RL` khi `v0` đã pass.

Tài liệu này chỉ còn vai trò overview lịch sử và bối cảnh.

