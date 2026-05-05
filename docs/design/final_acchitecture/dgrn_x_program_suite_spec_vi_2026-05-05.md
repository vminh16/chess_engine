# Bo spec DGRN-X 2026-05-05

Ngay: `2026-05-05`  
Trang thai: `index / controlling overview`  
Pham vi: dieu huong bo spec moi cho DGRN-X va khoa cac contract thiet ke muc cao

---

## 1) Ket luan dieu hanh

Bo spec nay thay the huong DGRN-X cu bang 5 contract tach rieng:

1. encode va data contract,
2. `v0` value teacher,
3. `v1` policy + distillation,
4. RL extension,
5. evaluation / promotion.

Sau khi doi chieu lai artifact Phase B/C1, code local, va nguon nen ben ngoai, bo spec moi khoa 4 quyet dinh:

- `history k=2` la **bat buoc** cho `v0`, khong con la future optional;
- `material_* planes` bi loai khoi encode mac dinh;
- `v0 teacher` dung topology residual CNN chuan, khong dung torso graph/ray tu che o vong dau;
- `v1` chi mo sau khi `v0` da pass broad gate, va distillation phai co temperature + feature contract.

---

## 2) Cac tai lieu nguon chinh

1. Encode  
   [`dgrn_x_encode_refresh_spec_vi_2026-05-05.md`](./dgrn_x_encode_refresh_spec_vi_2026-05-05.md)

2. `v0` Value Teacher  
   [`dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md`](./dgrn_x_v0_value_teacher_spec_vi_2026-05-05.md)

3. `v1` Policy + Distillation Contract  
   [`dgrn_x_v1_policy_and_distillation_spec_vi_2026-05-05.md`](./dgrn_x_v1_policy_and_distillation_spec_vi_2026-05-05.md)

4. RL Extension  
   [`dgrn_x_rl_extension_spec_vi_2026-05-05.md`](./dgrn_x_rl_extension_spec_vi_2026-05-05.md)

5. Evaluation / Promotion  
   [`dgrn_x_evaluation_protocol_spec_vi_2026-05-05.md`](./dgrn_x_evaluation_protocol_spec_vi_2026-05-05.md)

---

## 3) Trinh tu thuc thi

Thu tu dung:

1. cap nhat dataset contract de ho tro history `k=2`,
2. refresh encode theo schema moi,
3. train `v0` value-only teacher,
4. chi neu `v0` pass thi mo `v1`,
5. chi neu `v1` pass thi can nhac RL,
6. eval spec ap dung o moi stage.

Neu dataset hien tai khong du lich su nhat quan de lap input `k=2`, thi DGRN-X khong duoc mo train `v0`. Khi do phai dung va sua dataset truoc.

---

## 4) Nhung gi bo spec moi co y tranh

- khong goi mot encode thieu history la "search-safe",
- khong dua scalar material hand-crafted vao input mac dinh,
- khong mo dong thoi value + policy + RL + distillation trong vong dau,
- khong dung oracle subset lam selector chinh,
- khong dung legal-move scorer ragged lam main policy path o `v1`,
- khong giu spec cu nhu nguon implementation chinh.

---

## 5) Quan he voi spec cu

Hai tai lieu:

- [`dgrn_x_architecture_spec_vi_2026-04-30.md`](./dgrn_x_architecture_spec_vi_2026-04-30.md)
- [`dgrn_x_training_program_spec_vi_2026-04-30.md`](./dgrn_x_training_program_spec_vi_2026-04-30.md)

duoc giu lai nhu `deprecation notes` de bao toan lich su thiet ke, nhung khong con la nguon chinh cho implementation.

Neu noi dung nao trong hai tai lieu cu mau thuan voi bo spec `2026-05-05`, bo spec `2026-05-05` la nguon su that uu tien cao hon.

---

## 6) Nguon nen da doi chieu

- [AlphaZero preprint](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [Lc0 AlphaZero primer](https://lczero.org/dev/lc0/search/alphazero/)
- [Lc0 rule50 encoding bug](https://lczero.org/blog/2018/08/rule50-encoding-bug-is-found/)
- [Representation Matters for Mastering Chess](https://arxiv.org/abs/2304.14918)
- [Distilling the Knowledge in a Neural Network](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/)

Bo spec nay khong copy nguyen xi mot he thong nao, nhung cac contract cot loi duoc chon de giam aliasing, giu topology de kiem soat failure, va giu duong mo rong cho distillation/RL theo cach doc duoc bang artifact.
