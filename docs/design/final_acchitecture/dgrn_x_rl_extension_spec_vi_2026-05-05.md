# Spec RL Extension Cho DGRN-X

Ngày: `2026-05-05`  
Trạng thái: `design proposal`  
Phạm vi: self-play / search-improvement loop sau khi `v0` và `v1` đã ổn  
Phụ thuộc: [`dgrn_x_v1_policy_and_distillation_spec_vi_2026-05-05.md`](./dgrn_x_v1_policy_and_distillation_spec_vi_2026-05-05.md)

---

## 1) Kết luận điều hành

RL không thuộc vòng đầu của DGRN-X.

Repo hiện tại mới chỉ có evidence mạnh về:

- failure của head/objective proxy,
- thiếu encode state,
- broad validation mismatch.

Repo **chưa** có evidence rằng:

- teacher value đã đủ chuẩn để self-play khuếch đại,
- policy labels đã đủ sạch,
- benchmark search-side đã đủ ổn.

Vì vậy RL phải là spec riêng, chỉ được mở sau khi `v0` và `v1` vượt gate.

---

## 2) Vì sao phải tách RL

Nếu value chưa calibrated mà mở RL:

- search sẽ feed back sai lệch đó vào visit distribution,
- error có thể tự khuếch đại thay vì tự sửa.

Nguồn:

- [AlphaZero preprint](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [MuZero overview](https://www.nature.com/articles/s41586-020-03051-4)
- [Lc0 AlphaZero primer](https://lczero.org/dev/lc0/search/alphazero/)

Những hệ này đều ngầm dựa vào một policy/value stack đủ ổn trước khi scale self-play.

---

## 3) Điều kiện mở RL

Chỉ mở RL nếu:

1. `v0` pass broad validation và offline full-test.
2. `v1` pass policy warmup/joint mà không làm hỏng value.
3. Có search screening baseline đáng tin.
4. Có compute budget đủ để tạo self-play data hữu ích, không phải vài nghìn game rời rạc.

Nếu thiếu một trong bốn điều kiện trên, RL bị hoãn.

---

## 4) Mục tiêu RL trong repo này

Mục tiêu RL không phải “bắt đầu từ số 0 như AlphaZero”.

Mục tiêu thực dụng hơn:

- dùng self-play/search improvement để refine policy/value teacher đã được pretrain,
- giữ runtime/search stack tương thích với engine hiện có,
- không thay thế toàn bộ classical engine ngay lập tức.

---

## 5) RL loop tối thiểu

Loop tối thiểu hợp lệ:

1. dùng teacher hiện tại để search/self-play,
2. lưu:
   - state,
   - masked policy target từ visit counts,
   - outcome `z`,
   - metadata search nếu cần,
3. train teacher mới trên replay buffer,
4. benchmark bằng eval spec và search screening,
5. chỉ promote nếu broad + search cùng tốt.

Không mở các biến thể quá tham vọng ở vòng đầu:

- model-based planning,
- dual-network target hierarchy phức tạp,
- mixed reward shaping không chứng minh được.

---

## 6) RL objective

Khi đã có search visit target `π_search` và game outcome `z_game`, objective chuẩn là:

```text
L_RL =
  λ_v * L_value(z_pred, z_game)
  + λ_pi * KL(π_search || π_model)
  + λ_reg * regularization
```

`L_value` ở đây vẫn phải kế thừa contract của `v0`:

- bounded scalar semantics,
- broad eval vẫn đo trên target deployment hiện tại,
- không tự ý đổi target convention trong RL branch.

---

## 7) Replay và data mixing

RL stage phải định nghĩa rõ data mixing:

- `supervised corpus`
- `search-labeled corpus`
- `self-play fresh buffer`

Khuyến nghị:

- không vứt bỏ hoàn toàn supervised/value corpus ngay ở RL phase đầu,
- dùng mixing ratio có schedule,
- log drift giữa supervised distribution và self-play distribution.

---

## 8) Promotion logic cho RL

RL run chỉ được promote nếu:

- không thua teacher trước đó ở broad offline metrics vượt ngưỡng cho phép,
- search screening tốt hơn rõ ràng,
- không tạo ra center amplification regression lớn.

Nói cách khác:

- search thắng nhưng broad value collapse: `không promote`,
- broad value đẹp nhưng search không tiến: `không promote`,
- chỉ promote khi cả hai cùng pass.

---

## 9) Anti-goals

Spec RL này không cho phép:

- dùng RL để “cứu” một teacher chưa pass broad full-test,
- thay broad benchmark bằng Elo nội bộ quá sớm,
- kết luận từ các match nhỏ, noisy, không kiểm soát nodes/time.

---

## 10) Quyết định cuối cùng

RL là phase sau, không phải phần của redesign đầu tiên.

Spec này tồn tại để khóa nguyên tắc:

- `value first`,
- `policy second`,
- `RL last`,
- `promotion requires both offline quality and search quality`.

Đó là cách tránh lặp lại lỗi proxy mismatch ở một tầng phức tạp hơn.

