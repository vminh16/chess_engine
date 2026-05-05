# Spec Encode Cho DGRN-X

Ngay: `2026-05-05`  
Trang thai: `design proposal`  
Pham vi: input schema, history contract, dataset contract, cache/versioning cho dong DGRN-X  
Quan he: moi spec `v0/v1/RL/eval` deu phu thuoc vao tai lieu nay

---

## 1) Ket luan dieu hanh

Encode hien tai cua repo khong du cho mot value teacher dung cho screening/search:

- [`representation/encode.py`](../../representation/encode.py) hien chi tao `18 x 8 x 8` planes,
- [`core/board.py`](../../core/board.py) co `halfmove_clock` va `fullmove_number` nhung encode bo qua,
- [`data/process_data.ipynb`](../../data/process_data.ipynb) hien mo ta label theo `STM`, nhung khong chot contract lich su cho input.

Sau khi doi chieu local artifact va ly thuyet nen, encode moi chot 4 quyet dinh:

1. `rule50` la plane bat buoc.
2. `phase` la plane bat buoc.
3. `history k=2` la **bat buoc cho DGRN-X-v0**. Neu dataset khong tai tao duoc `k=2`, `v0` khong duoc train.
4. `material_self`, `material_opp`, `material_delta` **khong nam trong schema mac dinh**.

Schema mac dinh cua `DGRN-X-v0` la:

```text
58 x 8 x 8
= 20 plane hien tai
+ 19 plane history[-1]
+ 19 plane history[-2]
```

Trong do:

- current frame = `18 base + rule50 + phase`
- moi historical frame = `18 base + history_present_mask`

---

## 2) Dieu gi duoc chung minh boi local artifact, dieu gi la quyet dinh thiet ke

### 2.1 Da duoc chung minh boi local artifact

1. Encode hien tai dang thieu state co san trong `Board`  
   Nguon:
   - [`representation/encode.py`](../../representation/encode.py)
   - [`core/board.py`](../../core/board.py)

2. Objective-only khong du de sua broad fit  
   Nguon:
   - [`runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/selected_checkpoint_eval.json`](../../runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/selected_checkpoint_eval.json)
   - [`runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/decision_summary.json`](../../runs/dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1/reports/decision_summary.json)

   C1 sua center rat manh, nhung van thua L4 o `overall_mse`, `overall_pearson`, va `test_max_midband_abs_cal_gap`.

3. Pipeline label hien tai la `STM-relative`, khong phai white-POV  
   Nguon:
   - [`data/process_data.ipynb`](../../data/process_data.ipynb)

### 2.2 Chua duoc chung minh bang artifact, nhung phai khoa tu dau vi ly do deploy

1. History `k=2` khong duoc artifact local hien tai "chung minh" la root cause duy nhat.
2. Tuy nhien, neu muc tieu cua teacher la phuc vu screening/search, thi mot encode khong du lich su de phan biet repetition-related states la contract khong chap nhan duoc.

Noi ngan gon:

- `history mandatory` o day la **quyet dinh an toan cap thiet ke**,
- khong phai ket luan nhan-qua da duoc thuc nghiem local chung minh xong.

---

## 3) Co so toan hoc va ly thuyet

### 3.1 Aliasing tao irreducible variance

Neu encode la anh xa:

```text
x = phi(s)
```

va ton tai hai trang thai hop le `s1, s2` sao cho:

```text
phi(s1) = phi(s2)
```

nhung target deployment-relevant khac nhau:

```text
y(s1) != y(s2)
```

thi du doan toi uu theo MSE chi hoc duoc:

```text
f*(x) = E[Y | X = x]
```

voi loi khong the khu:

```text
E[(Y - f*(X))^2 | X = x] = Var(Y | X = x)
```

Do do, neu repetition-safe state bi alias voi non-repetition state, optimization khong the sua phan variance do representation gay ra.

### 3.2 History la contract can cho game-state input kieu AlphaZero

Ba nguon nen dong quy vao cung mot diem:

- AlphaZero dung stack lich su trong input board planes,
- Lc0 xac nhan rule50/encoding bug gay Elo regression thuc,
- nghien cuu `Representation Matters` cho thay cai tien representation co the mang lai loi ich lon hon doi topology sang transformer.

Nguon:

- [AlphaZero preprint](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [Lc0 AlphaZero primer](https://lczero.org/dev/lc0/search/alphazero/)
- [Lc0 rule50 encoding bug](https://lczero.org/blog/2018/08/rule50-encoding-bug-is-found/)
- [Representation Matters for Mastering Chess](https://arxiv.org/abs/2304.14918)

### 3.3 Vi sao bo `material_*` khoi schema mac dinh

Day khong phai dinh ly "material planes chac chan co hai".  
Ket luan chinh xac hon la:

- local artifact hien tai **khong** chung minh material planes la can thiet,
- material planes nhung vao input mot tong diem tuyen tinh co dinh,
- trong khi muc tieu cua `v0` la cho mang tu hoc gia tri dong tren raw board state + lich su + rule state.

Vi vay `material_*` bi loai khoi mac dinh de:

- tranh co dinh hoa mot prior tuyen tinh thu cong,
- giu failure de doc hon,
- giam rui ro model dua vao shortcut scalar thay vi hoc tuong tac spatial/temporal.

Day la quyet dinh thiet ke bao thu, khong phai phu dinh vinh vien moi material feature.

---

## 4) Nguyen tac encode cua DGRN-X

Encode moi phai thoa 6 yeu cau:

1. Giu target convention `STM-relative`.
2. Dua them state anh huong truc tiep den drawishness/no-progress.
3. Ho tro so sanh lich su toi thieu de giam repetition blindness.
4. Khong dua prior material tuyen tinh thu cong vao default schema.
5. Co versioning va cache invalidation ro rang.
6. Co test xac minh perspective/history semantics truoc khi train that.

---

## 5) Root-side perspective contract

`DGRN-X` su dung `root-side perspective` cho toan bo stack input.

Dinh nghia:

- `root side` = ben duoc danh gia o current position; trong repo hien tai, day chinh la `current side to move`.
- current frame duoc encode theo root-side perspective.
- moi history frame cung duoc re-project ve cung root-side perspective, khong dao qua lai theo side-to-move cua tung frame lich su.

Ly do:

- giu toan bo stack nam trong cung mot he toa do khong gian,
- cho phep conv trunk so sanh current/historical occupancy mot cach nhat quan,
- van giu thong tin temporal bang `side_to_move_plane` cua tung frame.

---

## 6) Exact schema cho `DGRN-X-v0`

### 6.1 Current frame: `20 x 8 x 8`

`18` plane co so:

1. `0..5`: quan cua root side (`P, N, B, R, Q, K`)
2. `6..11`: quan cua ben con lai
3. `12`: `side_to_move_plane` cua frame
   - `1` neu den luot root side o frame do
   - `0` neu den luot doi thu
4. `13`: root side con quyen castle kingside
5. `14`: root side con quyen castle queenside
6. `15`: doi thu con quyen castle kingside
7. `16`: doi thu con quyen castle queenside
8. `17`: en-passant target square cua frame, bieu dien trong root orientation

`2` plane state bo sung:

9. `18`: `rule50_plane`

```text
r50 = min(1, halfmove_clock / 100)
```

10. `19`: `phase_plane`

```text
phase_raw = n_N + n_B + 2*n_R + 4*n_Q
phase_norm = min(1, phase_raw / 20)
```

### 6.2 History frames: `2 x (19 x 8 x 8)`

Mac dinh dung `k=2`:

- `history[-1]`: 1 half-move truoc
- `history[-2]`: 2 half-moves truoc

Moi frame history co:

- `18` plane co so giong muc `6.1`, deu duoc re-project ve root-side perspective,
- `1` plane `history_present_mask`
  - `1` neu frame lich su ton tai thuc su,
  - `0` neu vi o opening/bi thieu chain nen khong co.

History frame **khong** mang them `rule50` hay `phase` trong `v0`.
Ly do:

- muc tieu chinh cua history o `v0` la giam aliasing trang thai,
- giu kenh lich su gon va so sanh duoc voi current frame,
- tranh nhan doi qua nhieu scalar planes ma chua co bang chung local can.

### 6.3 Tong so kenh

```text
20 + 19 + 19 = 58
```

Shape chuan:

```text
(58, 8, 8)
```

---

## 7) Dataset contract bat buoc

### 7.1 Dieu kien toi thieu de duoc train `v0`

Dataset builder phai cung cap du mot trong hai kha nang sau:

1. truoc moi sample co the truy ra chinh xac `2` half-moves truoc do tu game record goc; hoac
2. sample record san hai trang thai lich su can thiet trong metadata.

Chi co FEN hien tai la **khong du**.

### 7.2 Khong duoc reconstruct history tu split key canonical

Trong [`data/process_data.ipynb`](../../data/process_data.ipynb), split key hien tai co chu y bo `halfmove/fullmove` khoi canonical key de tranh leakage. Dieu nay hop ly cho split logic, nhung khong du cho history-aware encode.

Do do:

- split key logic co the giu nguyen cho muc dich chia tap,
- nhung dataset train cho `DGRN-X-v0` phai duoc build tu source sequential data truoc khi canonical key cat thong tin lich su.

### 7.3 Khong duoc suy luan repetition tu `halfmove_clock`

`halfmove_clock` chi la no-progress counter. No khong xac dinh duoc:

- repetition count,
- exact previous occupancy,
- castling/en-passant equivalence cua cac state truoc.

Vi vay:

- `rule50` va `history` la hai kenh khac nhau,
- `rule50` khong duoc xem la thay the cho history.

---

## 8) Chinh sach khong dua `material_*` vao default

`material_self`, `material_opp`, `material_delta` khong nam trong schema mac dinh cua `v0`.

Neu sau nay muon thu ablation material planes, phai:

1. tao schema version rieng,
2. benchmark doc lap,
3. chung minh broad gain khong phai do shortcut o center/midband.

Khong duoc them material plane vao `v0 core` roi van goi do la baseline chuan.

---

## 9) Cache, manifest, versioning

Moi dataset/encode cache cho `DGRN-X` phai khoa it nhat cac truong:

- `encode_schema_version`
- `history_depth`
- `root_perspective_contract`
- `rule50_enabled`
- `phase_enabled`
- `material_planes_enabled`
- `source_dataset_version`

Version de xuat cho schema nay:

```text
dgrn_x_encode_v0_hist2_rule50_phase_rootstm
```

Neu bat ky truong nao thay doi, cache cu khong duoc tai su dung.

---

## 10) Validation bat buoc truoc khi train that

Truoc khi mo run `v0`, phai co sanity suite cho encode:

1. `shape test`  
   Output dung `(58, 8, 8)`.

2. `perspective test`  
   Vi tri quan trong current va history khi re-project sang root orientation khop voi expectation.

3. `history-present test`  
   Opening sample thieu history phai co `history_present_mask = 0`.

4. `repetition-equivalence test`  
   Neu current position lap lai mot history position theo board+rights+ep+turn, tensor phan co so cua hai frame phai giong nhau.

5. `rule50 monotonicity test`  
   `rule50_plane` tang dung theo `halfmove_clock`.

6. `phase sanity test`  
   Opening > middlegame > endgame theo quy uoc phase duoc kiem tra tren bo vi du co dinh.

7. `cache key test`  
   Doi `history_depth` hoac `material_planes_enabled` phai doi manifest key.

---

## 11) Quy trinh neu dataset hien tai chua du lich su

Neu trong qua trinh implementation xac minh rang dataset hien tai khong truy vet duoc `k=2` history chain mot cach on dinh, hanh dong dung la:

1. dung DGRN-X `v0`,
2. sua dataset builder/source manifest,
3. tao dataset version moi,
4. chi sau do moi train teacher.

Khong duoc:

- ha schema xuong `20 x 8 x 8` roi van goi do la `v0`,
- dua history ve `v1+`,
- thay history bang scalar heuristics.

---

## 12) Quyet dinh cuoi cung

Encode cua DGRN-X khong con la mot patch nho tren `18-plane` cu.

No la mot dataset contract moi:

- `root-side perspective`,
- `rule50`,
- `phase`,
- `history k=2 mandatory`,
- `no material planes by default`,
- `strict cache/version discipline`.

Day la muc sua toi thieu de `v0` co co hoi hoc mot value teacher co the tong quat hoa tot hon cho search-facing deployment.
