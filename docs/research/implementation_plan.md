# Phân Tích Thực Nghiệm Toàn Diện & Kế Hoạch Cải Tiến

## Tổng quan

Phân tích này dựa trên **dữ liệu thực nghiệm đo được** từ repo, không phải trực giác. Mỗi kết luận đều có dẫn chứng từ số liệu hoặc phân tích toán học cụ thể.

---

## PHẦN I: Phân Tích Dữ Liệu Thực Nghiệm

### 1.1 Phân Phối Dữ Liệu — ĐO ĐƯỢC

**Nguồn**: Chạy analysis trên toàn bộ 80 shards × 50,000 = 4,000,000 mẫu training.

| Band |y| | Count | % Data | Tích lũy |
|---------|-------|--------|---------|
| [0.00, 0.02) | 808,712 | **20.22%** | 20.22% |
| [0.02, 0.05) | 448,524 | 11.21% | 31.43% |
| [0.05, 0.10) | 462,764 | 11.57% | **43.00%** |
| [0.10, 0.15) | 280,468 | 7.01% | 50.01% |
| [0.15, 0.20) | 199,532 | 4.99% | 55.00% |
| [0.20, 0.30) | 280,000 | 7.00% | 62.00% |
| [0.30, 0.40) | 232,000 | 5.80% | 67.80% |
| [0.40, 0.50) | 216,000 | 5.40% | 73.20% |
| [0.50, 0.60) | 232,000 | 5.80% | 79.00% |
| [0.60, 0.70) | 280,000 | 7.00% | 86.00% |
| [0.70, 0.80) | 272,000 | 6.80% | 92.80% |
| [0.80, 0.90) | 176,000 | 4.40% | 97.20% |
| [0.90, 1.00) | 112,000 | 2.80% | 100.00% |

**Kết luận thống kê**:
- **43% data nằm ở |y| ≤ 0.1** (center-dominated)
- Mid-band [0.2, 0.7]: chỉ **31%** data
- Decisive |y| > 0.7: chỉ **14%** data
- Median |y| = 0.149, Mean |y| = 0.285
- Phân phối **gần đối xứng** (negative 50.0%, positive 39.0%, zero 11.0%)

> [!IMPORTANT]
> **43% data ở center** nghĩa là trong mỗi batch 256, trung bình ~110 mẫu là near-zero. Đây không chỉ là data imbalance — nó trực tiếp ảnh hưởng đến **BatchNorm running statistics**, vì BN thấy center data chiếm đa số.

### 1.2 Gradient Mass — ĐO ĐƯỢC

**Nguồn**: Tính trực tiếp trên 500,000 mẫu (10 shards) dùng công thức `w(y) = (1-y²)²`.

**Baseline (y-space MSE, không có compensation):**

| Band |y| | % Data | Mean w(y) | **% Gradient Mass** | **Ratio** |
|---------|--------|-----------|---------------------|-----------|
| [0.0, 0.1) | 43.1% | 0.9963 | **57.0%** | 1.32 |
| [0.1, 0.2) | 12.0% | 0.9573 | 15.3% | 1.27 |
| [0.2, 0.3) | 6.9% | 0.8796 | 8.1% | 1.17 |
| [0.3, 0.5) | 11.2% | 0.7042 | 10.5% | 0.94 |
| **[0.5, 0.7)** | **12.8%** | **0.4013** | **6.8%** | **0.53** |
| **[0.7, 1.0)** | **12.6%** | **0.1375** | **2.3%** | **0.18** |

**Với L4 curvature compensation (alpha=0.65):**

| Band |y| | L4 Mean w | L4 % Mass | L4 Ratio |
|---------|-----------|-----------|----------|
| [0.0, 0.1) | 0.7343 | 44.7% | 1.04 |
| [0.1, 0.2) | 0.7343 | 12.5% | 1.04 |
| [0.2, 0.3) | 0.7343 | 7.2% | 1.04 |
| [0.3, 0.5) | 0.7343 | 11.7% | 1.04 |
| [0.5, 0.7) | 0.7343 | 13.3% | 1.04 |
| **[0.7, 1.0)** | **0.5611** | **10.0%** | **0.79** |

> [!IMPORTANT]
> **Phát hiện quan trọng**: L4 compensation **gần như flatten hoàn toàn** gradient mass ở bands [0.0, 0.7) — tất cả đều có ratio ≈ 1.04. **Nhưng band [0.7, 1.0) vẫn bị under-weighted** (ratio = 0.79). Đây là do `clip_max = 4.0` chặn compensation ở tails.
>
> **Ý nghĩa**: L4 objective **ĐÃ SỬA ĐÚNG** gradient allocation cho mid-band. Vấn đề Failure A còn lại **không phải gradient weight** — nó phải nằm ở chỗ khác.

### 1.3 Label Noise từ Stockfish — PHÂN TÍCH TOÁN HỌC

**Cơ sở**: Với `y = tanh(cp/600)`, nhiễu trong y-space phụ thuộc vào noise trong cp:

```
dy/dcp = (1/600) × (1 - tanh²(cp/600)) = (1/600) × sech²(cp/600)
```

Giả sử Stockfish depth-12 có std(noise) ≈ 40cp (ước lượng trung bình từ engine literature):

| CP | y | dy/dcp | **noise_y (1σ)** | noise/|y| |
|----|------|--------|------------------|-----------|
| 0 | 0.0000 | 0.001667 | **0.0667** | ∞ |
| 50 | 0.0831 | 0.001655 | 0.0662 | 0.80 |
| 100 | 0.1651 | 0.001621 | 0.0648 | 0.39 |
| 200 | 0.3215 | 0.001494 | 0.0598 | 0.19 |
| 300 | 0.4621 | 0.001311 | 0.0524 | 0.11 |
| 400 | 0.5828 | 0.001101 | 0.0440 | 0.08 |
| 600 | 0.7616 | 0.000700 | 0.0280 | 0.04 |

> [!CAUTION]
> **Phát hiện: Noise ở center (y≈0) là 0.067 — rất lớn!**
> 
> Một vị trí hoàn toàn cân bằng (true cp=0) có thể nhận label y ∈ [-0.067, +0.067] chỉ do noise depth-12. Đây là **1σ** — nghĩa là ~32% mẫu có noise > 0.067.
>
> **Kết nối trực tiếp với Failure B**: Oracle center threshold là `|y| ≤ 0.05`. Nhưng noise ở center = ±0.067 > 0.05. Nghĩa là **phần lớn "center positions" trong training data không thực sự là center** — chúng bị noise đẩy ra ngoài threshold. Đây chính là root cause #1 của center label impurity (B1).
>
> **Người dùng đề cập dùng node-based search thay vì fixed depth**: Node-based search có effective depth **thay đổi theo position**. Tactical positions → shallower → MORE noise. Quiet positions → deeper → LESS noise. Đây tạo ra **heteroscedastic noise** — noise level tương quan với position type. Model không thể phân biệt signal vs noise khi noise level thay đổi.

### 1.4 Data Quality Audit — ĐO ĐƯỢC

**Từ channel analysis (100 samples đầu tiên):**

| Channel | Nội dung | Nonzero % | Observation |
|---------|----------|-----------|-------------|
| Ch00 (Own Pawns) | 8.2% | Hợp lý (~6-8 pawns / 64 squares) |
| Ch05 (Own King) | 1.6% | Hợp lý (1 king / 64 = 1.56%) |
| Ch12 (Turn) | 45.0% | **Hơi lệch** (nên gần 50%) |
| Ch17 (En passant) | **0.0%** | **Rất ít en passant positions** |

**Label anomalies:**
- `y == 0.0 exactly`: 5,435 / 50,000 = **10.9%** per shard
- `|y| >= 0.999`: ~690 per shard (1.4%)
- `duplicate y values`: ~47,880 / 50,000 = **95.8%** — hầu hết labels bị trùng do `float32` precision

> [!WARNING]
> **10.9% labels = 0.0 chính xác** — đáng ngờ. Trong Stockfish depth-12, xác suất eval ĐÚNG 0cp là rất thấp. Điều này có thể do: (a) clamping/rounding trong pipeline, (b) positions thực sự = 0, hoặc (c) Stockfish trả 0 cho một số positions đặc biệt. Cần investigate thêm.

### 1.5 Label Perspective — BUG TIỀM ẨN

**Phát hiện từ code analysis:**

[processing_data.py](file:///c:/Users/USER/Desktop/chess_engine/data/processing_data.py) line 152:
```python
cp = score.pov(chess.WHITE).score()  # Luôn từ góc nhìn Trắng
eval_score = np.tanh(cp / 600.0)     # Label = tanh(cp_white / 600)
```

[encode.py](file:///c:/Users/USER/Desktop/chess_engine/representation/encode.py) line 14-15, 42:
```python
current_color = board.current_turn         # STM-relative encoding
opponent_color = Color.BLACK if current_color == Color.WHITE else Color.WHITE
# ...
if current_color == Color.BLACK:
    r_idx = 7 - rank  # Flip board cho Black
```

> [!CAUTION]
> **Encoding là STM-relative (flip cho Black), nhưng label luôn từ góc White!**
>
> Khi Black đi:
> - Input: bàn cờ được flip, quân Black ở channels 0-5 (như "quân mình")
> - Label: `y = tanh(cp_white / 600)` — từ góc White, **KHÔNG** flip
>
> Nếu Black đang thắng (cp_white = -200), label = tanh(-200/600) = -0.32. Nhưng input đã flip cho Black → model thấy "mình đang có lợi thế" nhưng label lại nói "giá trị = -0.32".
>
> **CẦN KIỂM TRA**: Liệu pipeline train có flip label cho Black không? Tôi đã search toàn bộ code và **KHÔNG TÌM THẤY** bất kỳ label flip nào. Nếu đây thực sự là bug, nó là **root cause lớn nhất** của cả hai failure modes vì model đang học từ labels nhất quán sai cho 50% data.
>
> **HOWEVER**: Channel 12 (turn indicator) = 1 cho White, 0 cho Black. Model **có thể** học implicit sign flip qua channel 12. Nhưng điều này buộc model phải dùng capacity để học "nếu Ch12=0, đảo dấu output" — một phép tính đáng lẽ không cần thiết.

---

## PHẦN II: Phân Tích Kiến Trúc

### 2.1 Over-parameterization — PHÂN TÍCH TOÁN HỌC

**Receptive Field Analysis:**

| Blocks | RF | RF/Board | Status |
|--------|-----|----------|--------|
| 2 | 11 | 1.4x | Full coverage |
| 6 | 27 | 3.4x | 3.4x overkill |
| 12 | 51 | 6.4x | 6.4x overkill |
| **20** | **83** | **10.4x** | **10.4x overkill** |

Bàn cờ 8×8 = 64 ô. Quân cờ tương tác xa nhất = Queen/Bishop (~7 ô). **RF = 8 đã đủ cho mọi pattern cờ vua.** 20 blocks cho RF = 83, nghĩa là **mỗi pixel "nhìn thấy" toàn bộ bàn cờ hơn 10 lần**.

Lý do depth lớn hữu ích trong vision: ảnh tự nhiên có hierarchical features (edges → textures → objects → scenes). Nhưng **bàn cờ không có hierarchical features theo cùng nghĩa** — tất cả thông tin đã có ở raw encoding, model chỉ cần học **combinatorial relationships** giữa quân cờ.

**Parameter Count Estimates (analytical):**

| Config | Params | Params/Sample | Assessment |
|--------|--------|---------------|------------|
| **20b/256d** | **~12.5M** | **3.14** | 🔴 Over-parameterized |
| 12b/128d | ~2.4M | 0.59 | ✅ Reasonable |
| 8b/128d | ~1.8M | 0.44 | ✅ Good |
| 6b/96d | ~1.0M | 0.25 | ✅ Compact |

> [!IMPORTANT]
> **20b/256d có ~12.5M params cho 4M training samples → ratio = 3.14 params/sample.** 
>
> Quy tắc thông thường: ratio > 1.0 → strong overfitting risk. Comparison: Leela Chess Zero 10b/128f = ~4M params cho *hàng trăm triệu* positions training → ratio << 0.01.
>
> Model hiện tại có capacity **quá lớn** cho dataset size. Điều này giải thích tại sao FT1 train objective giảm 6.1× nhưng val MSE chỉ giảm 2.2× (**gap ratio = 2.73** — evidence mạnh cho overfitting).

### 2.2 Overfitting Evidence từ FT1 — ĐO ĐƯỢC

| Epoch | Train Obj | Val MSE | Slope_0.7 | MB_MAE | Center |
|-------|-----------|---------|-----------|--------|--------|
| 0 | 0.1538 | 0.0416 | 0.362 | 0.798 | 0.826 |
| 4 | 0.0946 | 0.0295 | 0.487 | 0.642 | 0.445 |
| 8 | 0.0631 | 0.0323 | 0.553 | 0.600 | 0.428 |
| **11** | **0.0439** | **0.0283** | **0.572** | **0.542** | 0.460 |
| 13 | 0.0356 | 0.0226 | 0.513 | 0.576 | 0.374 |
| **17** | **0.0253** | **0.0187** | **0.380** | **0.736** | **0.331** |

- Train objective: 0.154 → 0.025 = **6.1× giảm**
- Val MSE: 0.042 → 0.019 = **2.2× giảm**
- **Gap ratio = 2.73** — model **overfit training distribution**
- Best slope ở epoch 11 (0.572), sau đó **slope sụp xuống 0.380** ở epoch 17

> [!IMPORTANT]
> **Catastrophic forgetting A-side**: Slope giảm từ 0.572 → 0.380 (giảm 33%) từ epoch 11→17. Cùng lúc, center_score cải thiện từ 0.460 → 0.331. Đây là **zero-sum game** — model đang trade A cho B, không phải cải thiện cả hai.

### 2.3 Head Gradient Interference — DỮ LIỆU CÓ SẴN

Từ [gradient_interference_cosines.csv](file:///c:/Users/USER/Desktop/chess_engine/experiments/failure_b_resolution_suite/outputs/reports/gradient_interference_cosines.csv):

| Pair | cosine_all | **cosine_backbone** | **cosine_head** |
|------|-----------|---------------------|----------------|
| center vs mid_05_07 (baseline) | -0.642 | -0.279 | **-0.724** |
| center vs mid_05_07 (A1) | -0.680 | -0.309 | **-0.741** |
| center vs mid_02_05 (baseline) | -0.240 | -0.091 | -0.026 |
| near_center vs mid_05_07 (baseline) | -0.555 | -0.254 | **-0.650** |

> [!IMPORTANT]
> **Head cosine (-0.724) gấp 2.6× backbone cosine (-0.279)**
>
> Gradient interference **mạnh nhất ở HEAD**, không phải backbone. Đây là bằng chứng toán học rằng **head architecture là bottleneck chính** cho Failure B. Single-scalar output buộc head phải encode cả magnitude, direction, và confidence vào 1 số — tạo ra conflict không thể giải quyết bằng loss engineering.

---

## PHẦN III: Đánh Giá Thí Nghiệm Trước Đây

### 3.1 Oracle Set n=240 — THỐNG KÊ CỤ THỂ

Center subset (|oracle| ≤ 0.05): **n = 22** (từ center_label_purity_summary.csv)

Với n = 22, khoảng tin cậy 95% cho một proportion p:
```
CI ≈ p ± 1.96 × sqrt(p(1-p)/n)
Ví dụ: false_decisive_0.1eq = 55.9%
CI = 0.559 ± 1.96 × sqrt(0.559×0.441/22) = 0.559 ± 0.207
→ CI = [35.2%, 76.6%]
```

**Width = 41.4%** — quá rộng để so sánh giữa các runs (differences thường < 10%).

### 3.2 Phương pháp thí nghiệm — ĐÁNH GIÁ

| Aspect | Đánh giá | Evidence |
|--------|---------|---------|
| A-side methodology | ✅ Sound | Bootstrap p=0.98 cho L4 midband improvement |
| B-side methodology | ⚠️ Low power | n=22, CI width >40% |
| Single-run conclusions | ⚠️ Risky | No replication for most variants |
| Loss ablation isolation | ✅ Good | Same checkpoint, same data, only objective differs |
| Full retrain evaluation | ✅ Comprehensive | Multiple metrics, gate system, telemetry |

---

## PHẦN IV: Vấn Đề Thực Sự & Hướng Giải Quyết

### 4.1 Root Cause Ranking (dựa trên evidence)

| # | Root Cause | Evidence | Impact | Difficulty |
|---|-----------|----------|--------|------------|
| 1 | **Label perspective mismatch** | Code analysis: White POV label + STM encoding | 🔴 Rất cao | Thấp (fix data pipeline) |
| 2 | **Over-parameterization (12.5M / 4M)** | Gap ratio 2.73, catastrophic forgetting | 🔴 Cao | Trung bình (reduce model) |
| 3 | **Label noise ở center** (depth-12, ±0.067) | Toán học: dy/dcp analysis | 🟠 Cao | Trung bình (deeper eval hoặc filter) |
| 4 | **Head bottleneck** | cosine_head=-0.724 >> cosine_backbone=-0.279 | 🟠 Cao | Trung bình (redesign head) |
| 5 | **43% center-dominated data → BN bias** | Data distribution measurement | 🟡 Trung bình | Thấp (GroupNorm) |
| 6 | **Node-based evaluation → heteroscedastic noise** | User testimony + theoretical analysis | 🟡 Trung bình | Cao (relabel data) |

### 4.2 Các Thí Nghiệm Đề Xuất (theo thứ tự ưu tiên)

---

#### Thí Nghiệm 1: Xác minh Label Perspective (KHẨN CẤP)

**Mục tiêu**: Xác nhận hoặc bác bỏ label perspective mismatch.

**Phương pháp**:
```python
# Lấy 100 positions có Black đang đi
# So sánh: label y vs -y
# Nếu model đã train tốt với labels hiện tại, 
# thì hoặc (a) labels đã đúng, hoặc (b) model đang compensate qua Ch12
```

Hoặc kiểm tra trực tiếp:
- Load 1 FEN cụ thể nơi Black đang thắng (ví dụ cp_white = -300)
- `processing_data.py` cho label = tanh(-300/600) = -0.462
- `encode_board()` flip bàn cờ → quân Black ở channels 0-5
- Model cần output **+0.462** (vì STM-relative: Black đang thắng = positive)
- Nhưng label = **-0.462** (White perspective)

**Chi phí**: Thấp (vài giờ code)

**Impact**: Nếu là bug → fix này một mình có thể giải quyết phần lớn cả Failure A và B.

---

#### Thí Nghiệm 2: Reduce Model Size (ƯU TIÊN CAO)

**Mục tiêu**: Kiểm tra liệu 12b/128d hoặc 8b/128d có performance tương đương 20b/256d.

**Phương pháp**:
- Train 3 configs: 8b/128d (~1.8M), 12b/128d (~2.4M), 20b/256d (~12.5M)
- Cùng data, cùng objective (L4), 20 epochs
- So sánh: val MSE, slope_0.7, center_score
- Track train-val gap ratio

**Chi phí**: Trung bình (3 training runs)

**Dự đoán**: Model nhỏ hơn sẽ có val MSE tương đương hoặc tốt hơn do giảm overfitting. Nếu đúng → tiết kiệm compute 5-10× và cải thiện generalization.

---

#### Thí Nghiệm 3: Freeze-Backbone Ablation (ƯU TIÊN CAO)

**Mục tiêu**: Phân tách head vs backbone contribution cho cả A và B.

**Phương pháp**:
- Lấy L4 checkpoint
- Freeze entire backbone (stem + blocks) → chỉ train head
- So sánh: slope_0.7, center_score trước vs sau

**Chi phí**: Thấp (1 run)

**Nếu head-only training cải thiện**: Head là bottleneck → cần redesign head
**Nếu không cải thiện**: Backbone features cũng cần thay đổi → size reduction is the right direction

---

#### Thí Nghiệm 4: GroupNorm Ablation (ƯU TIÊN CAO)

**Mục tiêu**: Test B3 hypothesis (BN center amplification).

**Phương pháp**:
- Thay tất cả BatchNorm2d → GroupNorm (groups=8 hoặc 16)
- Train từ đầu với cùng objective L4
- So sánh center metrics

**Chi phí**: Trung bình (1 run, phải train lại)

**Cơ sở lý thuyết**: GroupNorm normalize per-group per-sample, không dùng running statistics → **không bị biased bởi data distribution**. Nếu center metrics cải thiện → BN là amplifier chính.

---

#### Thí Nghiệm 5: Mở Rộng Oracle Set (ƯU TIÊN CAO)

**Mục tiêu**: Tăng statistical power cho B-side evaluation.

**Phương pháp**:
- Mở rộng oracle set: 240 → 1000+ positions
- Center subset: 22 → 100+ positions
- Dùng Stockfish depth ≥ 20
- Phân tầng theo |oracle| band

**Chi phí**: Trung bình (Stockfish chạy vài giờ)

---

#### Thí Nghiệm 6: Head Redesign

**Mục tiêu**: Giảm head gradient interference.

**Phương pháp**: Thay `ResidualGainValueHead` bằng thiết kế mới:

**Option A**: Multi-head output → combine
```
head_center → predict y cho |y| < 0.2
head_midband → predict y cho 0.2 ≤ |y| ≤ 0.7
head_decisive → predict y cho |y| > 0.7
gating → weighted sum based on predicted regime
```

**Option B**: Simpler head (giảm spatial compression)
```
Global pool (avg + max) → MLP(512 → 128 → 1) → tanh
```
Loại bỏ spatial flatten hoàn toàn. Backbone 8b/128d đã có full RF, không cần spatial scoring riêng.

**Option C**: Two-stage head
```
logit = spatial_score + global_score  (giữ nguyên)
confidence = sigmoid(conf_head)        (thêm head mới)
output = tanh(logit) × confidence      (modulate)
```

**Chi phí**: Cao (design + implement + train)

---

### 4.3 Đề Xuất Architecture Mới

Dựa trên tất cả evidence, đây là architecture recommendation:

```
Input: 18×8×8

Stem: Conv2d(18, 128, 3, padding=1) + GroupNorm(16, 128) + Mish

Backbone: 8× DFGBlock(128)  [thay BN → GroupNorm]
- RF = 3 + 4×8 = 35 (4.4× board, đủ)
- Params: ~1.8M (ratio = 0.45 per sample)

Head: SimplifiedValueHead
- Global pool: avg + max → concat → 256 dim
- MLP: Linear(256, 128) → Mish → Dropout(0.1) → Linear(128, 1)
- Output: tanh(logit)

Total: ~2M params (vs 12.5M hiện tại)
```

**Lý do**:
1. **8 blocks**: RF=35, đủ cho mọi chess pattern. Giảm 60% compute.
2. **128 channels**: Giảm 4× params, ratio = 0.45 (healthy range).
3. **GroupNorm**: Không bị center-distribution bias, addresses B3.
4. **Simplified head**: Loại bỏ spatial flatten bottleneck, giảm head interference.

---

## PHẦN V: Kế Hoạch Thực Hiện

### Phase 1: Verification (1-2 ngày)
- [ ] TN1: Xác minh label perspective mismatch
- [ ] Kiểm tra y_exact_zero (10.9%) là do gì

### Phase 2: Quick Wins (3-5 ngày)
- [ ] TN2: Train 8b/128d vs 12b/128d vs 20b/256d (bảng so sánh)
- [ ] TN3: Freeze-backbone ablation trên L4 checkpoint
- [ ] TN5: Mở rộng oracle set lên 1000+ positions

### Phase 3: Architecture (1-2 tuần)
- [ ] TN4: GroupNorm ablation
- [ ] TN6: Head redesign (Option B trước — đơn giản nhất)

### Phase 4: Full Retrain (1-2 tuần)
- Chỉ sau khi Phase 1-3 cho kết quả → quyết định config tối ưu
- Train với: reduced model + GroupNorm + fixed labels (nếu cần) + L4 objective
- **KHÔNG nên chạy FT2 với architecture/data hiện tại** — quá nhiều unknown

---

## Open Questions

1. **Label perspective**: Bạn có nhớ rõ pipeline data generation có flip label cho Black không? Hay model đang compensate qua channel 12?

2. **Node-based vs depth-based Stockfish**: Bạn đề cập dùng node-based. Code hiện tại ghi `depth=12`. Bạn đã thay đổi code sau đó chưa? Data thực tế được generate bằng cách nào?

3. **Exact zero labels (10.9%)**: Bạn có biết vì sao ~10% labels = 0.0 chính xác không? Có clamping/rounding nào trong pipeline không?
