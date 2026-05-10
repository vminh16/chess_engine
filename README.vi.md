# Chess Engine

> Chess engine kết hợp mạng neural, giao diện web và UCI

![Python](https://img.shields.io/badge/python-%E2%89%A53.8-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md)

---

## Tổng quan dự án

Repo này phục vụ hai mục đích:

1. **Chess engine có thể chơi được** — được expose qua Flask web app và UCI engine chuẩn, kết nối được với bất kỳ GUI nào hỗ trợ UCI (Arena, Cute Chess, v.v.).
2. **Nền tảng nghiên cứu** — lịch sử thực nghiệm có cấu trúc để chẩn đoán và cải thiện chất lượng value calibration trong tree search.

Hai họ mô hình neural được bao gồm:

| Họ mô hình | Vị trí | Vai trò |
|------------|--------|---------|
| `PhantomChessNet` / `DGRNChessNet` | [`model/architecture`](model/architecture) | Legacy v1, dùng cho web UI |
| `DGRNChessNetV2` | [`model/architecture_v2`](model/architecture_v2) | Baseline nghiên cứu hiện tại |

Nhánh nghiên cứu hiện tại tập trung vào hai failure mode trong value calibration:

- **Failure A: mid-band magnitude compression**
- **Failure B: ultra-center over-confidence**

---

## Kiến trúc tổng quan

### Biểu diễn bàn cờ

Vị trí được encode thành tensor **18 × 8 × 8** theo góc nhìn side-to-move (xem [`representation/encode.py`](representation/encode.py)):

| Channel | Nội dung |
|---------|---------|
| 0 – 5 | Quân mình: Tốt, Mã, Tượng, Xe, Hậu, Vua |
| 6 – 11 | Quân địch: Tốt, Mã, Tượng, Xe, Hậu, Vua |
| 12 | Lượt đi (1 = Trắng, 0 = Đen) |
| 13 – 16 | Quyền nhập thành (mình kingside/queenside, địch kingside/queenside) |
| 17 | Ô en-passant |

Khi đến lượt Đen, bàn cờ được lật dọc để mạng luôn thấy quân mình ở phía dưới.

### Tìm kiếm

Negamax với các kỹ thuật tăng cường (xem [`search/negamax.py`](search/negamax.py)):

- Alpha-beta pruning
- Quiescence search
- Null-move pruning (R = 3, min depth 3)
- Killer moves (2 slot mỗi ply)
- History heuristic
- Late Move Reduction (LMR)
- Static Exchange Evaluation (SEE) để sắp xếp nước bắt
- Bảng chuyển vị (Transposition table)

### Đánh giá

**Evaluator lai** kết hợp điểm vật chất và điểm mạng neural:

```
eval = (1 − ε) × material_score + ε × neural_score
```

`ε` được điều chỉnh qua biến môi trường `ENGINE_EPSILON` (mặc định: `0.2` cho UCI engine, `0.1` cho web app).

### Backbone mạng neural

```
Input (18 × 8 × 8)
  → Conv2d stem (3 × 3, BatchNorm, Mish)
  → N × DFGBlock (Dual-Focus Gated residual, stochastic depth)
  → ResidualGainValueHead
  → scalar output trong [−1, 1]
```

`DGRNChessNetV2` mặc định dùng 12 blocks, 128 hidden channels; alias `DGRNChessNet` lớn hơn dùng 20 blocks và 256 channels.

---

## Cấu trúc repo

```text
core/                   Luật cờ, bàn cờ, sinh nước đi
evaluation/             naive.py (vật chất), nn.py (cầu nối neural), static_eval.py
search/                 negamax.py, ordering.py, see.py, transition_table.py, utils.py
representation/         encode.py — bàn cờ → tensor 18×8×8
model/architecture/     PhantomChessNet / DGRNChessNet v1 (legacy web UI)
model/architecture_v2/  DGRNChessNetV2 (baseline nghiên cứu hiện tại)
data/                   Script xử lý dữ liệu và dataset shard .npz
train/                  Notebook huấn luyện local (legacy)
train_v2_TF1/           Notebook FT1 tối ưu cho Google Colab
experiments/            Chẩn đoán root cause, ablation, pilot run
docs/                   Đặc tả kiến trúc, design note, research journal
bench/                  Kết quả benchmark (NPS, latency inference)
scripts/                Script tiện ích
static/ templates/      Asset giao diện web Flask
```

---

## Yêu cầu & Cài đặt

**Yêu cầu:** Python ≥ 3.8, pip

```bash
# 1. Clone repo
git clone https://github.com/vminh16/chess_engine.git
cd chess_engine

# 2. Cài đặt thư viện
pip install torch numpy flask
```

**Trọng số mô hình** cần có để engine đánh giá vị trí:

| File trọng số | Dùng bởi |
|---------------|---------|
| `model/param_model/PhantomChessNet.pth` | Web UI (`app.py`) |
| `model/nn_parameters.pth` | UCI engine (`uci.py`) |

Đặt file `.pth` vào đúng đường dẫn trên trước khi chạy. Nếu thiếu file trọng số, engine vẫn khởi động nhưng sẽ dùng thuần đánh giá vật chất và in cảnh báo.

---

## Cách sử dụng

### Giao diện web

```bash
python app.py
```

Mở `http://127.0.0.1:5000` trên trình duyệt. Engine tìm kiếm ở depth 4 và trả về nước đi tốt nhất qua REST endpoint.

### UCI Engine

```bash
python uci.py
```

Kết nối với bất kỳ GUI UCI nào (Arena, Cute Chess, ChessBase, v.v.). Các lệnh được hỗ trợ: `uci`, `isready`, `ucinewgame`, `position [startpos | fen] [moves …]`, `go`, `quit`.

Có thể điều chỉnh tỉ lệ blend neural khi khởi động:

```bash
ENGINE_EPSILON=0.3 python uci.py   # 30% neural, 70% vật chất
```

### Huấn luyện

```bash
# Huấn luyện local (legacy)
jupyter notebook train/train.ipynb

# FT1 — tối ưu cho Google Colab
jupyter notebook train_v2_TF1/train.ipynb
```

### Benchmark

Kết quả benchmark đã ghi sẵn (NPS và latency inference) nằm trong [`bench/benchmark_results.txt`](bench/benchmark_results.txt). Baseline: ~226 NPS trên CPU ở depth 4.

---

## Bối cảnh nghiên cứu

### Pipeline thực nghiệm

Các suite thực nghiệm nằm trong [`experiments/`](experiments/). Mỗi suite nhắm vào một giả thuyết cụ thể (thay đổi objective, làm sạch nhãn, điều chỉnh kiến trúc) và cache output để tái tạo kết quả. Kết quả tổng hợp nằm trong [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md).

### Kết luận hiện tại

Từ các suite trong repo, bằng chứng mạnh nhất hiện nay chỉ tới:

- **Failure A** chủ yếu là lỗi ở **phía objective** (hàm loss và target scaling).
- **Failure B** là tổ hợp của **nhãn center bẩn** và **gradient interference**.
- Fine-tune ngắn cuối kỳ không phải lời giải đáng tin cho Failure B.
- Hướng tiếp theo hợp lý nhất vẫn là can thiệp từ epoch 0 với supervision center sạch hơn.

---

## Bản đồ tài liệu

| Tài liệu | Mục đích |
|----------|---------|
| [`docs/README.md`](docs/README.md) | Mục lục tài liệu |
| [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md) | Nhật ký thực nghiệm tổng hợp |
| [`docs/design/ft1_full_retrain_pipeline_spec.md`](docs/design/ft1_full_retrain_pipeline_spec.md) | Đặc tả pipeline huấn luyện FT1 |
| [`docs/reports/project_report_vi.md`](docs/reports/project_report_vi.md) | Báo cáo dự án tiếng Việt |

---

## Đóng góp

- **Phong cách code:** Python xuyên suốt. Code lõi (board, move generator, search) có comment tiếng Việt — hãy giữ nguyên phong cách này khi chỉnh sửa.
- **Dataset và output thực nghiệm** được lưu trong repo có chủ đích để đảm bảo kết quả có thể tái tạo và truy vết mà không phụ thuộc hệ thống ngoài.
- Hiện chưa có quy ước branch cứng; khuyến khích dùng feature branch từ `main` với tên mô tả rõ ràng.
- Khi thêm hoặc sửa một suite thực nghiệm, hãy cập nhật [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md) với tóm tắt kết quả.

---

## Giấy phép

Repo hiện chưa có file license. Khuyến nghị thêm MIT License (hoặc license permissive khác) để làm rõ quyền sử dụng cho người đóng góp và người dùng.
