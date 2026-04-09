# Kho Mã Chess Engine Nghiên Cứu

[English](README.md)

Đây là repo kết hợp giữa chess engine tự xây dựng, bộ đánh giá thế cờ bằng mạng neural, và toàn bộ lịch sử thực nghiệm dùng để chẩn đoán cũng như cải thiện chất lượng value network. Nhánh nghiên cứu hiện tại tập trung vào `architecture_v2`, biểu diễn `18x8x8` theo góc nhìn side-to-move, và chuỗi suite dùng để tách riêng hai failure mode chính:

- **Failure A: mid-band magnitude compression**
- **Failure B: ultra-center over-confidence**

## Repo này chứa gì

- Logic cờ và sinh nước đi trong [`core`](core).
- Search cổ điển trong [`search`](search) và cầu nối evaluator trong [`evaluation`](evaluation).
- Hai họ kiến trúc mạng trong [`model`](model), trong đó baseline nghiên cứu hiện tại nằm ở [`model/architecture_v2`](model/architecture_v2).
- Pipeline xử lý dữ liệu, dữ liệu shard, và điểm vào huấn luyện trong [`data`](data), [`train`](train), [`train_v2_TF1`](train_v2_TF1).
- Các suite thực nghiệm và báo cáo cached trong [`experiments`](experiments).
- Tài liệu kỹ thuật đã được gom nhóm lại trong [`docs`](docs).

## Bản đồ tài liệu

- [`docs/README.md`](docs/README.md): mục lục tài liệu
- [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md): nhật ký thực nghiệm tổng hợp
- [`docs/design/ft1_full_retrain_pipeline_spec.md`](docs/design/ft1_full_retrain_pipeline_spec.md): đặc tả FT1
- [`docs/reports/project_report_vi.md`](docs/reports/project_report_vi.md): báo cáo dự án bằng tiếng Việt

## Cấu trúc chính

```text
core/               Luật cờ, bàn cờ, sinh nước đi
evaluation/         Các evaluator static và neural
search/             Negamax, ordering, SEE, TT
model/              Kiến trúc mạng và mã model
data/               Notebook và script xử lý dữ liệu
train/              Pipeline huấn luyện cũ / local
train_v2_TF1/       Notebook FT1 tối ưu cho Colab
experiments/        Chẩn đoán root cause, ablation, pilot run
docs/               Kiến trúc, design note, research log, report
```

## Kết luận nghiên cứu hiện tại

Từ các suite trong repo, bằng chứng mạnh nhất hiện nay đang chỉ tới:

- Failure A là lỗi chủ yếu ở **objective**.
- Failure B là tổ hợp của **nhãn center bẩn** và **gradient interference**.
- Fine-tune ngắn cuối kỳ không phải lời giải đáng tin cho Failure B.
- Hướng tiếp theo hợp lý nhất vẫn là can thiệp từ epoch 0 với supervision center sạch hơn.

## Cách chạy nhanh

- Web UI: `python app.py`
- UCI engine: `python uci.py`
- Huấn luyện cũ: mở [`train/train.ipynb`](train/train.ipynb)
- FT1 trên Colab: mở [`train_v2_TF1/train.ipynb`](train_v2_TF1/train.ipynb)

## Ghi chú

- Repo giữ nhiều output thực nghiệm để tiện nghiên cứu và truy vết.
- Một số đường dẫn trong tài liệu cũ là tham chiếu lịch sử tới checkpoint ngoài repo; bố cục mới gom tài liệu đang dùng về dưới [`docs`](docs).
