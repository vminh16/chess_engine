# train_v2_FT1

Gói Colab cho FT1, không kèm `data/`.

## Có gì trong gói

- `train/train.ipynb`: notebook train FT1
- `train/ft1_colab_helpers.py`: helper train/eval FT1
- `model/`, `core/`, `evaluation/`, `representation/`: code model và dependency cần để import
- `experiments/...`: chỉ các helper, checkpoint, và cache oracle/bundle mà FT1 thực sự dùng
- `runs/`: thư mục output rỗng

## Không có trong gói

- `data/process`

Bạn cần mount data riêng trên Colab, ví dụ:

`/content/drive/MyDrive/chess_engine_data/process`

Notebook đã được sửa để:

- ưu tiên `REPO_ROOT/data/process` nếu thư mục này tồn tại
- nếu không có, dùng biến môi trường `CHESS_DATA_ROOT`
- nếu biến môi trường không có, fallback về:

`/content/drive/MyDrive/chess_engine_data/process`

## Cách dùng trên Colab

1. Upload toàn bộ thư mục `train_v2_FT1` lên Drive.
2. Đảm bảo data shards nằm ở một thư mục ngoài package, ví dụ:
   `/content/drive/MyDrive/chess_engine_data/process/train`
   `/content/drive/MyDrive/chess_engine_data/process/val`
   `/content/drive/MyDrive/chess_engine_data/process/test`
3. Mở notebook:
   [train.ipynb](/C:/Users/USER/Desktop/chess_engine/train_v2_FT1/train/train.ipynb)
4. Nếu data không nằm ở path mặc định, sửa `COLAB_DATA_ROOT` ở cell data scan hoặc set:

```python
import os
os.environ["CHESS_DATA_ROOT"] = "/content/drive/MyDrive/your_path/process"
```

## Oracle / Stockfish

Notebook này không gọi Stockfish khi train.

FT1 dùng:

- `oracle_subset_rows.csv` làm oracle probe cố định để eval
- `oracle_role_bundle` làm trusted-center oracle bank
- `pooled_center_bundle` làm center eval bundle
- `L4_A1_plus_A2_best.pt` làm reference checkpoint

Các artifact trên đã được copy sẵn vào package.

## Lưu ý thực nghiệm

- `RUN_NAME` mặc định vẫn là `dgrn_5m_ft1_colab_t4_run1`.
- Helper hiện có khả năng resume nếu trong `runs/<RUN_NAME>/checkpoints` đã có `ckpt_latest.pt`.
- Package này giữ nguyên logic FT1 hiện tại; nó không tự tạo oracle mới.
