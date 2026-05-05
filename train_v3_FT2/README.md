# FT2 Colab Training

Folder này chứa pipeline train mới cho `FT2`, được thiết kế riêng cho:

- full rerun từ `epoch 0`,
- train trên Colab GPU `T4`,
- mount repo từ Google Drive,
- stage shard data từ Drive sang runtime disk `/content` trước khi train,
- resume an toàn khi runtime bị cắt,
- và autotune throughput thực tế thay vì chỉ chọn batch size theo OOM.

## Thành phần

- `ft2_colab_helpers.py`
  - helper chính của FT2
  - role-bundle split
  - autotune profile
  - train loop BN-safe
  - dynamic task balancing
  - PCGrad backbone merge
  - checkpoint/resume
- `train_ft2_colab.ipynb`
  - notebook runner cho Colab + Google Drive

## Những gì khác FT1

1. Main và aux không còn đi chung một BN path.
2. Aux weights không còn cố định.
3. `oracle_role_bundle` được split thành `train/val` theo position identity khi metadata cho phép.
4. Notebook tự benchmark profile T4 trước khi train.
5. Checkpointing mặc định chỉ giữ `latest`, `best_any`, `best_gate`, `best_pareto_A`, `best_pareto_B`; `latest` được ghi đè theo chu kỳ và không sinh thêm file `latest` mới.
6. Checkpoint FT2 lưu cả `model_cfg` để final eval và resume không phụ thuộc cấu hình mặc định.
7. `selected_checkpoint_eval` dùng `aux_val` thay vì full aux bundle để tránh contaminate metric role-level.

## Vì sao pipeline này nhanh hơn FT1 trên T4

FT1 run hiện tại trong [history.csv](../runs/dgrn_5m_ft1_colab_pcgrad_run1/reports/history.csv) cho thấy:

- epoch điển hình khoảng `9,000 - 9,400` giây
- median throughput chỉ khoảng `425` main samples/giây
- FT1 dùng profile `main_batch_size=256`, `grad_accum_steps=2`

Pipeline FT2 này cải thiện compute utilization bằng:

- autotune thực tế trên T4, không chỉ OOM probe
- ưu tiên `grad_accum_steps=1` nếu profile an toàn
- `channels_last` cho conv-heavy backbone
- preload shard sang `float16` khi train AMP
- prefetch shard + band order trên CPU
- mặc định không gọi `pin_memory()` thủ công trên từng batch vì với pipeline slice-from-numpy hiện tại, thao tác đó dễ chặn main thread hơn là giúp tăng throughput
- giảm số micro-step vô ích khi T4 còn trống VRAM

## Cách dùng

1. Mở `train_v3_FT2/train_ft2_colab.ipynb` trên Colab.

2. Notebook sẽ mount repo từ Google Drive. Thiết lập các env vars nếu cần:

- `CHESS_REPO_ROOT`
- `CHESS_RUNS_ROOT`
- `CHESS_DATA_ROOT`
- `CHESS_STAGE_DATA_LOCAL`
- `CHESS_LOCAL_DATA_ROOT`
- `CHESS_PERIODIC_SAVE_MINUTES`
- `CHESS_SAVE_EPOCH_CHECKPOINTS`

3. Chạy notebook từ trên xuống.

## Ghi chú thực dụng

- Notebook mặc định split `oracle_role_bundle` hiện có thành `train/val`.
- Nếu cần refresh bundle từ upstream mining pipeline, nên làm ở nhánh chuẩn bị dữ liệu trước rồi mới vào notebook train.
- `selected_checkpoint_eval.json` cuối run vẫn dùng bộ metric FT1 để so sánh công bằng với L4 và FT1.
- Mặc định `CHESS_SAVE_EPOCH_CHECKPOINTS=0` để tránh sinh thêm một file checkpoint mới sau mỗi epoch trên Google Drive.
