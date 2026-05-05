# Report 1 Training

Folder này chứa pipeline train mới cho line `report 1`, được thiết kế để chạy:

- `R1`: `8b/128d + ResidualGainValueHead + FT1-style objective`
- `R2`: `8b/128d + ResidualGainValueHead + FT1-style objective + sign-stratified sampling`

Mục tiêu của folder này là:

1. giữ **canonical objective/eval** của FT1 để so sánh công bằng với `L4` và `FT1`,
2. dùng **train loop mới** sạch hơn, có autotune profile, checkpoint cố định và resume an toàn,
3. chạy được cả **Colab** và **local CUDA**.

## Thành phần

- `report1_train_helpers.py`
  - helper chính của line report 1
  - autotune profile theo GPU thật
  - sampler `band_balanced` và `sign_stratified`
  - checkpoint `latest`, `best_any`, `best_gate`, `best_pareto_A`, `best_pareto_B`
  - final eval trên test split
- `train_report1.ipynb`
  - notebook runner cho Colab hoặc local

## Môi trường

### Colab

- Notebook tự mount Drive nếu đang chạy trên Colab.
- Dữ liệu được stage từ Drive sang `/content/chess_engine_data/process` để giảm nghẽn I/O.
- Theo [Google Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html), nên tránh nhiều I/O nhỏ trực tiếp trên Drive.
- Với `Tesla T4`, autotune hiện ưu tiên batch lớn hơn đáng kể và thử cả đường dữ liệu `preload_shard_dtype=none` để tránh bottleneck CPU khi runtime chỉ có ít vCPU.

### Local

- Notebook không mount Drive.
- Notebook tự phát hiện GPU qua `torch.cuda` và `nvidia-smi` metadata từ runtime.
- Nếu `nvidia-smi` thấy GPU nhưng `torch.cuda.is_available()` là `False`, notebook sẽ dừng sớm và báo rõ đây là lỗi kernel/PyTorch CPU-only, không phải lỗi dữ liệu.
- Cấu hình mặc định local hiện được tối ưu theo VRAM thực; với máy hiện tại, `nvidia-smi` báo `NVIDIA GeForce RTX 2050` với `4 GB`.

## Mặc định phase

- `CHESS_REPORT1_PHASE=R1`
  - `epochs=15`
  - `sampling_mode=band_balanced`
  - `run_name=dgrn_5m_report1_r1_shrink_run1`

- `CHESS_REPORT1_PHASE=R2`
  - `epochs=12`
  - `sampling_mode=sign_stratified`
  - `run_name=dgrn_5m_report1_r2_sign_run1`

Bạn vẫn có thể override toàn bộ bằng env vars.

## Env vars quan trọng

- `CHESS_REPO_ROOT`
- `CHESS_RUNS_ROOT`
- `CHESS_DATA_ROOT`
- `CHESS_STAGE_DATA_LOCAL`
- `CHESS_LOCAL_DATA_ROOT`
- `CHESS_REPORT1_PHASE`
- `CHESS_RUN_NAME`
- `CHESS_RESUME_IF_EXISTS`
- `CHESS_DISABLE_PROFILE_AUTOTUNE`
- `CHESS_PERIODIC_SAVE_MINUTES`
- `CHESS_SAVE_EPOCH_CHECKPOINTS`

## Checkpoint policy

- `ckpt_latest.pt`
  - ghi đè định kỳ, mặc định mỗi `30` phút
- `ckpt_best_any.pt`
- `ckpt_best_gate.pt`
- `ckpt_best_pareto_A.pt`
- `ckpt_best_pareto_B.pt`

Mặc định không lưu checkpoint riêng cho từng epoch để tránh tăng I/O và số file trên Drive.

## Metric và gate

Notebook/helper giữ cùng metric canonical đang dùng trong FT1/FT2:

- `oracle_midband_mae_sum_stable`
- `oracle_stable_0.7_slope`
- `center_score`
- `pooled_center_*`
- `clean_center_*`
- `ambiguous_center_*`

Gate cứng:

1. `oracle_midband_mae_sum_stable <= L4 * 1.05`
2. `oracle_stable_0.7_slope >= L4 - 0.02`

## Ghi chú thực dụng

- `sign_stratified` của line `R2` không bịa threshold mới. Nó tái dùng đúng `balance_band_edges_y600=(0.0, 0.05, 0.2, 0.5, 0.7, 1.0)` từ line `L4_A1_plus_A2`, rồi chỉ tách thêm theo `sign` cho các band ngoài center.
- `channels_last`, `AMP`, và `non_blocking=True` được giữ theo khuyến nghị hiệu năng từ [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), [Channels Last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html), và [pin_memory/non_blocking guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
- Notebook chỉ bật `TF32` khi GPU có compute capability `>= 8`; vì vậy T4 sẽ đi theo fast path `FP16 + AMP`, còn GPU Ampere local vẫn tận dụng được `TF32`.
- Helper không còn ép `uint8 -> float16/float32` ở CPU trước khi copy lên GPU; nó chuyển trực tiếp sang GPU rồi cast theo AMP path, để giảm CPU work và PCIe pressure khi dùng shard `uint8`.
- `pin_memory()` thủ công trên từng batch không bật mặc định, vì chính tài liệu PyTorch chỉ ra việc pin thủ công từ Python main thread có thể làm mất lợi ích throughput.

## Đã kiểm tra

- `python -m py_compile train_v4_report1/report1_train_helpers.py`
- import sạch helper mới
- smoke test 1 epoch trên dữ liệu thật:
  - train loop
  - checkpoint
  - val eval
  - final test eval
  - cleanup artifact tạm
