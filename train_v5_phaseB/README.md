# Train V5: Phase B Head Program

Folder này chứa pipeline huấn luyện riêng cho `Pha B`, bám theo:

- [phase_b_head_program_spec_vi_2026-04-19.md](C:/Users/USER/Desktop/chess_engine/docs/design/phase_b_head_program_spec_vi_2026-04-19.md)
- [post_ft2_progressive_plan_spec_vi_2026-04-17.md](C:/Users/USER/Desktop/chess_engine/docs/design/post_ft2_progressive_plan_spec_vi_2026-04-17.md)

## Thành phần

- [phase_b_train_helpers.py](C:/Users/USER/Desktop/chess_engine/train_v5_phaseB/phase_b_train_helpers.py)
  - wrapper mỏng bọc quanh engine `R1` ở `train_v4_report1`
  - định nghĩa run matrix `B1-B4`
  - chọn `model_cfg`, `head_type`, `sampling_mode`
  - scale autotune profile theo độ lớn model
- [train_phase_b.ipynb](C:/Users/USER/Desktop/chess_engine/train_v5_phaseB/train_phase_b.ipynb)
  - notebook env-aware
  - Colab: mount Drive, stage data sang `/content/chess_engine_data/process`
  - local: không mount, dùng CUDA local

## Run matrix

- `B1`: `16/256 + SimplifiedGlobalHead + band_balanced`
- `B2`: `16/256 + SimplifiedGlobalHead + sign_stratified`
- `B3`: `16/256 + RegimeSeparatedHead + best sampling from B1/B2`
- `B4`: `20/256 + winning head recipe from B1-B3`

## Biến môi trường chính

- `CHESS_PHASE_B_EXPERIMENT`
  - bắt buộc là `B1`, `B2`, `B3`, hoặc `B4`
  - mặc định: `B1`
- `CHESS_PHASE_B_SOURCE_EXPERIMENT`
  - cần cho `B3` nếu muốn kế thừa sampling từ `B1/B2`
  - bắt buộc cho `B4`
- `CHESS_PHASE_B_SOURCE_SAMPLING`
  - cần khi `B3` hoặc `B4(source=B3)` phải chỉ định sampling rõ
  - giá trị: `band_balanced` hoặc `sign_stratified`
  - không nên để sót biến này khi quay lại `B1/B2`; helper sẽ fail-fast nếu phát hiện source env vars thừa
- `CHESS_RUN_SUFFIX`
  - thêm suffix vào `run_name`
- `CHESS_EPOCHS_OVERRIDE`
  - ép số epoch nhỏ hơn để smoke-test
- `CHESS_STAGE_DATA_LOCAL`
  - trên Colab mặc định `1`

## Ví dụ cấu hình

### B1 trên Colab

```python
%env CHESS_PHASE_B_EXPERIMENT=B1
%env CHESS_STAGE_DATA_LOCAL=1
```

### B2 trên Colab

```python
%env CHESS_PHASE_B_EXPERIMENT=B2
%env CHESS_STAGE_DATA_LOCAL=1
```

Khi đổi từ `B3/B4` về `B1/B2`, cần xóa các biến cũ:

```python
%env CHESS_PHASE_B_SOURCE_EXPERIMENT=
%env CHESS_PHASE_B_SOURCE_SAMPLING=
```

### B3 dùng sampling thắng từ B2

```python
%env CHESS_PHASE_B_EXPERIMENT=B3
%env CHESS_PHASE_B_SOURCE_EXPERIMENT=B2
```

### B4 xác nhận recipe thắng từ B2 trên 20/256

```python
%env CHESS_PHASE_B_EXPERIMENT=B4
%env CHESS_PHASE_B_SOURCE_EXPERIMENT=B2
```

### B4 xác nhận recipe thắng từ B3

```python
%env CHESS_PHASE_B_EXPERIMENT=B4
%env CHESS_PHASE_B_SOURCE_EXPERIMENT=B3
%env CHESS_PHASE_B_SOURCE_SAMPLING=sign_stratified
```

## Ghi chú kỹ thuật

- Pipeline này **không fork lại** train loop `R1`; nó tái dùng engine đó để giữ:
  - checkpointing,
  - eval/gate,
  - logging,
  - periodic overwrite `ckpt_latest.pt`,
  - autotune trên T4.
- `architecture_v2.model` đã được mở rộng để hỗ trợ `head_type`:
  - `residual_gain`
  - `simplified_global`
  - `regime_separated`
- T4 profile cho `Pha B` được scale riêng theo độ lớn model:
  - `16/256` được phép thử batch lớn hơn `FT2`,
  - `20/256` giữ batch thận trọng hơn để cân bằng hội tụ và throughput.

## Validation tối thiểu nên chạy

1. mở notebook
2. chạy `B1` với `CHESS_EPOCHS_OVERRIDE=1`
3. kiểm tra:
   - `train_batch_autotune.json`
   - `history.csv`
   - `decision_summary.json`
   - `selected_checkpoint_eval.json`

Chỉ khi dry-run sạch mới nên launch run dài.
