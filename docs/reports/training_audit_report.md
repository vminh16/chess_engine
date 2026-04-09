# Báo Cáo Audit Pipeline Huấn Luyện

## Phạm vi kiểm tra

Báo cáo này kiểm tra 4 lớp vấn đề của pipeline huấn luyện hiện tại:

1. Logic resume/checkpoint trong notebook Colab [train.ipynb](C:/Users/USER/Downloads/train.ipynb)
2. Kiến trúc `architecture_v2` trong repo:
   - [model.py](../../model/architecture_v2/model.py)
   - [head.py](../../model/architecture_v2/head.py)
   - [blocks.py](../../model/architecture_v2/blocks.py)
3. Phân phối dữ liệu giữa `train/val/test`
4. Kiểm tra thực nghiệm trực tiếp trên checkpoint thật trong [runs/dgrn_5m_v2_run1](../../runs/dgrn_5m_v2_run1)

Tất cả kết luận trong báo cáo này chỉ dùng dữ kiện có thể xác minh trực tiếp từ code, checkpoint và dữ liệu local. Không có phần nào dựa trên phỏng đoán không kiểm chứng.

## Tóm tắt kết luận

Kết luận đáng tin cậy nhất hiện tại là:

- Resume của run Colab `dgrn_5m_v2_run1` **không bị hỏng** theo kiểu “luôn fine-tune thay vì continue”.
- Drift mà bạn thấy trên Colab là **có thật ngay cả khi resume đúng**.
- Nguyên nhân mạnh nhất hiện tại là **objective mismatch**:
  - train tối ưu `logit-space MSE`
  - val và chọn checkpoint tốt nhất lại dùng `y-space MSE`
- Kiến trúc `architecture_v2` vẫn có rủi ro về generalization vì:
  - mạng khá lớn (`9.05M` tham số)
  - rất nhiều `BatchNorm2d` (`84` lớp)
  - regularization của run hiện tại tương đối nhẹ (`drop_path_rate=0.05`, `weight_decay=1e-4`)
- Tuy nhiên, với bằng chứng hiện có, **kiến trúc không phải nghi phạm số 1**. Pipeline optimization là nghi phạm số 1.

## Nguồn dữ liệu đã audit

- Notebook Colab: [train.ipynb](C:/Users/USER/Downloads/train.ipynb)
- Notebook train local: [train/train.ipynb](../../train/train.ipynb)
- Source run history: [history.csv](../../runs/dgrn_5m_v2_run1/history.csv)
- Local run history: [history.csv](../../runs/dgrn_5m_v2_local_4gb_run1/history.csv)
- Checkpoint đã kiểm tra:
  - [ckpt_best.pt](../../runs/dgrn_5m_v2_run1/ckpt_best.pt)
  - [ckpt_latest.pt](../../runs/dgrn_5m_v2_run1/ckpt_latest.pt)
- Kiến trúc:
  - [model.py](../../model/architecture_v2/model.py)
  - [head.py](../../model/architecture_v2/head.py)
  - [blocks.py](../../model/architecture_v2/blocks.py)

## Các sự thật đã được xác minh

### 1. Resume của run Colab là tiếp tục train thật, không phải fine-tune ngầm

Bằng chứng từ code trong [train.ipynb](C:/Users/USER/Downloads/train.ipynb):

- `try_resume()` load đầy đủ:
  - `model`
  - `optimizer`
  - `scaler`
  - `scheduler`
  - `history`
  - `rng_state`
  - `epoch_state`
- Nó khôi phục đúng vị trí sampler bằng:
  - `train_batch_sampler.set_epoch(epoch)`
  - `train_batch_sampler.set_start_batch(resume_batch_offset)`
- Nó khôi phục luôn running average giữa epoch từ `epoch_state` khi resume ở giữa epoch.

Bằng chứng từ checkpoint [ckpt_latest.pt](../../runs/dgrn_5m_v2_run1/ckpt_latest.pt):

- có đủ các khóa: `model`, `optimizer`, `scheduler`, `scaler`, `history`, `config`, `rng_state`, `epoch_state`
- `optimizer` có state thật: `366` entries
- `scheduler` có state thật
- `rng_state` có cả `torch_cuda`
- metadata thật của checkpoint:
  - `epoch=9`
  - `epoch_step=1623`
  - `is_epoch_end=False`
  - `global_step=19200`

Bằng chứng thực nghiệm mạnh nhất:

Log resume Colab của bạn ghi:
- `Resumed from ... ckpt_latest.pt: epoch=6, step=682, is_epoch_end=False, global_step=12400`
- cuối epoch 6 sau resume:
  - `train_metric=0.080470`
  - `train_obj=0.223432`
  - `val_loss=0.082142`

Các số này khớp với [history.csv](../../runs/dgrn_5m_v2_run1/history.csv) của epoch 6 ở mức chính xác đúng kiểu log thông thường.

### Kết luận

Với source run Colab `dgrn_5m_v2_run1`, giả thuyết “load checkpoint bị hỏng nên thực ra luôn fine-tune” là **không được dữ liệu ủng hộ**.

Mức tin cậy: **cao**.

### 2. Phân phối target của train/val/test gần như trùng khớp hoàn toàn

Tôi đã đo trực tiếp từ toàn bộ `y_*.npy`:

- `train`: `4,000,000` mẫu, `80` shard
- `val`: `500,000` mẫu, `10` shard
- `test`: `500,000` mẫu, `10` shard

Histogram 20 bucket của target:

- `train`:
  `[56000, 88000, 136000, 140000, 116000, 108000, 116000, 140000, 240000, 860000, 860000, 240000, 140000, 116000, 108000, 116000, 140000, 136000, 88000, 56000]`
- `val`:
  `[7000, 11000, 17000, 17500, 14500, 13500, 14500, 17500, 30000, 107500, 107500, 30000, 17500, 14500, 13500, 14500, 17500, 17000, 11000, 7000]`
- `test`:
  `[7000, 11000, 17000, 17500, 14500, 13500, 14500, 17500, 30000, 107500, 107500, 30000, 17500, 14500, 13500, 14500, 17500, 17000, 11000, 7000]`

Các tỷ lệ chính trùng nhau ở cả ba split:

- `|y| <= 0.1`: `43.0%`
- `|y| > 0.5`: `26.8%`

### Kết luận

Drift hiện tại **không thể giải thích bằng việc train/val/test bị lệch phân phối target**.

Mức tin cậy: **cao**.

### 3. Mô hình hiện tại lớn và rất phụ thuộc BatchNorm

Đo trực tiếp từ [model.py](../../model/architecture_v2/model.py):

- tổng tham số: `9,047,682`
- stem: `41,984`
- backbone blocks: `7,733,120`
- head: `1,272,578`
- số lớp `BatchNorm2d`: `84`
- số lớp `Conv2d`: `124`
- số lớp `Linear`: `7`

### Kết luận

Kiến trúc hiện tại không nhỏ. Đây là một teacher network hợp lý, nhưng không phải kiểu model “khó overfit”. Số lượng `BatchNorm2d` cao khiến mô hình nhạy với regime train và running statistics.

Mức tin cậy: **cao**.

## Rủi ro chính số 1: Objective mismatch giữa train và cách chọn checkpoint

### Code hiện tại đang làm gì

Trong [train.ipynb](C:/Users/USER/Downloads/train.ipynb):

- objective khi train:
  - `objective = MSE(logits, atanh(clamp(y)))`
- metric khi val / chọn checkpoint tốt nhất:
  - `metric = MSE(tanh(logits), y)`
- checkpoint tốt nhất được quyết định theo `val_loss`, tức là metric ở `y-space`

Tức là hệ thống đang tối ưu một hàm nhưng lại chọn model tốt nhất bằng một hàm khác.

### Hệ quả toán học

Gọi `y in [-1, 1]` là target và `z` là logit của model.

Objective train:

```text
L_train(z, y) = (z - atanh(y_clamped))^2
```

Metric val:

```text
L_val(z, y) = (tanh(z) - y)^2
```

Hai loss này không tương đương.

Đặc biệt, gần biên `|y| -> 1`, ta có:

```text
d/dy atanh(y) = 1 / (1 - y^2)
```

Nghĩa là `atanh(y)` tăng rất nhanh khi `|y|` tiến sát `1`. Hệ quả là train objective vô tình đặt trọng số lớn hơn cho việc khớp chính xác logit ở các target extreme. Vì vậy ở giai đoạn cuối, model hoàn toàn có thể tiếp tục cải thiện `L_train` bằng cách đẩy logits quyết liệt hơn, trong khi `tanh(z)` ở `y-space` lại trở nên kém calibrated hơn.

### Chứng minh thực nghiệm trực tiếp

Tôi đã đánh giá **toàn bộ validation** và **toàn bộ test** trên hai checkpoint thật của cùng một run:

Validation, `500,000` mẫu:

- `ckpt_best.pt`
  - `y_mse = 0.08214281`
  - `z_mse = 0.23280982`
  - `mean_abs_pred = 0.26060`
- `ckpt_latest.pt`
  - `y_mse = 0.08313516`  xấu hơn
  - `z_mse = 0.23031766`  tốt hơn
  - `mean_abs_pred = 0.27250`  dự đoán quyết liệt hơn

Test, `500,000` mẫu:

- `ckpt_best.pt`
  - `y_mse = 0.08184389`
  - `z_mse = 0.23395949`
- `ckpt_latest.pt`
  - `y_mse = 0.08270908`  xấu hơn
  - `z_mse = 0.23135314`  tốt hơn

Đây là bằng chứng trực tiếp rằng checkpoint mới hơn tiếp tục cải thiện đúng objective đang tối ưu, nhưng lại làm metric đánh giá và metric triển khai xấu đi, trên cả validation lẫn test.

### Kết luận

Rủi ro này đã được **xác nhận trực tiếp** và hiện là lời giải thích mạnh nhất cho calibration drift.

Mức tin cậy: **rất cao**.

### Hướng khắc phục

- Dùng objective gần hơn với metric triển khai thực tế.
- Nếu vẫn muốn giữ target bounded trong `[-1,1]`, có hai hướng hợp lý:
  - train trực tiếp ở `y-space`
  - hoặc dùng objective hỗn hợp gồm cả `logit-space` và `y-space`
- Nếu vẫn giữ `logit-space`, khi chọn checkpoint không nên chỉ nhìn một scalar `val_loss`; cần theo dõi thêm:
  - bucketed calibration
  - signed bias
  - mean absolute prediction

## Rủi ro chính số 2: Generalization drift ở cuối run là thật, không phải nhiễu log

### Bằng chứng từ history

Source run [history.csv](../../runs/dgrn_5m_v2_run1/history.csv):

- epoch 6:
  - `train_loss = 0.08047017`
  - `val_loss = 0.08214210`
- epoch 8:
  - `train_loss = 0.07539968`
  - `val_loss = 0.08555712`

Generalization gap `val - train`:

- epoch 4: `+0.001651`
- epoch 5: `+0.005225`
- epoch 6: `+0.001672`
- epoch 7: `+0.006640`
- epoch 8: `+0.010157`

Local fine-tune run [history.csv](../../runs/dgrn_5m_v2_local_4gb_run1/history.csv) cũng cho đúng xu hướng này, còn mạnh hơn:

- best ở epoch 3: `val_loss = 0.08067560`
- epoch 9: `val_loss = 0.08378450`
- gap tăng tới `+0.025044`

### Bằng chứng thực nghiệm bổ sung

Từ đo full-val trên checkpoint thật:

- `ckpt_latest.pt` có `mean_abs_pred` lớn hơn `ckpt_best.pt`
- signed mean prediction drift từ âm nhẹ sang dương nhẹ:
  - `ckpt_best`: `mean_pred = -0.01716`
  - `ckpt_latest`: `mean_pred = +0.00720`
  - target mean chỉ khoảng `-0.00456`

Đây là dấu hiệu rất điển hình của calibration drift chứ không phải dao động ngẫu nhiên.

### Kết luận

Generalization degradation sau checkpoint tốt nhất là **có thật**.

Mức tin cậy: **cao**.

### Hướng khắc phục

- Sau khi model đạt cực tiểu validation đầu tiên, không được dùng việc “train loss còn giảm” làm dấu hiệu tốt duy nhất.
- Việc chọn checkpoint nên phụ thuộc ít nhất vào:
  - `val_loss`
  - signed bias
  - bucketed calibration
- Với regime train hiện tại, checkpoint tốt nhất xuất hiện sớm hơn checkpoint cuối run.

## Rủi ro phụ số 3: Scheduler hiện tại phản ứng quá chậm

### Bằng chứng từ code và history

Notebook Colab [train.ipynb](C:/Users/USER/Downloads/train.ipynb) dùng:

- `CosineAnnealingLR(T_max=50, eta_min=1e-5)`
- `LR = 1e-4`

LR quanh vùng best:

- epoch 6: `9.57e-05`
- epoch 7: `9.44e-05`
- epoch 8: `9.30e-05`

Tức là ngay cả sau khi validation đã đạt điểm tốt nhất, learning rate vẫn gần như giữ nguyên mức ban đầu. Scheduler hiện tại không có cơ chế phản ứng khi validation bắt đầu xấu đi.

### Kết luận

Đây không phải bug, nhưng là một rủi ro huấn luyện có thật và hợp với pattern overfit hiện tại.

Mức tin cậy: **trung bình-cao**.

### Hướng khắc phục

- Dùng schedule bảo thủ hơn ở vùng cuối hoặc có phản ứng với validation.
- Nếu vẫn giữ objective hiện tại, phần late training phải ngắn hơn và thận trọng hơn.

## Rủi ro phụ số 4: Regularization đang nhẹ hơn cấu hình mặc định của model

### Bằng chứng từ code

Trong [model.py](../../model/architecture_v2/model.py), alias `DGRNChessNet` mặc định dùng:

- `drop_path_rate = 0.1`

Nhưng notebook Colab [train.ipynb](C:/Users/USER/Downloads/train.ipynb) lại train với:

- `drop_path_rate = 0.05`
- `weight_decay = 1e-4`

Với một mạng `9.05M` tham số, đây là regularization tương đối nhẹ.

### Kết luận

Đây là một tác nhân plausible góp phần làm generalization xấu đi, nhưng dữ kiện hiện tại chưa đủ để cô lập nó là nguyên nhân chính.

Mức tin cậy: **trung bình**.

### Hướng khắc phục

- Tăng regularization sau khi đã xử lý xong bài toán objective mismatch.
- Không nên tune regularization tách rời objective.

## Rủi ro phụ số 5: Resume strictness trong notebook Colab đang quá lỏng

### Bằng chứng từ code

Trong [train.ipynb](C:/Users/USER/Downloads/train.ipynb):

- `RESUME_STRICT_CONFIG = False`
- `ALLOW_OVERWRITE_EXISTING_RUN = True`

Điều này có nghĩa là nếu bạn thay config mà vẫn giữ cùng `RUN_NAME`, notebook vẫn có thể resume tiếp mà không chặn mismatch.

### Kết luận

Đây là rủi ro vận hành có thật, nhưng **không cần đến nó** để giải thích drift hiện tại của source run. Drift hiện tại đã tồn tại ngay cả dưới true resume.

Mức tin cậy:

- **trung bình** nếu xem như rủi ro tiềm ẩn
- **thấp** nếu xem nó là lời giải thích cho drift hiện tại

### Hướng khắc phục

- Bật strict config matching cho các run nghiêm túc.
- Mỗi khi thay một trong các thứ sau, phải đổi `RUN_NAME`:
  - model config
  - objective
  - batch size
  - schedule
  - weight decay

## Đánh giá kiến trúc cho repo này

### Điều phù hợp

Với repo này, value network phải phục vụ hai vai trò:

1. Teacher-quality regression trong giai đoạn train
2. Scalar evaluation nhanh trong search khi suy luận

`architecture_v2` là một teacher hợp lý vì:

- vẫn giữ bias không gian 8x8 phù hợp với bàn cờ
- head có direct global score path
- output vẫn bounded được để dùng trong engine

### Điều còn đắt đỏ

Cho mục tiêu triển khai trực tiếp vào search:

- `9.05M` params là không nhỏ
- `84` lớp BatchNorm và `124` conv là tốn chi phí
- kiến trúc này phù hợp hơn với vai trò teacher hơn là model cuối cùng trong engine

### Hệ quả cho triển khai tương lai

Một lộ trình hợp lý là:

1. sửa generalization của teacher trước
2. sau đó distill sang student nhỏ hơn
3. rồi quantize student
4. cuối cùng mới ghép vào search

Không nên distill quá sớm từ một teacher mà calibration vẫn đang drift dưới objective hiện tại.

## Đánh giá phân phối dữ liệu

Phân phối của `train/val/test` đã được kiểm soát rất chặt:

- histogram target gần như trùng hoàn toàn giữa các split
- center-heavy design là chủ đích và được giữ nhất quán

Vì vậy failure mode hiện tại **không phải do split bị lệch nhau**.

Điều này không có nghĩa là center-heavy design vô hại; nó chỉ có nghĩa là nó không giải thích được sự khác biệt train/val hiện tại nếu đứng một mình.

## Xếp hạng rủi ro hiện tại

1. **Objective mismatch giữa train và cách chọn checkpoint**
   - đã được xác nhận
   - ưu tiên cao nhất
2. **Late-stage calibration drift sau checkpoint tốt nhất**
   - đã được xác nhận
   - ưu tiên rất cao
3. **Scheduler quá chậm trong vùng sau cực tiểu validation**
   - plausible và được history ủng hộ
4. **Regularization hơi yếu so với kích thước mô hình**
   - plausible, chưa tách bạch hoàn toàn
5. **Resume strictness lỏng**
   - rủi ro vận hành, nhưng không phải nguyên nhân chính của drift hiện tại
6. **Kiến trúc bị lỗi nghiêm trọng**
   - không được bằng chứng hiện tại ủng hộ

## Hướng hành động đề xuất

### Ưu tiên cao nhất

- Sửa objective train để gần metric triển khai hơn.
- Tiếp tục dùng full `500k` validation vì nó đang cho tín hiệu rất sạch.
- Dùng `ckpt_best` hiện tại làm checkpoint chuẩn, không dùng `ckpt_latest` làm mốc tốt nhất.

### Ưu tiên trung bình

- Siết resume policy của notebook Colab.
- Tăng regularization sau khi đã xử lý objective mismatch.
- Theo dõi calibration trực tiếp bằng:
  - signed bias
  - mean absolute prediction
  - bucketed MSE

### Dài hạn

- Giữ `architecture_v2` như teacher candidate.
- Sau khi teacher generalize ổn định, mới distill sang student để triển khai engine.

## Kết luận cuối cùng

Bằng chứng hiện tại ủng hộ một kết luận rất rõ:

- Colab run đang resume đúng.
- Phân phối train/val/test là nhất quán.
- Failure mode chính hiện tại không phải “resume bug”, và cũng chưa có bằng chứng mạnh cho “kiến trúc bị lỗi gốc”.
- Vấn đề mạnh nhất đã được xác nhận là **objective train và metric chọn checkpoint đang lệch nhau**.

Chỉ riêng vấn đề đó đã đủ để tạo ra calibration drift như bạn quan sát, và kết luận này hiện đã được hỗ trợ đồng thời bởi:

- đọc code notebook
- kiểm tra checkpoint thật
- đánh giá full validation
- đánh giá full test

