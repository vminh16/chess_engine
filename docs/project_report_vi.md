# Báo Cáo Tổng Hợp Dự Án Chess Engine

## 1. Mục tiêu dự án

Dự án này là một chess engine tự xây dựng, kết hợp:

- bộ biểu diễn bàn cờ và luật cờ riêng
- thuật toán search kiểu `negamax` với nhiều heuristic cổ điển
- mạng neural để hỗ trợ đánh giá vị trí
- pipeline xử lý dữ liệu và huấn luyện chủ yếu chạy bằng notebook

Mục tiêu thực tế của hệ hiện tại không phải mô phỏng AlphaZero toàn phần, mà là:

- xây một **static evaluator** tốt hơn các heuristic đơn giản
- sau đó để **search** sửa phần tactical và tận dụng evaluator đó ở leaf / stand-pat / move ordering

---

## 2. Cấu trúc chính của dự án

Các thành phần quan trọng nhất hiện tại:

- [core](/C:/Users/USER/Desktop/chess_engine/core)
  - cài đặt bàn cờ, luật đi, sinh nước đi
- [representation](/C:/Users/USER/Desktop/chess_engine/representation)
  - mã hóa trạng thái bàn cờ thành tensor
- [search](/C:/Users/USER/Desktop/chess_engine/search)
  - `negamax`, quiescence, ordering, SEE, TT, LMR, killer/history heuristic
- [model](/C:/Users/USER/Desktop/chess_engine/model)
  - kiến trúc mạng và tham số đã huấn luyện
- [data](/C:/Users/USER/Desktop/chess_engine/data)
  - notebook xử lý dữ liệu từ nguồn raw
- [train](/C:/Users/USER/Desktop/chess_engine/train)
  - notebook huấn luyện chính
- [docs](/C:/Users/USER/Desktop/chess_engine/docs)
  - các tài liệu phân tích và audit kỹ thuật

Hai nhánh mô hình hiện tồn tại trong repo:

- nhánh cũ: [model/architecture](/C:/Users/USER/Desktop/chess_engine/model/architecture)
- nhánh mới đang được dùng để nghiên cứu: [model/architecture_v2](/C:/Users/USER/Desktop/chess_engine/model/architecture_v2)

---

## 3. Biểu diễn đầu vào

Tệp chính: [representation/encode.py](/C:/Users/USER/Desktop/chess_engine/representation/encode.py)

Mô hình dùng tensor `18 x 8 x 8`, trong đó:

- plane `0-5`: quân của phía hiện tại
- plane `6-11`: quân của đối thủ
- plane `12`: side-to-move
- plane `13-16`: castling rights
- plane `17`: en passant target

Điểm quan trọng của encoding:

- bàn cờ được xoay theo phía đang đi
- tức là representation mang tính **STM-relative**
- đây là lựa chọn hợp lý cho evaluator kiểu search, vì giảm gánh nặng học đối xứng trắng/đen

Giới hạn hiện tại của encoding:

- không thấy thông tin rõ về repetition
- không thấy halfmove clock / 50-move rule trong tensor
- không có feature thủ công kiểu mobility, king zone, pinnedness, attack maps

Điều này quan trọng vì nó giới hạn lượng thông tin “static meaning” mà mạng có thể học từ một snapshot duy nhất.

---

## 4. Dữ liệu huấn luyện

Nguồn dữ liệu chính được xử lý trong notebook:

- [data/process_data.ipynb](/C:/Users/USER/Desktop/chess_engine/data/process_data.ipynb)

Các cấu hình chính đã xác minh được từ notebook:

- nguồn raw: `lichess_db_eval.jsonl.zst`
- `CP_SCALE = 600.0`
- `FIXED_DEPTH = 25`
- `MIN_KNODES = 50_000`
- `MAX_ABS_CP = 1200`
- `KEEP_MATE_PROB = 0.10`
- tổng target mẫu: `5,000,000`
- kích thước shard: `50,000`
- split:
  - train `80%`
  - val `10%`
  - test `10%`

Sau xử lý, tập dữ liệu thực tế có:

- train: `4,000,000` mẫu, `80` shard
- val: `500,000` mẫu, `10` shard
- test: `500,000` mẫu, `10` shard

### Chuẩn hóa target

Target hiện tại được map bằng:

```text
y = tanh(cp / 600)
```

Hệ quả của phép biến đổi này:

- gần `0`, ánh xạ gần như tuyến tính
- ở tails, các giá trị centipawn lớn bị nén mạnh
- ví dụ:
  - `cp = 300 -> y ≈ 0.462`
  - `cp = 600 -> y ≈ 0.762`
  - `cp = 1000 -> y ≈ 0.931`
  - `cp = 1200 -> y ≈ 0.964`

Về mặt thực dụng, phép nén này làm cho:

- vùng cực trị bớt chi phối hơn nếu nhìn theo `y-space`
- nhưng cũng khiến thông tin phân biệt giữa các position thắng rất lớn bị co lại

### Phân phối dữ liệu

Notebook chủ động áp bucket quota để ép phân phối gần đối xứng quanh `0`.

Phân phối đích 20 bucket hiện tại:

- rất nặng ở trung tâm
- hai bucket quanh `0` mỗi bucket khoảng `21.5%`
- tổng `|y| <= 0.1` khoảng `43%`
- tổng `|y| > 0.5` khoảng `26.8%`

Đây là một phân phối có chủ đích:

- ưu tiên trạng thái cân bằng / lợi thế nhỏ
- giảm xác suất model chỉ học “đánh mạnh mọi thứ”

Tuy nhiên, phân phối này cũng tạo ra một trade-off:

- rất hợp nếu mục tiêu là evaluator phục vụ search
- nhưng khiến bài toán tails khó hơn nếu vẫn đo thành công chủ yếu bằng MSE toàn cục

---

## 5. Pipeline huấn luyện

Notebook huấn luyện chính:

- [train/train.ipynb](/C:/Users/USER/Desktop/chess_engine/train/train.ipynb)

Ngoài ra còn có notebook Colab ngoài repo đã được vá và audit trong quá trình nghiên cứu.

### Quy trình hiện tại

Quy trình tổng quát:

1. đọc shard `X_*.npy`, `y_*.npy`
2. khởi tạo `architecture_v2`
3. train bằng `AdamW`
4. scheduler dạng `CosineAnnealingLR`
5. dùng checkpoint `best/latest`
6. theo dõi thêm metric bucket / calibration ở các giai đoạn audit gần đây

### Loss hiện tại

Qua quá trình audit, loss đã được tiến hóa từ:

- pure `y-space`
- sang pure `logit-space`
- rồi sang hybrid loss có curriculum

Lý do là:

- pure `y-space` học tails yếu
- pure `logit-space` gây calibration drift
- hybrid curriculum cân bằng hai mục tiêu đó tốt hơn

Tài liệu chi tiết:

- [docs/disigner_loss_func.md](/C:/Users/USER/Desktop/chess_engine/docs/disigner_loss_func.md)

---

## 6. Kiến trúc mạng hiện tại

Tệp chính:

- [model/architecture_v2/model.py](/C:/Users/USER/Desktop/chess_engine/model/architecture_v2/model.py)
- [model/architecture_v2/blocks.py](/C:/Users/USER/Desktop/chess_engine/model/architecture_v2/blocks.py)
- [model/architecture_v2/head.py](/C:/Users/USER/Desktop/chess_engine/model/architecture_v2/head.py)

### 6.1. Tổng quan

`DGRNChessNetV2` gồm 3 phần:

1. `stem`
2. chuỗi `DFGBlock`
3. `ResidualGainValueHead`

Default tuned config hiện tại:

- `num_blocks = 20`
- `hidden_dim = 256`
- `input_channels = 18`
- `drop_path_rate = 0.1`
- `output_mode = "tanh"`

### 6.2. Stem

Stem gồm:

- `Conv2d(18 -> 256, k=3, padding=1, bias=False)`
- `BatchNorm2d`
- `Mish`

Vai trò:

- đưa input chess tensor vào không gian feature rộng hơn

### 6.3. DFGBlock

Mỗi block trong [blocks.py](/C:/Users/USER/Desktop/chess_engine/model/architecture_v2/blocks.py) chia feature theo channel thành hai nhánh:

- `local_conv`
  - `3x3`
- `remote_conv`
  - `3x3 dilation=2`

Sau đó:

- concat hai nhánh
- đưa qua `CoordinateAttention(reduction=8)`
- fuse bằng `1x1 conv + BN + Mish`
- cộng residual shortcut

Ý tưởng:

- local branch học motif cục bộ
- remote branch học quan hệ xa hơn
- coordinate attention giữ thông tin theo trục hàng/cột

### 6.4. ResidualGainValueHead

Head hiện tại trong [head.py](/C:/Users/USER/Desktop/chess_engine/model/architecture_v2/head.py) đã sửa so với head cũ.

Các thành phần:

- `gain_mlp`
  - lấy `GAP + GMP`
  - sinh ra hệ số gain theo channel
- `spatial_conv`
  - đọc feature không gian `8x8`
- `spatial_score`
  - sinh scalar từ feature không gian
- `global_score`
  - sinh scalar trực tiếp từ pooled context

Điểm quan trọng:

- head mới dùng **residual gain**
  - có thể amplify và suppress trong biên có kiểm soát
- có **global score branch trực tiếp**
  - sửa bottleneck lớn nhất của `ContextGatedHead` cũ

### 6.5. Quy mô mô hình

Theo audit trước đó, teacher hiện tại cỡ:

- tổng tham số khoảng `9.05M`
- số lớp `BatchNorm2d` rất nhiều

Điều này có hai mặt:

- đủ lớn để học feature phức tạp
- nhưng cũng nhạy với regime train và dễ drift nếu objective/scheduler không hợp

---

## 7. Search và cách dùng evaluator

Tệp chính:

- [search/negamax.py](/C:/Users/USER/Desktop/chess_engine/search/negamax.py)

Search hiện tại đã có nhiều heuristic engine truyền thống:

- iterative deepening
- alpha-beta / PVS
- quiescence
- transposition table
- null-move pruning
- SEE
- killer moves
- history heuristic
- LMR

Điểm quan trọng:

- search đang dùng **hybrid evaluation**
- kết hợp `material_evaluate` và `nn_eval`

Đây là một thiết kế đúng hướng cho engine thực dụng, vì:

- evaluator neural không cần ôm trọn toàn bộ nhiệm vụ
- search và material eval có thể sửa phần tactical / obvious

---

## 8. Những vấn đề chính đã được xác minh

### 8.1. Objective mismatch trong training từng là nguyên nhân lớn

Theo [training_audit_report.md](/C:/Users/USER/Desktop/chess_engine/docs/training_audit_report.md), phiên bản notebook cũ từng có vấn đề:

- train tối ưu ở `logit-space`
- nhưng val / chọn checkpoint lại đo ở `y-space`

Hệ quả:

- `z_mse` tốt hơn
- nhưng `y_mse` xấu đi
- gây calibration drift

Đây là nguyên nhân đã được xác minh trực tiếp bằng checkpoint thật.

### 8.2. Mô hình vẫn under-confident ở tails

Bucket analysis gần đây cho thấy:

- total MSE đã tốt hơn bản cũ
- nhưng lỗi lớn nhất vẫn nằm ở decisive buckets
- mô hình vẫn co biên độ dự đoán ở vùng thắng/thua rõ

Ví dụ:

- extreme target khoảng `±0.964`
- prediction chỉ khoảng `±0.63 ~ ±0.65`

Nghĩa là:

- model đúng dấu
- nhưng gain còn thấp

### 8.3. Vùng gần cân bằng vẫn rất nguy hiểm

Một kết quả test rất quan trọng:

- trên subset `|y| <= 0.2`
- model có `R2 < 0`
- scatter plot cho thấy dự đoán bị phân tán quá mạnh quanh vùng `y ≈ 0`

Đây là dấu hiệu nguy hiểm vì:

- static evaluator đáng tin phải dè dặt quanh vị trí cân bằng
- hiện tại model vẫn có xu hướng quá tự tin ở nhiều position gần cân bằng

### 8.4. “Static subset” hiện đang bị định nghĩa chưa đủ sạch

Một điểm cần phân biệt rõ:

- `|y| <= 0.2` không đồng nghĩa với “thế cờ tĩnh”

Nó có thể gồm:

- vị trí cân bằng yên tĩnh thật
- hoặc vị trí tactical, chưa search đủ sâu, đang gần cân bằng tạm thời

Nghĩa là label hiện tại vẫn trộn lẫn:

- static meaning
- search volatility

Đây là một nguồn nhiễu nền rất lớn.

### 8.5. Kiến trúc hiện tại có thể chưa có inductive bias tốt nhất cho “static meaning”

Mạng hiện tại mạnh hơn ở:

- direction / sign
- trạng thái advantage rõ

Nhưng yếu hơn ở:

- độ tin cậy quanh `0`
- calibration kiểu “drawish / neutral / stable”

Điều này gợi ý rằng:

- kiến trúc hiện tại có capacity
- nhưng chưa chắc có inductive bias tốt kiểu NNUE cho static eval

---

## 9. Những vấn đề tồn tại nổi bật

### Vấn đề 1. Target hiện tại không hoàn toàn khớp mục tiêu engine

Mạng đang học một scalar search-derived, đã qua `tanh(cp/600)`.

Nhưng engine thực sự cần:

- một evaluator đáng tin cho search
- không nhất thiết phải khớp tuyệt đối full-MSE ở mọi bucket

Điều này làm metric train/test hiện tại chưa chắc phản ánh đúng engine strength.

### Vấn đề 2. Extreme tails vẫn đốt loss rất mạnh

Ngay cả sau các cải tiến, decisive buckets vẫn chiếm phần lớn lỗi.  
Nếu vẫn lấy tổng MSE làm mục tiêu chính, model sẽ luôn bị kéo vào việc sửa tails.

### Vấn đề 3. Near-zero reliability còn yếu

Đây là rủi ro lớn nhất về mặt engine behavior:

- ở những vị trí đáng lẽ phải gần trung lập
- model vẫn có thể dự đoán lệch khá xa

Điều này nguy hiểm hơn việc tails chưa đủ chính xác.

### Vấn đề 4. Kiến trúc lớn nhưng task chưa “sạch”

Mạng có nhiều tham số, nhưng:

- input không chứa mọi thông tin động
- target lại mang tính search
- nên đây là bài toán có giới hạn thông tin

Thêm block chưa chắc giải quyết được gốc rễ.

### Vấn đề 5. Trong repo vẫn còn sự phân tách giữa nhánh mới và nhánh cũ

Hiện có song song:

- `model/architecture`
- `model/architecture_v2`

và một số chỗ runtime/search vẫn tham chiếu nhánh cũ hoặc artifact cũ.  
Đây là vấn đề kỹ thuật của dự án, vì dễ gây lệch giữa:

- notebook train
- checkpoint thật
- model được dùng trong engine

---

## 10. Hướng xử lý hợp lý tiếp theo

### Hướng 1. Chuyển sang mục tiêu residual trên nền classical/material eval

Đây là hướng có tính thực dụng cao nhất.

Ý tưởng:

- classical eval lo phần obvious
- network học phần residual khó

Lợi ích:

- giảm burden ở tails cực trị
- neo tốt hơn quanh vùng cân bằng
- phù hợp với engine có search

### Hướng 2. Dùng metric và loss theo hướng engine-centric

Thay vì chỉ nhìn full-MSE, nên theo dõi thêm:

- weighted MSE tập trung vào vùng `|y| <= 0.8`
- bucket MSE ở vùng `0.3 - 0.7`
- reliability quanh `0`
- Elo thực tế ở fixed nodes/time

### Hướng 3. Bổ sung head dự đoán confidence / volatility

Nếu mục tiêu là evaluator đáng tin:

- một scalar duy nhất là chưa đủ

Nên cân nhắc:

- value head
- confidence / dynamicity head

để search biết khi nào nên tin NN, khi nào nên tin search/classical hơn.

### Hướng 4. Làm sạch khái niệm “static position”

Không nên dùng `|y| nhỏ` làm proxy duy nhất cho “tĩnh”.

Nên định nghĩa thêm yếu tố stability, ví dụ:

- độ ổn định qua nhiều depth
- độ biến động eval
- tactical volatility

### Hướng 5. Chỉ giảm block sau khi task đã được sửa đúng hơn

Nếu target/objective đúng hơn, rất có thể:

- model nhỏ hơn vẫn đủ tốt
- tốc độ eval tăng
- Elo thực tế cao hơn nhờ search sâu hơn

Nhưng giảm block quá sớm, trước khi sửa task, sẽ chỉ làm khó phân tích nguyên nhân hơn.

---

## 11. Kết luận

Trạng thái hiện tại của dự án có thể tóm tắt như sau:

- Đây là một chess engine có nền tảng kỹ thuật tốt và khá đầy đủ ở phần search.
- Pipeline dữ liệu và huấn luyện đã tiến hóa đáng kể, đặc biệt ở nhánh `architecture_v2`.
- Kiến trúc mới đã sửa được nhiều vấn đề của head cũ và cho kết quả tốt hơn rõ rệt.
- Tuy nhiên, hệ hiện tại vẫn chưa đạt độ tin cậy cần thiết ở vùng gần cân bằng và chưa giải quyết triệt để lỗi ở decisive buckets.
- Vấn đề lớn nhất bây giờ không còn là một bug đơn lẻ, mà là:
  - target chưa khớp hoàn toàn với mục tiêu engine
  - evaluator tĩnh đang phải học một đại lượng có thành phần search/noise
  - kiến trúc hiện tại chưa có bias đủ mạnh cho static reliability kiểu NNUE

Nếu mục tiêu là xây một evaluator mạnh, tổng quát tốt và đáng tin cho engine thực dụng, hướng đáng làm nhất hiện nay là:

1. làm rõ lại target theo hướng **residual-to-classical**
2. đánh giá theo **metric bám sát engine behavior**
3. chỉ sau đó mới tiếp tục tối ưu kích thước mô hình, distillation và quantization
