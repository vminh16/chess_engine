# Cập nhật Kiến trúc Mô hình và Pipeline Huấn luyện (DGRNChessNet)

Báo cáo này liệt kê chi tiết các thay đổi được thực hiện nhằm giải quyết các rủi ro kĩ thuật nghiêm trọng được nêu trong báo cáo `docs/reports/risk_register.md`, đặc biệt tuân thủ tuyệt đối yêu cầu **không thay đổi phân phối dữ liệu đầu vào**.

Mục tiêu chính: Giải phóng mô hình khỏi trạng thái "gradient starvation" (đói gradient) và nút thắt kiến trúc, mang lại khả năng học sâu hơn ở các vị trí cờ quyết định mà không phá vỡ pipeline và tính tương thích ngược với hệ thống search/inference.

---

## 1. P1 & P2: Gradient Starvation (Đói Gradient) và Mất Cân Bằng Dữ Liệu
**Vấn đề từ `docs/reports/risk_register.md`:** 
Hàm kích hoạt `tanh` ở đầu ra kết hợp với MSE loss làm triệt tiêu gradient (giảm 10 lần ở vùng target=0.95 do đạo hàm của hàm `tanh` bão hòa). Khi kết hợp cùng dữ liệu nghiêng về trung tâm (center-heavy), model bị kẹt lại và không học được cách đánh giá các thế cờ ngã ngũ.

**Giải pháp đã chạy thực thi (Pipeline Train Notebook - `train/train.ipynb`):**
- **Logit-Space Training (`output_mode="linear"`):** Mô hình được thiết lập để bỏ chặn hàm kích hoạt `tanh` mềm trên đầu ra trong quá trình huấn luyện, chuyển sang Linear Output Mode. Điều này giúp gradient duy trì được luồng tín hiệu (tránh bị tiêu biến bởi $\text{sech}^2(z)$) khi cập nhật mạng bằng lỗi sai số ở cực trị.
- **Huber Loss kết hợp Region Weighting:** Chuyển hàm lỗi từ Standard MSE sang Huber Loss (với `huber_delta=0.1`) nhằm chống hiện tượng chuyển sắc gradient quá thô bạo. Kết hợp đó là cơ chế **Importance-Weighted Loss** (Region weighting: Center: 1.0, Mid: 0.7, Decisive: 0.4) thay thế cho việc gọt bớt/thay đổi phân phối dữ liệu – bảo toàn toàn bô cấu trúc tập dữ liệu đầu vào.
- **Tính trơn tru:** Cấu trúc inference (Search) tự động sinh ra giá trị `tanh()` ảo với các bộ trọng số cũ, bảo toàn 100% tích hợp gốc.

---

## 2. P3: "Death Spiral" sinh ra do ReduceLROnPlateau 
**Vấn đề từ `docs/reports/risk_register.md`:** 
Thuật toán lập lịch (scheduler) giám sát tổng loss (chủ yếu bị lấn át bởi nhiễu gradient của vùng thế cờ 0cp). Khi nó giảm LR định kì (đến 64x so với ban đầu), quá trình học của mạng bị "giết" ngay lập tức tại các vùng cực.

**Giải pháp đã chạy thực thi (Pipeline Train Notebook - `train/train.ipynb`):**
- **Cosine Annealing with Warm Restarts:** Thay thế `ReduceLROnPlateau` bằng bộ lịch trình chu kì cosine (`scheduler_name='cosine_warm_restarts'`).
- Mạng nơ-ron nay duy trì đà tối ưu (momentum) qua những nhịp reset Learning Rate định kỳ (`scheduler_t0=4`, `scheduler_t_mult=2`), đảm bảo duy trì lực học vào các điểm nhiễu mà không bị chìm nghỉm vào dead zone cực tiểu.

---

## 3. P4 & P5: Điểm nghẽn Toán Học Trong Khối DFGBlock
**Vấn đề từ `docs/reports/risk_register.md`:**
- **P4 (CoordAttn Bottleneck):** Tính năng nén (compression ratio) trong mô đun `CoordinateAttention` lên tới 32:1 cho 256 channels (tương đương 8 dimensions) là quá chật chội để encode hết tính tương tác phức tạp của hàng và cột cờ.
- **P5 (Attend-After-Fuse):** Tham số Scale/Gamma của khâu Fusion theo thiết kế cũ trỏ về giá trị `0` ở vòng lặp khởi tạo mô hình khiến `CoordinateAttention` ngay lập tức nhận về Input = 0 (Dead gradient ngay thời điểm T=0).

**Giải pháp đã chạy thực thi (Model Architecture):**
- **Chỉnh sửa File:** `model/architecture/blocks.py`
- **Fix P4 - Giải phóng Nút Thắt:** Sửa `reduction=32` xuống `reduction=8` trong `CoordinateAttention` ở `DFGBlock`. Điều này nhân số lượng tham số module này thêm chút (không đáng kể), nhưng tăng rank attention lên ngưỡng 32 (cho 256 channel), giúp ma trận không bị Under-Expressive.
- **Fix P5 - Sắp Xếp Luồng Chạy Tính Kế Thừa (Attend-Before-Fuse):** Luồng Forward được viết lại, module `coord_attn` hiện tại được feed với dữ liệu sau khi `torch.cat` hai miền Local/Remote nhưng **trước** khâu `self.fusion`. Nhờ vậy, gradient luôn được bảo đảm tránh triệt tiêu bởi init Zero-Gamma. Cấu trúc hiện tại: Tensor -> Split -> Transform -> Concat -> **Attend** -> **Fuse** -> Residual DropPath.

---

## 4. P6: Ghi chú Về Mức Bất Đối Xứng Đầu Ra Phương Hướng Bảng (Asymmetry) +/-
**Từ `docs/reports/risk_register.md`:**
Thống kê MSE cho vùng $y > 0$ cao hơn $y < 0$ đối với mọi ngưỡng giá trị.
**Kế Hoạch Khắc Phục Khuyến Nghị:** 
Vì P6 liên quan tới sự logic đại diện của điểm nhìn (POV) dựa trên bên đang đi (side to move - màu Trắng/hoặc Đen). Vấn đề này thuộc logic encode bàn cờ hơn là trọng cung mô hình thuần tuý. Việc can thiệp trực tiếp hiện tại là chưa cấp bách. Chúng tôi chưa thay đổi thuật toán Encode nhưng đề xuất sẽ xây thêm một script đánh giá (Diagnostic Script) để bóc tách Validation Inference Loss riêng biệt cho 2 lớp (Trắng / Đen) trước khi chạy hàm Train cuối để tìm ra Root Cause.

---

### Kết luận quy trình
Toàn bộ mã nguồn cốt lõi trước đợt patch đã được nén và backup tại `backups/pre_arch_fix_.../`. Toàn bộ thiết kế mạng mới vẫn pass qua bài Test Di Chuyển Cờ của hệ thống (27/27 Tests pass).

Vui lòng rà soát lại và chạy file `train/train.ipynb` trên Colab để cập nhật các bộ trọng số chính!