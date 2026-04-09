# Hướng dẫn và Giải thích Metrics trong `history.csv` (FT1 Pipeline)

Tài liệu này giải thích chi tiết ý nghĩa của các metric được lưu lại trong file `history.csv` sau mỗi epoch huấn luyện, cách đọc hiểu chúng và cách đánh giá xem mô hình có đang hội tụ đúng hướng hay không. Pipeline hiện tại sử dụng **PCGrad** (Gradient Surgery) để cân bằng giữa các mục tiêu (main objective và aux objective).

---

## 1. Các Metrics Cốt lõi về Quá trình Huấn luyện (Training Metrics)

Các metric này phản ánh quá trình tối ưu hóa loss trên tập training.

- **`train_main_objective` / `train_aux_objective`**: 
  - **Ý nghĩa**: Giá trị loss trung bình của mục tiêu chính (main) và mục tiêu phụ (aux) trong quá trình huấn luyện.
  - **Đánh giá**: Cả hai cần giảm dần qua các epoch. Nếu `main` giảm nhưng `aux` tăng mạnh, có thể đang xảy ra xung đột (gradient conflict).
- **`train_clean_loss` / `train_ambiguous_loss`**:
  - **Ý nghĩa**: Sự phân cực của loss dựa trên trạng thái thế cờ rõ ràng (clean) và nhập nhằng/phức tạp (ambiguous).
  - **Đánh giá**: `clean_loss` thường hội tụ nhanh và ổn định hơn. Sự giảm của `ambiguous_loss` chứng tỏ mô hình bắt đầu học được các tình huống khó thay vì chỉ học thuộc các trạng thái dễ.
- **`train_margin_penalty`, `train_mean_main_weight`, `train_downweighted_frac`**:
  - **Ý nghĩa**: Các chỉ số liên quan đến việc phạt (penalty) hoặc giảm trọng số (downweight) đối với các sample nhiễu (outliers) nhằm tăng tính ổn định (robustness) của mô hình.
- **Các metric về Gradient (PCGrad)**: `grad_cosine_backbone`, `grad_conflict_backbone`, `grad_norm_main_backbone`, `grad_norm_aux_backbone`, v.v.
  - **Ý nghĩa**: Đo lường sự đồng thuận giữa gradient của main loss và aux loss.
  - **Đánh giá**: Nếu `grad_conflict_backbone` cao (mức độ xung đột lớn), PCGrad sẽ can thiệp (`grad_cosine_backbone_post` sẽ được điều chỉnh) để tránh việc cập nhật trọng số làm triệt tiêu lẫn nhau. Sự ổn định của các norm gradient (`grad_norm...`) cho thấy mô hình không gặp hiện tượng vanishing hoặc exploding gradients.

---

## 2. Các Metrics trên Tập Kiểm tra (Test/Validation Metrics)

Đánh giá tổng quát khả năng dự đoán điểm số thế cờ của mô hình (thường tính bằng MSE hoặc đo lường trên các ngách dữ liệu).

- **`test_mse_0.1eq`, `test_mse_0.2eq`, `test_mse_0.5eq`, `test_mse_0.7eq`**:
  - **Ý nghĩa**: MSE (Mean Squared Error) phân chia theo giới hạn (band) của nhãn (ví dụ: các vị trí có điểm cực kì cân bằng `<= 0.1`, hoặc điểm chênh lệch cao `<= 0.7`).
  - **Đánh giá**: Khi các giá trị này giảm, mô hình ngày càng chính xác ở nhiều dải điểm khác nhau. Đặc biệt quan tâm dải `0.1eq` và `0.2eq` vì đây là vùng trung tâm của bàn cờ (thế cờ cân bằng).
- **`test_slope_0.1eq`, `test_slope_0.2eq`, `test_slope_0.7eq`**:
  - **Ý nghĩa**: Độ dốc (Slope) của dự đoán so với nhãn thực tế ở các vùng điểm tương ứng. Độ dốc lý tưởng là 1.0.
  - **Đánh giá**: Nếu độ dốc quá thấp (< 0.5), mô hình đang quá an toàn (bị kéo về 0 đối với các nhãn lớn), gọi là hiện tượng *under-confident*. Nếu độ dốc gần L4 Reference, tức là mô hình học được đúng xu hướng biên độ và không bị "phẳng" ở các dải điểm.
- **`test_center_false_0.1eq`, `test_center_wrong_sign_0.1eq`, `test_center_spread_ratio`**:
  - **Ý nghĩa**: Đo lường mức độ sai lệch hoặc lạc hướng (wrong sign - dự đoán nhầm bên thắng), dự đoán sai (false) trong khoảng điểm cân bằng ở giữa bàn cờ (center).
  - **Đánh giá**: Càng nhỏ càng tốt. Mô hình tốt không được phép dự đoán chênh lệch lớn ở các thế cờ có giá trị thực là 0 (hoà hoặc hai bên cân bằng).

---

## 3. Quỹ đạo Oracle và Các Metrics Quyết định Hội tụ (Oracle & Gating Metrics)

Đây là thước đo quan trọng nhất để xem mô hình có đạt đủ tiêu chuẩn vượt qua bài kiểm tra chốt chặn (Gates) hay không.

- **`oracle_stable_0.7_slope`**:
  - **Vai trò**: Đại diện cho "Gate Slope". 
  - **Đánh giá Hội tụ**: Phải vượt ngưỡng của Baseline (L4 - 0.02). Nghĩa là mô hình duy trì được độ nhạy (chênh lệch điểm) với các thay đổi thực tế của đối thủ, không dự đoán bằng phẳng (flat). Nếu giá trị này tăng tịnh tiến đều qua mỗi epoch là dấu hiệu mô hình đang "sống" và học đúng hướng.
- **`oracle_midband_mae_sum_stable`**:
  - **Vai trò**: Đại diện cho "Gate Midband". Là tổng Error ở vùng không quá căng thẳng và không quá nhàm chán.
  - **Đánh giá Hội tụ**: Phải nhỏ hơn ngưỡng `1.05 * L4 Baseline`. Nếu giá trị này thấp xuống hoặc duy trì độ ổn định qua từng vòng, phần thân (backbone) của mô hình đang nắm bắt được các cấu trúc thế cờ chủ chốt.
- **`oracle_teacher_mae`, `oracle_closer_rate`**:
  - Khả năng mô hình "bắt chước" theo hướng dự đoán từ vị trí Teacher model, closer_rate càng lớn thì độ định hướng tới đáp án đúng càng cao.
- **`center_score`**:
  - **Vai trò**: Chỉ số an toàn cuối cùng phối hợp từ các thước đo lỗi trung tâm, biên độ (amp), sai số nhỏ (false 0.1, 0.2). Công thức thường là:
    `center_score = center_mae + 0.3 * (false_0.1) + 0.2 * (false_0.2) + ...`
  - **Đánh giá Hội tụ**: Mức **cực kỳ quan trọng**. Điểm này đặc trưng cho việc mạng học có "an toàn" không khi đối mặt với một state rỗng/cân bằng, mô hình không được phép sinh ra các giá trị thiên vị 1 bên. Volatility (độ lệch chuẩn dao động) của lượng này giảm dần kèm theo sự giảm về lượng trung bình có nghĩa là mô hình hội tụ tốt.
- **Phân tách Hạng mục (Pooled / Clean / Ambiguous)**:
  - `clean_center_mae` / `ambiguous_center_mae`
  - Giúp phát hiện nhanh mô hình đang cải thiện center nhờ học thuộc các phần rõ ràng (`clean_center`) hay thật sự thẩm thấu được hàm giá trị phức tạp (`ambiguous_center`). Một mô hình đang tiệm cận sự tối ưu hoàn toàn khi cả `ambiguous_center_mae` cũng giảm (dù nó khó hơn rất nhiều so với clean).

---

## Tóm tắt: Làm thế nào để biết Mô hình Đang Hội tụ Đúng Hướng?

1. **Hiệu suất cơ sở (Midband MAE):** Nhìn vào `oracle_midband_mae_sum_stable`. Nếu giá trị này nhỏ hơn Threshold L4 ngay từ Epoch thứ 3-5 trở đi, có nghĩa là mô hình đã tạo ra được định hình chung đúng, không bị chệch khỏi distribution của dữ liệu cờ vua chuẩn.
2. **Khuynh hướng hồi quy (Slope):** Nhìn vào `oracle_stable_0.7_slope`. Metric này thường xuất phát rất thấp và leo dần lên trên. Chừng nào đường dốc (trendline) này còn dương, mô hình vẫn đang học tiếp, chưa bão hòa. Khi nào Slope vượt qua Threshold, mô hình thoát khỏi giai đoạn "sợ sệt an toàn" và đưa ra dự đoán có biên độ tự tin hơn.
3. **An toàn vùng Cân Bằng (Center):** Nhìn vào `center_score` và `pooled_center_false_0.1eq`. Những giá trị này phải giảm và càng về sau càng ít nhảy cóc linh tinh (độ biến động giảm). 
4. **Giảm xung đột nội bộ:** `grad_norm_main` và `grad_norm_aux` phải có độ lớn nằm trong biên độ kiểm soát được, không bị giật (spike). Mức giảm `cosine_backbone` và kết quả hậu can thiệp PCGrad cho thấy tối ưu hoá đã trơn tru hơn.

**Kết luận:** Quá trình huấn luyện không chỉ đánh giá qua việc MSE giảm. Ở dự án này với yêu cầu "Regression Mạng Đánh Giá Trạng Thái Cờ", chúng ta hướng vào Gating (qua được Midband, qua được Slope) và An Toàn Trung Tâm. Khi cả hai Gate này hòm hòm đạt chuẩn và xu hướng Center thu gọn, nghĩa là **Mô Hình Đang Hội Tụ Đúng Hướng**.
