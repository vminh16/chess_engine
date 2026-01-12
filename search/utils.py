from core.rules import is_in_check
from core.constants import Color, PieceType
import math

def is_forced_move(board, move):
    """
    Trả về True nếu nước đi quá quan trọng, không được phép Prune bằng SEE.
    """
    # 1. Chúng ta đang bị Chiếu (Check Evasion)
    # Nếu đang bị chiếu, MỌI nước đi hợp lệ đều là Forced (để thoát chiếu).
    # Trong QS, thường ta sinh tất cả evasions chứ không chỉ captures.
    # Logic kiểm tra:
    if is_in_check(board, board.current_turn):
        return True

    # 2. Nước đi gây ra Chiếu (Give Check)
    # Đây là nước tấn công mạnh, nên giữ lại để tìm Mate.
    board.apply_move(move)
    gives_check = is_in_check(board, board.current_turn) # Turn đã đổi
    board.undo_move()
    if gives_check:
        return True

    # 3. Phong cấp (Promotion)
    if move.is_promotion():
        return True
        
    # 4. Ngăn chặn đối thủ Phong cấp (Prevent Promotion)
    # Nếu đối thủ có Tốt ở hàng 7 (hoặc 2 với Trắng), và nước đi này bắt nó.
    # Hoặc nước đi này block đường phong cấp (phức tạp hơn, tạm thời chỉ xét bắt quân).
    if move.is_capture(board):
        captured = board.squares[move.to_square]
        if captured and captured.type == PieceType.PAWN:
            # Kiểm tra hàng
            row = move.to_square // 8
            if captured.color == Color.WHITE and row == 6: # Tốt trắng ở hàng 7
                return True
            if captured.color == Color.BLACK and row == 1: # Tốt đen ở hàng 2
                return True
    
    # 5. Perpetual Check (Chiếu vĩnh viễn/Cầu hòa)
    # Nếu nước đi này tiếp tục chiếu liên tục, có thể dẫn đến hòa.
    # Tạm thời không xét kỹ trường hợp này.

    return False


def calculate_lmr_reduction(move_index, depth, move, board, is_capture, 
                            see_result, is_in_check_status, nn_eval, alpha, beta):
    """
    Tính toán độ sâu cần giảm (Reduction R) cho LMR.
    Trả về 0 nếu không LMR.
    """
    
    # 1. ĐIỀU KIỆN KHỞI ĐẦU (Prerequisites)
    # Bắt đầu LMR từ move index >= 4 (tức là nước thứ 5 trở đi)
    if move_index < 4:
        return 0
    
    # Độ sâu còn lại phải đủ lớn mới bõ công giảm (thường > 2)
    if depth < 3:
        return 0

    # Không LMR nếu đang bị chiếu (Forced reply)
    if is_in_check_status:
        return 0

    # Không LMR cho nước Phong cấp
    if move.is_promotion():
        return 0

    # 2. XÁC ĐỊNH MỤC TIÊU ÁP DỤNG (Target Selection)
    # Áp dụng cho: Non-capture HOẶC Capture yếu (SEE <= 0)
    is_weak_capture = is_capture and (see_result <= 0)
    
    # Nếu là Good Capture (ăn quân lời) -> Không giảm
    if is_capture and not is_weak_capture:
        return 0
    
    # 3. TÍNH TOÁN REDUCTION CƠ BẢN (Base Reduction)
    # Phụ thuộc move index (chính) và depth.
    # Công thức Logarit giúp tăng mượt mà: R = 1 + log(depth) * log(index) / constant
    # Constant = 2.5 là giá trị phổ biến để tinh chỉnh.
    R = 1.0 + (math.log(depth) * math.log(move_index) / 2.5)
    
    # 4. ĐIỀU CHỈNH THEO NN EVAL (NN Adjustment)
    # nn_eval ở đây là đánh giá tĩnh của node hiện tại (trước khi đi quân)
    # Logic: 
    # - Nếu thế cờ đang rất tốt (NN > Beta): Engine tự tin -> Giảm mạnh tay hơn.
    # - Nếu thế cờ đang xấu (NN < Alpha): Engine lo lắng -> Giảm nhẹ đi (để tìm kỹ hơn).
    
    # Normalize nn_eval về đơn vị centipawn hoặc tương đương để so sánh
    # Giả sử nn_eval nằm trong khoảng [-1, 1] như code cũ của bạn -> scale lên 100-300 cho dễ so sánh
    # Hoặc nếu nn_eval của bạn đã là điểm số kiểu Stockfish (centipawn), dùng trực tiếp.
    
    # Giả định nn_eval đang là float [-1, 1] hoặc score lớn. Hãy dùng ngưỡng tương đối.
    # so sánh trực tiếp với alpha/beta.
    
    if nn_eval > beta + 0.5:  # Thế trận áp đảo
        R += 1.0           # Tăng reduction (mạnh tay)
    elif nn_eval < alpha - 0.5: # Thế trận kém
        R -= 1.0             # Giảm reduction (nhẹ tay / không tắt hẳn)
        
    # Làm tròn và giới hạn an toàn
    R = int(R)
    
    # Đảm bảo không giảm quá mức (luôn để lại ít nhất depth 1 để check tính hợp lệ)
    # R tối thiểu là 1 nếu đã lọt vào đây
    return max(1, min(R, depth - 2))