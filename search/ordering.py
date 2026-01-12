# search/ordering.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.constants import PieceType, Color
from evaluation.static_eval import PIECE_VALUES  # Import từ file static_eval đã tách
from core.rules import is_in_check
from search.see import see_capture

# Các hằng số điểm ưu tiên (Priority Scores)
SCORE_TT_MOVE = 2000000
SCORE_THREAT_MATE = 1500000
SCORE_THREAT_PROMO = 1400000
SCORE_THREAT_ATTACK = 1200000
SCORE_DEFENSE = 900000
SCORE_CAPTURE_GOOD = 600000
SCORE_CAPTURE_NEUTRAL = 300000
SCORE_QUIET = 100000
SCORE_CAPTURE_WEAK = -500000
SCORE_IGNORE = -10000000

# --- SIMPLIFIED PST FOR ORDERING ONLY ---
# Chỉ dùng để định hướng sơ bộ ("Knight should go center", "King stay safe")
# Không dùng cho Evaluation chính xác.

# Giả định: Mailbox 0=a8 (Top-Left), 63=h1 (Bottom-Right).
# Bảng viết theo hướng nhìn của White (Top=Rank 8, Bottom=Rank 1).

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

# King Midgame: Tránh trung tâm, ở đáy an toàn hơn
PST_KING_MID = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

def get_pst_ordering_value(piece_type, square, color):
    """
    Lấy giá trị PST thô cho việc sắp xếp nước đi.
    Chỉ hỗ trợ Knight và King để tối ưu tốc độ.
    """
    table = None
    if piece_type == PieceType.KNIGHT:
        table = PST_KNIGHT
    elif piece_type == PieceType.KING:
        table = PST_KING_MID
    else:
        return 0 # Các quân khác không dùng PST cho ordering delta
    
    # Xử lý tọa độ (Orientation Fix)
    # - Nếu là White: Giữ nguyên như bảng.
    # - Nếu là Black: Cần Mirror dọc (Flip Vertical).
    #   a8 (0) của bàn cờ đóng vai trò là a1 của Black.
    
    idx = square
    if color == Color.BLACK:
        idx = square ^ 56 # Flip chiều dọc (0 <-> 56)
        
    return table[idx]

def calculate_delta_eval(board, move):
    """
    Tính Delta Eval "Thô & Rẻ" (Rough & Cheap).
    Mục đích: Chỉ để phân loại Good/Bad capture nhanh chóng.
    Công thức: Material Gain + (Simple Positional Bias for Knights/King)
    """
    # 1. Material Gain (Quan trọng nhất)
    # Tối ưu: Không check board.squares cho captured nếu không cần thiết
    # Nhưng logic capture cần biết loại quân.
    
    mat_delta = 0
    
    if move.is_capture(board):
        # En Passant
        if move.is_en_passant:
            mat_delta = PIECE_VALUES[PieceType.PAWN]
        else:
            # Capture thường
            captured = board.squares[move.to_square]
            if captured:
                mat_delta = PIECE_VALUES[captured.type]
    
    # Promotion: Cộng thẳng giá trị chênh lệch (ví dụ Queen - Pawn)
    if move.is_promotion():
        mat_delta += PIECE_VALUES[move.promotion] - PIECE_VALUES[PieceType.PAWN]

    # 2. Positional Bias (Rất nhẹ, chỉ cho Knight/King)
    # Không tính cho Pawn/Bishop/Rook/Queen để tiết kiệm CPU
    # Chỉ tính khi KHÔNG phải capture (Quiet Moves) hoặc Capture cân bằng
    # Để break-tie.
    
    pos_delta = 0
    piece = board.squares[move.from_square]
    
    # Chỉ tính positional delta cho Knight và King
    if piece and piece.type in [PieceType.KNIGHT, PieceType.KING]:
        val_from = get_pst_ordering_value(piece.type, move.from_square, piece.color)
        val_to = get_pst_ordering_value(piece.type, move.to_square, piece.color) # Lưu ý: type có thể đổi nếu promo, nhưng ở đây chỉ xét K/N
        pos_delta = val_to - val_from

    # Trả về tổng: Material là chính, Position là phụ (bias)
    return mat_delta + pos_delta

def get_king_distance(board, square, color):
    """Tính khoảng cách Manhattan đến vua đối phương"""
   
    king_sq = -1
    target_color = Color.BLACK if color == Color.WHITE else Color.WHITE
    
    for i, p in enumerate(board.squares):
        if p and p.type == PieceType.KING and p.color == target_color:
            king_sq = i
            break
            
    if king_sq == -1: return 99
    
    r1, c1 = square // 8, square % 8
    r2, c2 = king_sq // 8, king_sq % 8
    return max(abs(r1 - r2), abs(c1 - c2))

def get_move_priority(move, board, tt_move=None):
    # ... (Giữ nguyên các hằng số SCORE...)
    
    # 1. TT / PV MOVE
    if tt_move is not None and move == tt_move:
        return SCORE_TT_MOVE

    is_capture = move.is_capture(board)
    is_promo = move.is_promotion()
    
    # Check threat sơ bộ (Optional: Nếu quá chậm có thể bỏ qua phần apply_move này cho QS)
    # Ở đây giữ lại logic Threat cho Main Search Ordering
    gives_check = False
    # board.apply_move(move) ... (Code check threat của bạn) ...
    
    
    # Để code chạy nhanh, ta tạm bỏ qua phần check threat nặng nề nếu không cần thiết
    # Hoặc chỉ check nếu là Quiet Move.
    
    # --- LOGIC MỚI CHO DELTA ---
    
    # Tính Delta Eval (Thô)
    delta = calculate_delta_eval(board, move)
    
    # Nếu là Capture
    if is_capture:
        # SEE (Vẫn cần thiết để lọc nước lỗ)
        see_good = see_capture(board, move, threshold=0)
        
        # MVV-LVA (Tính nhanh)
        victim = board.squares[move.to_square]
        attacker = board.squares[move.from_square]
        victim_val = 100 if move.is_en_passant else (PIECE_VALUES[victim.type] if victim else 0)
        attacker_val = PIECE_VALUES[attacker.type] if attacker else 0
        mvv_lva = victim_val * 10 - attacker_val
        
        # Phân loại dựa trên SEE và Delta
        if see_good:
            if delta >= 0:
                return SCORE_CAPTURE_GOOD + delta + mvv_lva
            else:
                # SEE tốt (trao đổi an toàn) nhưng Delta âm (ví dụ thí quân chiến thuật được SEE bảo kê?)
                # Hoặc đơn giản là Capture Trung tính
                return SCORE_CAPTURE_NEUTRAL + mvv_lva
        else:
            # SEE < 0 (Lỗ)
            # Logic: Đẩy xuống cuối (Bad capture)
            # Đừng trả về SCORE_IGNORE ở đây nếu muốn Dynamic Threshold bên ngoài xử lý
            # Trả về điểm rất thấp
            return SCORE_CAPTURE_WEAK + delta # Delta âm sẽ kéo điểm xuống thêm

    # Non-capture (Quiet)
    # Dùng delta (chủ yếu là pos_delta của Knight/King) để sort nhẹ
    return SCORE_QUIET + delta

def sort_moves_priority(moves, board, tt_move=None):
    scored_moves = []
    for move in moves:
        # Lưu ý: Cần tối ưu việc gọi get_move_priority đừng apply_move quá nhiều
        score = get_move_priority(move, board, tt_move)
        scored_moves.append((score, move))
    
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored_moves]