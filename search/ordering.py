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
SCORE_KILLER_1 = 1100000   # Killer move slot 1
SCORE_KILLER_2 = 1050000   # Killer move slot 2
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

def get_move_priority(move, board, tt_move=None, ply=0, killer_moves=None, history_table=None):
    """
    Tính điểm ưu tiên cho move ordering.
    Thứ tự: TT > Good Captures > Killers > Quiet với History
    """
    # 1. TT / PV MOVE - Ưu tiên cao nhất
    if tt_move is not None and move == tt_move:
        return SCORE_TT_MOVE

    is_capture = move.is_capture(board)
    is_promo = move.is_promotion()
    
    # 2. Promotions - Ưu tiên rất cao
    if is_promo:
        promo_bonus = 0
        if move.promotion == PieceType.QUEEN:
            promo_bonus = 900
        elif move.promotion == PieceType.KNIGHT:
            promo_bonus = 300  # Under-promotion có thể là mate
        return SCORE_THREAT_PROMO + promo_bonus
    
    # 3. Captures - Phân loại bằng SEE
    if is_capture:
        delta = calculate_delta_eval(board, move)
        see_good = see_capture(board, move, threshold=0)
        
        victim = board.squares[move.to_square]
        attacker = board.squares[move.from_square]
        victim_val = 100 if move.is_en_passant else (PIECE_VALUES[victim.type] if victim else 0)
        attacker_val = PIECE_VALUES[attacker.type] if attacker else 0
        mvv_lva = victim_val * 10 - attacker_val
        
        if see_good:
            return SCORE_CAPTURE_GOOD + delta + mvv_lva
        else:
            return SCORE_CAPTURE_WEAK + delta
    
    # 4. Killer Moves - Nước yên tĩnh đã gây cutoff ở ply khác
    if killer_moves is not None and ply < len(killer_moves):
        if move == killer_moves[ply][0]:
            return SCORE_KILLER_1
        if move == killer_moves[ply][1]:
            return SCORE_KILLER_2
    
    # 5. Quiet Moves - Sắp xếp theo History Heuristic + PST
    delta = calculate_delta_eval(board, move)
    history_score = 0
    
    if history_table is not None:
        color_idx = 0 if board.current_turn == Color.WHITE else 1
        history_score = history_table[color_idx][move.from_square][move.to_square]
    
    return SCORE_QUIET + delta + history_score


def sort_moves_priority(moves, board, tt_move=None, ply=0, killer_moves=None, history_table=None):
    """Sắp xếp nước đi theo điểm ưu tiên"""
    scored_moves = []
    for move in moves:
        score = get_move_priority(move, board, tt_move, ply, killer_moves, history_table)
        scored_moves.append((score, move))
    
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored_moves]