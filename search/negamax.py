"""
Negamax search algorithm for chess engine.
Simple implementation without alpha-beta pruning, move ordering, or transposition table.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.board import Board
from core.constants import PieceType
from core.move_generator import MoveGenerator
from core.rules import is_in_check
from evaluation.naive import evaluate
from evaluation.nn import NeuralNetwork
from model.architecture.model import PhantomChessNet
from evaluation.static_eval import material_evaluate
from core.constants import Color
from representation.encode import encode_board
from search.transition_table import TranspositionTable
from search.see import see_capture
from search.utils import is_forced_move, calculate_lmr_reduction
from search.ordering import sort_moves_priority
import numpy as np
import time

PATH_DATASET = 'data/data_set_v1/dataset_full.npz'
PATH_MODEL = "model/param_model/PhantomChessNet.pth"

# Initialize transposition table
TT = TranspositionTable()    

# --- Hằng số Search ---
MATE_SCORE = 10000
MATE_THRESHOLD = 9000
MAX_QS_PLY = 100

# Giá trị quân: 0:None, 1:PAWN, 2:KNIGHT, 3:BISHOP, 4:ROOK, 5:QUEEN, 6:KING
PIECE_VALUES = [0, 100, 300, 310, 500, 900, 10000]

# --- Futility Pruning Margins ---
# Margin cho mỗi depth (centipawn). Nếu static_eval + margin < alpha -> prune
FUTILITY_MARGINS = [0, 100, 200, 300, 400]  # depth 0-4

# --- Null Move Pruning ---
NULL_MOVE_REDUCTION = 3  # R = 3 cho Null Move
NULL_MOVE_MIN_DEPTH = 3  # Chỉ áp dụng khi depth >= 3

# --- Killer Moves (2 slots mỗi ply) ---
MAX_PLY = 128
killer_moves = [[None, None] for _ in range(MAX_PLY)]

# --- History Heuristic ---
# history[color][from_sq][to_sq] = score
history_table = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]


def hybrid_evaluate(board, material_eval, epsilon, model, hash_key, tt_entry=None):
    """
    Hàm đánh giá lai: Kết hợp Material và Neural Network.
    Trả về điểm từ góc nhìn người chơi hiện tại (side-to-move relative).
    """
    nn_eval = None
    
    # Thử lấy NN eval từ TT cache
    if tt_entry is not None and tt_entry.get('static_eval') is not None:
        nn_eval = tt_entry['static_eval']
    
    if nn_eval is None:
        nn_eval = model.predict(encode_board(board))
        # Cache NN eval vào TT để tái sử dụng
        TT.store(hash_key, depth=-1, value=0, flag=TranspositionTable.EXACT, 
                 age=board.fullmove_number, static_eval=nn_eval)
    
    # BUG FIX: Kiểm tra epsilon hợp lệ
    epsilon = max(0.0, min(1.0, epsilon))
    
    # Kết hợp material (nhanh) + NN (chính xác)
    # material_eval đã là side-relative từ static_eval.py
    # nn_eval từ model cũng đã được flip trong encode_board()
    hybrid_score = (1 - epsilon) * material_eval + epsilon * nn_eval
    
    # BUG FIX: Chỉ negate nếu material_eval chưa được negate
    # Kiểm tra: material_evaluate() trả về White-relative, nên cần negate cho Black
    if board.current_turn == Color.BLACK:
        hybrid_score = -hybrid_score
    
    return hybrid_score

def sort_move_mvv_lva(move, board: Board, tt=None) -> int:
    """Sắp xếp nước đi theo MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)"""
    
    # BUG FIX: Kiểm tra TT move TRƯỚC - ưu tiên cao nhất
    if tt is not None and tt == move:
        return 1000000
    
    # Phong cấp - ưu tiên rất cao
    if move.is_promotion():
        piece = board.get_piece(move.from_square)
        if piece is None:
            return 90000
        aggressor_value = PIECE_VALUES[piece.type]
        promotion_value = PIECE_VALUES[move.promotion]
        return 90000 + (promotion_value * 10) - aggressor_value

    # Nước đi yên tĩnh (không ăn quân)
    if not move.is_capture(board):
        return 0

    # En passant
    if move.is_en_passant:
        return (100 * 10) - 100  # Tốt ăn tốt

    # Ăn quân thông thường - BUG FIX: Thêm null check
    captured_piece = board.get_piece(move.to_square)
    moving_piece = board.get_piece(move.from_square)
    
    if captured_piece is None or moving_piece is None:
        return 0
    
    victim_value = PIECE_VALUES[captured_piece.type]
    aggressor_value = PIECE_VALUES[moving_piece.type]
    return 100000 + (victim_value * 10) - aggressor_value

def move_ordering_capture(moves, board: Board, tt = None):
    """Sắp xếp nước đi ưu tiên các nước ăn quân theo MVV-LVA"""
    return sorted(moves, key=lambda move: sort_move_mvv_lva(move, board, tt), reverse=True)


def update_killer_move(move, ply):
    """Lưu nước đi yên tĩnh gây cắt beta vào Killer table"""
    if ply >= MAX_PLY:
        return
    # Không lưu nếu đã có trong slot 1
    if killer_moves[ply][0] == move:
        return
    # Đẩy slot 1 xuống slot 2, lưu mới vào slot 1
    killer_moves[ply][1] = killer_moves[ply][0]
    killer_moves[ply][0] = move


def update_history(move, color, depth):
    """Cập nhật history heuristic khi nước đi gây cắt beta"""
    color_idx = 0 if color == Color.WHITE else 1
    # Thưởng = depth^2 để ưu tiên nước đi từ depth cao
    bonus = depth * depth
    history_table[color_idx][move.from_square][move.to_square] += bonus
    # Giới hạn để tránh overflow
    if history_table[color_idx][move.from_square][move.to_square] > 10000:
        # Aging: Giảm tất cả xuống 50%
        for i in range(64):
            for j in range(64):
                history_table[color_idx][i][j] //= 2


def get_history_score(move, color):
    """Lấy điểm history cho nước đi"""
    color_idx = 0 if color == Color.WHITE else 1
    return history_table[color_idx][move.from_square][move.to_square]


def is_killer_move(move, ply):
    """Kiểm tra nước đi có trong Killer table"""
    if ply >= MAX_PLY:
        return False
    return move == killer_moves[ply][0] or move == killer_moves[ply][1]


def has_non_pawn_material(board, color):
    """Kiểm tra còn quân lớn (không phải tốt/vua) - dùng cho Null Move"""
    for piece in board.squares:
        if piece and piece.color == color:
            if piece.type in [PieceType.KNIGHT, PieceType.BISHOP, 
                             PieceType.ROOK, PieceType.QUEEN]:
                return True
    return False

def quiescence_search(board: Board, alpha: float, beta: float,
                      epsilon: float, model: NeuralNetwork, ply: int) -> float:
    
    # Stand-pat evaluation
    if ply > MAX_QS_PLY:
        # Trả về đánh giá vật chất đơn giản để thoát nhanh
        eval_score = material_evaluate(board)
        if board.current_turn == Color.BLACK:
            eval_score = -eval_score
        return eval_score
    hash_key = TT.compute_hash(board)
    entry = TT.lookup(hash_key)
    if entry is not None and entry['depth'] >= 0:
        val = entry['value']
        if val > MATE_THRESHOLD: val -= ply # Giải mã điểm Mate
        elif val < -MATE_THRESHOLD: val += ply
        
        if entry['flag'] == TranspositionTable.EXACT:
            return val
        elif entry['flag'] == TranspositionTable.LOWERBOUND:
            alpha = max(alpha, val)
        elif entry['flag'] == TranspositionTable.UPPERBOUND:
            beta = min(beta, val)
        
        if alpha >= beta:
            return val
        

    stand_pat = hybrid_evaluate(board, material_evaluate(board), epsilon, model, hash_key, entry)
    if stand_pat >= beta:
        return stand_pat
    if alpha < stand_pat:
        alpha = stand_pat

    move_gen = MoveGenerator(board)
    capture_moves = move_gen.generate_pseudo_captures()
    move_ordered = move_ordering_capture(capture_moves, board, tt=entry['best_move'] if entry else None)

    # --- DYNAMIC SEE THRESHOLD LOGIC ---
    
    # Mặc định: Chỉ chấp nhận nước đi hòa vốn hoặc lời (Standard)
    see_threshold = 0
    
    # 1. CHECK MATE DANGER (Ưu tiên cao nhất)
    # Nếu điểm đánh giá rơi vào vùng bị chiếu hết (ví dụ < -8000)
    # Lưu ý: MATE_THRESHOLD = 9000, MATE_SCORE = 10000
    if stand_pat < -(MATE_THRESHOLD - 100): 
        # Đang sắp bị chiếu hết: "Hung hãn tột độ"
        # Không được prune bất cứ nước bắt quân nào.
        see_threshold = -float('inf') 
        
    # 2. CHECK ABSOLUTE EVAL (Thua vật chất)
    # Nếu đang thua hơn 1 quân nhẹ (300 điểm ~ 3.0 pawns)
    elif stand_pat < -300: 
        # "Chó cùng rứt giậu": Cho phép các nước đổi lỗ (Xe đổi Mã/Tượng)
        # Hy vọng tạo complications hoặc stalemate.
        see_threshold = -200 
        
    # 3. CHECK RELATIVE ALPHA (Fail Low)
    # Nếu thế cờ hiện tại kém hơn kỳ vọng Alpha
    elif stand_pat < alpha - 50:
        # Thử các nước Gambit (thí tốt)
        see_threshold = -75
    # --- END DYNAMIC LOGIC ---
    
    for move in move_ordered:
        # Chỉ mở rộng nếu nước đi là bắt buộc (ví dụ chiếu, phong cấp)
        if not is_forced_move(board, move):
            if not see_capture(board, move, threshold=see_threshold):
                continue  # Prune nước đi ăn không lợi
        board.apply_move(move)
        score = -quiescence_search(board, -beta, -alpha, epsilon, model, ply + 1)
        board.undo_move()
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    # Lưu kết quả EXACT hoặc UPPERBOUND
    # Nếu alpha > stand_pat ban đầu nghĩa là ta tìm được nước ăn quân tốt hơn đứng yên -> EXACT
    # Nếu không thì ta chỉ biết nó tối đa là alpha (UPPERBOUND)
    tt_val = alpha
    if alpha > MATE_THRESHOLD:
        tt_val = alpha + ply   # CỘNG PLY
    elif alpha < -MATE_THRESHOLD:
        tt_val = alpha - ply
    flag = TranspositionTable.EXACT if alpha > stand_pat else TranspositionTable.UPPERBOUND
    TT.store(hash_key, depth=0, value=alpha, flag=flag, age=board.fullmove_number)
    
    return alpha

def negamax(board: Board, depth: int, alpha: float, beta: float,
            epsilon: float, model: NeuralNetwork, ply: int) -> float:
    """
    Negamax search algorithm.
    
    Args:
        board: Current board position
        alpha, beta: Alpha-beta pruning bounds
        depth: Search depth remaining
        epsilon: Weight for neural network evaluation (0.0 to 1.0)
        model: Neural network model
        
    Returns:
        Best score from current player's perspective
    """
    # Transposition table lookup
    alpha_orig = alpha

    hash_key = TT.compute_hash(board)
    entry = TT.lookup(hash_key)
    
    # BUG FIX: Xóa dòng trùng lặp
    static_eval = entry['static_eval'] if entry is not None else None
    
    # Nếu chưa có static_eval, hãy tính toán ngay.
    if static_eval is None:
        # Dùng material_evaluate cho nhanh (để phục vụ LMR/Null Move)
        # Nếu bạn muốn chính xác hơn có thể dùng model.evaluate nhưng sẽ chậm hơn
        static_eval = material_evaluate(board)
        
        # Nếu đang là lượt đen, đảo dấu để static_eval luôn là góc nhìn của người đang đi
        if board.current_turn == Color.BLACK:
            static_eval = -static_eval
   
   
    best_move_tt = entry['best_move'] if entry is not None else None
    if entry is not None and entry['depth'] >= depth:
        val = entry['value']
        if val > MATE_THRESHOLD: val -= ply # Giải mã điểm Mate
        elif val < -MATE_THRESHOLD: val += ply
        
        if entry['flag'] == TranspositionTable.EXACT:
            return val
        elif entry['flag'] == TranspositionTable.LOWERBOUND:
            alpha = max(alpha, val)
        elif entry['flag'] == TranspositionTable.UPPERBOUND:
            beta = min(beta, val)
        
        if alpha >= beta:
            return val
        
    
    if depth <= 0:
        return quiescence_search(board, alpha, beta, epsilon, model, ply)
    
    # --- NULL MOVE PRUNING ---
    # Điều kiện: Không bị chiếu, còn quân lớn, depth đủ, không ở PV node
    if (depth >= NULL_MOVE_MIN_DEPTH and 
        not is_in_check(board, board.current_turn) and
        has_non_pawn_material(board, board.current_turn) and
        static_eval >= beta):
        
        # Thực hiện null move (bỏ lượt)
        board.current_turn = Color.BLACK if board.current_turn == Color.WHITE else Color.WHITE
        
        # Tìm kiếm với depth giảm R
        null_score = -negamax(board, depth - 1 - NULL_MOVE_REDUCTION, 
                              -beta, -beta + 1, epsilon, model, ply + 1)
        
        # Hoàn tác null move
        board.current_turn = Color.BLACK if board.current_turn == Color.WHITE else Color.WHITE
        
        # Nếu vẫn >= beta sau khi bỏ lượt -> vị trí quá tốt, cắt luôn
        if null_score >= beta:
            # Tránh trả về mate score từ null move
            if null_score > MATE_THRESHOLD:
                null_score = beta
            return null_score
    
    # --- FUTILITY PRUNING SETUP ---
    # Nếu ở depth thấp và static_eval quá kém -> có thể prune quiet moves
    futility_pruning = False
    if depth <= 4 and not is_in_check(board, board.current_turn):
        margin = FUTILITY_MARGINS[depth] if depth < len(FUTILITY_MARGINS) else 400
        if static_eval + margin <= alpha:
            futility_pruning = True
    
    move_gen = MoveGenerator(board)
    moves = move_gen.generate_legal_moves()
    # Truyền killer_moves và history_table để sort tốt hơn
    moves = sort_moves_priority(moves, board, tt_move=best_move_tt, 
                                ply=ply, killer_moves=killer_moves, 
                                history_table=history_table)

    if not moves:
        if is_in_check(board, board.current_turn):
            return -(MATE_SCORE - ply)  # Checkmate: gần gốc = thắng nhanh hơn
        return 0.0  # Stalemate = hòa
    
    max_score = -float('inf')
    best_move_at_node = None  # BUG FIX: Khởi tạo để tránh UnboundLocalError

    # Biến kiểm tra bị chiếu (để truyền vào hàm LMR)
    in_check = is_in_check(board, board.current_turn)
    
    for i, move in enumerate(moves):
        is_capture = move.is_capture(board)
        is_promo = move.is_promotion()
        gives_check = False
        
        # --- FUTILITY PRUNING (cho quiet moves) ---
        # Prune nếu: depth thấp, không capture, không promo, không killer
        if futility_pruning and i > 0:  # Không prune nước đầu tiên
            if not is_capture and not is_promo and not is_killer_move(move, ply):
                # Kiểm tra nhanh xem có gây chiếu không
                board.apply_move(move)
                gives_check = is_in_check(board, board.current_turn)
                board.undo_move()
                
                if not gives_check:
                    continue  # Bỏ qua nước đi này
        
        # Logic LMR
        R = 0
        # Chỉ tính toán LMR cho các nước đi muộn (Move 4+)
        if depth >= 3 and i >= 3 and not in_check and not is_promo:
            # Tính gives_check nếu chưa tính
            if not gives_check:
                board.apply_move(move)
                gives_check = is_in_check(board, board.current_turn)
                board.undo_move()
            
            if not gives_check:
                # Tính SEE và R
                see_result = -1
                if is_capture:
                    if see_capture(board, move, threshold=0): 
                        see_result = 1
                
                # Gọi hàm tính R
                R = calculate_lmr_reduction(i, depth, move, board, is_capture, 
                                            see_result, in_check, static_eval, alpha, beta)
                
                # Giảm R cho killer moves (chúng đã tốt ở ply khác)
                if is_killer_move(move, ply):
                    R = max(0, R - 1)
                
                # Giảm R cho nước có history score cao
                if get_history_score(move, board.current_turn) > 1000:
                    R = max(0, R - 1)
        
        # --- BẮT ĐẦU PVS SEARCH ---
        board.apply_move(move)
        score = -float('inf')
        
        if i == 0:
            # Nước đi đầu tiên: Tìm kiếm full window
            score = -negamax(board, depth - 1, -beta, -alpha, epsilon, model, ply + 1)
        else:
            # PVS + LMR: Tìm kiếm với cửa sổ hẹp (null window) và giảm độ sâu
            score = -negamax(board, depth - 1 - R, -alpha - 1, -alpha, epsilon, model, ply + 1)
            
            # Re-search 1: Nếu LMR thất bại (score > alpha), thử lại full depth
            if score > alpha and R > 0:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, epsilon, model, ply + 1)
            
            # Re-search 2: Nếu null window thất bại, cần full window search
            # BUG FIX: Điều kiện phải là (score > alpha) thay vì (score > alpha and score < beta)
            # vì nếu score >= beta thì không cần re-search, sẽ cắt ngay
            if score > alpha:
                score = -negamax(board, depth - 1, -beta, -alpha, epsilon, model, ply + 1)
        
        board.undo_move()

        
        if score > max_score:
            max_score = score
            best_move_at_node = move
        if max_score > alpha:
            alpha = max_score
        if alpha >= beta:
            # --- Cắt Beta: Update Killer và History ---
            if not is_capture:  # Chỉ lưu quiet moves
                update_killer_move(move, ply)
                update_history(move, board.current_turn, depth)
            break
   
    # Lưu vào TT - BUG FIX: Logic flag bị đảo ngược
    # LOWERBOUND: Ta biết score >= max_score (fail-high, cắt beta)
    # UPPERBOUND: Ta biết score <= max_score (fail-low, không cải thiện alpha)  
    # EXACT: Tìm được giá trị chính xác trong cửa sổ [alpha, beta]
    if max_score <= alpha_orig:
        flag = TranspositionTable.UPPERBOUND  # Không cải thiện được alpha
    elif max_score >= beta:
        flag = TranspositionTable.LOWERBOUND  # Cắt beta, biết score >= max_score
    else:
        flag = TranspositionTable.EXACT  # Giá trị nằm trong cửa sổ
    tt_val = max_score
    
    if max_score > MATE_THRESHOLD:
        tt_val = max_score + ply   # CỘNG PLY ĐỂ CHUẨN HÓA
    elif max_score < -MATE_THRESHOLD:
        tt_val = max_score - ply   # TRỪ PLY ĐỂ CHUẨN HÓA (cho số âm)
    TT.store(hash_key, depth, tt_val, flag, board.fullmove_number, best_move_at_node)
    return max_score


def find_best_move(board: Board, depth: int, epsilon: float = 0.5, model: NeuralNetwork = None, verbose: bool = True):
    """
    Tìm nước đi tốt nhất với Iterative Deepening + Aspiration Windows.
    """
    if model is None:
        model = NeuralNetwork()
        model.load_model(PATH_MODEL)
    
    best_move = None
    prev_score = 0  # Score từ depth trước cho Aspiration Windows
    
    # --- ASPIRATION WINDOW PARAMETERS ---
    ASPIRATION_DELTA = 50  # Bắt đầu với cửa sổ ±50 centipawn
    
    # Reset killer moves cho search mới
    global killer_moves
    killer_moves = [[None, None] for _ in range(MAX_PLY)]
    
    # Đánh dấu search mới cho TT
    TT.new_search()
    
    for current_depth in range(1, depth + 1):
        move_gen = MoveGenerator(board)
        moves = move_gen.generate_legal_moves()
        
        if not moves:
            return None  # Không có nước đi hợp lệ
        
        # Lấy best move từ TT/depth trước để sort lên đầu
        hash_key = TT.compute_hash(board)
        tt_entry = TT.lookup(hash_key)
        tt_move = tt_entry['best_move'] if tt_entry else best_move
        
        moves = sort_moves_priority(moves, board, tt_move=tt_move)
        
        # --- ASPIRATION WINDOWS ---
        # Depth 1-2: Full window (chưa có thông tin)
        # Depth 3+: Dùng cửa sổ hẹp xung quanh score trước
        if current_depth <= 2:
            alpha = -float('inf')
            beta = float('inf')
        else:
            alpha = prev_score - ASPIRATION_DELTA
            beta = prev_score + ASPIRATION_DELTA
        
        current_best_move = None
        current_best_score = -float('inf')
        
        # Số lần thử lại nếu fail-high/fail-low
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            current_best_score = -float('inf')
            
            for i, move in enumerate(moves):
                board.apply_move(move)
                
                if i == 0:
                    # Full window cho nước đầu tiên
                    score = -negamax(board, current_depth - 1, -beta, -alpha, epsilon, model, ply=1)
                else:
                    # PVS: Null window trước
                    score = -negamax(board, current_depth - 1, -alpha - 1, -alpha, epsilon, model, ply=1)
                    if score > alpha and score < beta:
                        # Re-search với full window
                        score = -negamax(board, current_depth - 1, -beta, -alpha, epsilon, model, ply=1)
                
                board.undo_move()
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
                
                if score > alpha:
                    alpha = score
                
                # Fail-high trong aspiration window
                if alpha >= beta:
                    break
            
            # Kiểm tra Aspiration Window
            if current_depth > 2:
                if current_best_score <= prev_score - ASPIRATION_DELTA:
                    # Fail-low: Mở rộng alpha
                    alpha = -float('inf')
                    retries += 1
                    if verbose:
                        print(f"  Depth {current_depth}: Fail-low, re-search...")
                    continue
                elif current_best_score >= prev_score + ASPIRATION_DELTA:
                    # Fail-high: Mở rộng beta
                    beta = float('inf')
                    retries += 1
                    if verbose:
                        print(f"  Depth {current_depth}: Fail-high, re-search...")
                    continue
            
            # Tìm kiếm thành công
            break
        
        best_move = current_best_move
        prev_score = current_best_score
        
        if verbose:
            # Hiển thị thông tin đẹp hơn
            score_str = f"{current_best_score:.2f}"
            if current_best_score > MATE_THRESHOLD:
                mate_in = (MATE_SCORE - current_best_score + 1) // 2
                score_str = f"M{mate_in}"
            elif current_best_score < -MATE_THRESHOLD:
                mate_in = (MATE_SCORE + current_best_score + 1) // 2
                score_str = f"-M{mate_in}"
            print(f"Depth {current_depth}: {best_move} | Score: {score_str}")

    return best_move

if __name__ == "__main__":
    # Example usage
    model = PhantomChessNet()
    model.load_model(PATH_MODEL)
    fen_code = "r4knr/p4ppp/2Qpb3/2b3B1/8/2q2N2/P1P2PPP/R4RK1 w - - 0 1"
    board = Board(fen_code)
    start_time = time.time()
    best_move = find_best_move(board, depth=3, epsilon=0.7, model=model)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Best move: {best_move}")
