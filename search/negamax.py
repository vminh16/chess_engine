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

MATE_SCORE = 10000
MATE_THRESHOLD = 9000
MAX_QS_PLY = 100

# Chỉ số tương ứng: 0:None, 1:PAWN, 2:KNIGHT, 3:BISHOP, 4:ROOK, 5:QUEEN, 6:KING
PIECE_VALUES = [0, 100, 300, 310, 500, 900, 10000]


def hybrid_evaluate(board, material_eval, epsilon, model, hash_key, tt_entry=None):
    """
    Hàm đánh giá lai: Kết hợp Material và Neural Network.
    Được tích hợp chặt chẽ với Transposition Table để cache NN.
    """
    nn_eval = None
    
    # Thử lấy NN eval từ TT nếu có
    if tt_entry is not None and tt_entry.get('static_eval') is not None:
        nn_eval = tt_entry['static_eval']
    
   
    if nn_eval is None:
        # nn_eval = model.evaluate(encode_board(board))
        nn_eval = model.predict(encode_board(board))
        # Ta lưu với depth = -1 hoặc logic giữ nguyên các trường khác để chỉ update static_eval
        TT.store(hash_key, depth=-1, value=0, flag=TranspositionTable.EXACT, 
                 age=board.fullmove_number, static_eval=nn_eval)
    hybrid_score = (1 - epsilon) * material_eval + epsilon * nn_eval
    # Negate điểm nếu là lượt đi của Đen
    if board.current_turn == Color.BLACK:
        hybrid_score = -hybrid_score
    return hybrid_score

def sort_move_mvv_lva(move, board: Board, tt=None) -> int:
    """Sắp xếp nước đi theo MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)"""
    
     # Đặc biệt cho phong cấp
    if move.is_promotion():
        aggressor_value = PIECE_VALUES[board.get_piece(move.from_square).type]
        promotion_value = PIECE_VALUES[move.promotion]
        return 90000 + (promotion_value * 10) - aggressor_value

    if not move.is_capture(board):
        return 0
    if tt is not None and tt == move:
        return 1000000  # Nước đi trong bảng chuyển vị có ưu tiên cao nhất

    if move.is_en_passant:
        victim_value = PIECE_VALUES[1]  # Giá trị của tốt
        aggressor_value = PIECE_VALUES[1]  # Giá trị của tốt
        return (victim_value * 10) - aggressor_value

    # ăn quân thông thường
    capture_move = board.get_piece(move.to_square)
    victim_value = PIECE_VALUES[capture_move.type]
    aggressor_value = PIECE_VALUES[board.get_piece(move.from_square).type]
    return 100000 + (victim_value * 10) - aggressor_value

def move_ordering_capture(moves, board: Board, tt = None):
    """Sắp xếp nước đi ưu tiên các nước ăn quân theo MVV-LVA"""
    return sorted(moves, key=lambda move: sort_move_mvv_lva(move, board, tt), reverse=True)

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
    
    static_eval = entry['static_eval'] if entry is not None else None
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
        #return material_evaluate(board)
        return quiescence_search(board, alpha, beta, epsilon, model, ply)
    
    
    move_gen = MoveGenerator(board)
    moves = move_gen.generate_legal_moves()
    moves = sort_moves_priority(moves, board, tt_move=best_move_tt)

    if not moves:
        if is_in_check(board, board.current_turn):
            return -(MATE_SCORE - ply) # Gần gốc hơn thì điểm âm nặng hơn
        return 0.0 # Hòa
    max_score = -float('inf')

    # Biến kiểm tra bị chiếu (để truyền vào hàm LMR)
    in_check = is_in_check(board, board.current_turn)
    
    for i, move in enumerate(moves):
        # Chỉ tính gives_check khi thực sự cần cho LMR (i >= 4)
        
        is_capture = move.is_capture(board)
        gives_check = False
        
        # Logic LMR
        R = 0
        # Chỉ tính toán LMR cho các nước đi muộn (Move 4+) để tiết kiệm CPU
        if depth >= 3 and i >= 3 and not in_check and not move.is_promotion():
            # Bây giờ mới tốn tiền tính gives_check
            board.apply_move(move)
            gives_check = is_in_check(board, board.current_turn)
            board.undo_move()
            
            if not gives_check:
                # Tính SEE và R
                see_result = -1
                if is_capture:
                     # Gọi SEE
                    if see_capture(board, move, threshold=0): see_result = 1
                
                # Gọi hàm tính R
                R = calculate_lmr_reduction(i, depth, move, board, is_capture, 
                                            see_result, in_check, static_eval, alpha, beta)
        
        # --- BẮT ĐẦU PVS SEARCH ---
        board.apply_move(move)
        score = -float('inf')
        
        if i == 0:
            score = -negamax(board, depth - 1, -beta, -alpha, epsilon, model, ply + 1)
        else:
            # LMR Search
            score = -negamax(board, depth - 1 - R, -alpha - 1, -alpha, epsilon, model, ply + 1)
            
            # Re-search logic (như cũ)
            if score > alpha and R > 0:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, epsilon, model, ply + 1)
            if score > alpha and score < beta:
                score = -negamax(board, depth - 1, -beta, -alpha, epsilon, model, ply + 1)
        
        board.undo_move()

        
        if score > max_score:
            max_score = score
            best_move_at_node = move
        if max_score > alpha:
            alpha = max_score
        if alpha >= beta:
            break
   
    # save to transposition table
    if max_score <= alpha_orig:
        flag = TranspositionTable.UPPERBOUND
    elif max_score >= beta:
        flag = TranspositionTable.LOWERBOUND
    else:
        flag = TranspositionTable.EXACT
    tt_val = max_score
    
    if max_score > MATE_THRESHOLD:
        tt_val = max_score + ply   # CỘNG PLY ĐỂ CHUẨN HÓA
    elif max_score < -MATE_THRESHOLD:
        tt_val = max_score - ply   # TRỪ PLY ĐỂ CHUẨN HÓA (cho số âm)
    TT.store(hash_key, depth, tt_val, flag, board.fullmove_number, best_move_at_node)
    return max_score


def find_best_move(board: Board, depth: int, epsilon: float = 0.5, model: NeuralNetwork = None, verbose: bool = True):
    if model is None:
        model = NeuralNetwork()
        model.load_model(PATH_MODEL)
    
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    # Sử dụng Iterative Deepening (Tăng dần độ sâu)
    # Giúp Transposition Table có dữ liệu để Move Ordering tốt hơn
    for current_depth in range(1, depth + 1):
        move_gen = MoveGenerator(board)
        moves = move_gen.generate_legal_moves()
        
        # Lấy best move từ TT của depth trước để sort lên đầu
        hash_key = TT.compute_hash(board)
        tt_entry = TT.lookup(hash_key)
        tt_move = tt_entry['best_move'] if tt_entry else best_move
        
        # Sort moves
        moves = sort_moves_priority(moves, board, tt_move=tt_move)
        
        current_best_move = None
        current_best_score = -float('inf')
        
        # Reset Alpha Beta cho mỗi depth mới
        alpha = -float('inf')
        
        for move in moves:
            board.apply_move(move)
            # QUAN TRỌNG: Truyền -beta, -alpha vào để duy trì chuỗi pruning
            score = -negamax(board, current_depth - 1, -beta, -alpha, epsilon, model, ply=1)
            board.undo_move()
            
            if score > current_best_score:
                current_best_score = score
                current_best_move = move
            
            # Cập nhật Alpha ngay tại Root
            if score > alpha:
                alpha = score
                
        best_move = current_best_move
        if verbose:
            print(f"Depth {current_depth} finished. Best: {best_move} Score: {current_best_score}")

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
