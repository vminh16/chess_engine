# evaluation/static_eval.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.board import Board
from core.constants import Color

# Di chuyển PIECE_VALUES sang đây
# Chỉ số tương ứng: 0:None, 1:PAWN, 2:KNIGHT, 3:BISHOP, 4:ROOK, 5:QUEEN, 6:KING
PIECE_VALUES = [0, 100, 300, 310, 500, 900, 10000]

# Di chuyển hàm material_evaluate sang đây
def material_evaluate(board: Board) -> float:
    """
    Simple material evaluation function.
    """
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
    }
    
    score = 0
    for piece in board.squares:
        if piece is not None:
            type_str = ["", "P", "N", "B", "R", "Q", "K"][piece.type]
            symbol = type_str if piece.color == Color.WHITE else type_str.lower()
            score += piece_values[symbol]
    
    return score/39.0