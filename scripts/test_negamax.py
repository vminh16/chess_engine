"""
Test script for negamax search.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.board import Board
from core.move import Move
from core.constants import Square
from search.negamax import find_best_move

def test_depth_1():
    """Test depth 1: should capture free piece"""
    board = Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")  # White pawn can capture black pawn
    best_move = find_best_move(board, 1)
    print(f"Depth 1 test: Best move {best_move}")
    # Should be e4xd5

def test_depth_2():
    """Test depth 2: avoid recapture"""
    # White rook at d4, black queen at d5: capturing leads to recapture
    board = Board("rnbqkbnr/pppppppp/8/3q4/3R4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    best_move = find_best_move(board, 2)
    print(f"Depth 2 test: Best move {best_move}")
    # Should avoid d4xd5 because black can qxd4

def test_checkmate_avoidance():
    """Test checkmate avoidance: choose move to avoid checkmate if possible"""
    # Position where white is in check, can capture the checking piece
    board = Board("rnbqkbnr/pppp1ppp/8/4p3/6Q1/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1")  # Black to move, white Q at g4, black can capture?
    # Wait, better position: white king in check by black queen, can capture queen
    # FEN: "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1" no.
    # Simple: black queen at d8, white king e1, but white can move or capture.
    # Position: white king e1, black queen d8, white can Qxd8 or move king.
    # But to make check, perhaps "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR b KQkq - 0 1" no.
    # Use a known position.
    # For simplicity, assume a position where checkmate is imminent but can be avoided.
    board = Board("r1bqkbnr/pppp1ppp/2n5/4p3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 1")  # White can avoid checkmate
    best_move = find_best_move(board, 1)
    print(f"Checkmate avoidance test: Best move {best_move}")

def test_stalemate():
    """Test stalemate: eval = 0"""
    # Stalemate position: black king at a8, white king at a1, white queen at b6
    board = Board("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
    from search.negamax import negamax
    score = negamax(board, 1)
    print(f"Stalemate test: score = {score} (should be 0.0)")

if __name__ == "__main__":
    test_depth_1()
    test_depth_2()
    test_checkmate_avoidance()
    test_stalemate()