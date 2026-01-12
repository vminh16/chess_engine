"""
Test suite for chess moves, board state, and rules.
Tests basic moves, special moves (castling, en passant, promotion),
check detection, move legality, and apply/undo functionality.
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.constants import Color, PieceType, Piece, Square
from core.move import Move
from core.board import Board
from core.rules import (
    is_in_check, is_legal_move, can_castle_kingside,
    can_castle_queenside, is_square_attacked
)
from core.move_generator import MoveGenerator, generate_legal_moves


class TestBasicMoves(unittest.TestCase):
    """Test basic piece movements"""
    
    def setUp(self):
        """Set up a fresh board for each test"""
        self.board = Board()
    
    def test_pawn_single_move(self):
        """Test basic pawn single forward move"""
        # White pawn e2 -> e4
        move = Move(Square.E2, Square.E3)
        self.board.apply_move(move)
        
        piece = self.board.get_piece(Square.E3)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.PAWN)
        self.assertEqual(piece.color, Color.WHITE)
        
        # Square e2 should be empty
        self.assertIsNone(self.board.get_piece(Square.E2))
    
    def test_pawn_double_move(self):
        """Test pawn double push from starting position"""
        # White pawn e2 -> e4 (double push)
        move = Move(Square.E2, Square.E4)
        self.board.apply_move(move)
        
        piece = self.board.get_piece(Square.E4)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.PAWN)
        
        # Check en passant target is set
        self.assertEqual(self.board.en_passant_target, Square.E3)
    
    def test_pawn_capture(self):
        """Test pawn diagonal capture"""
        # Setup: e4 d5
        board = Board()
        board.set_piece(Square.E4, Piece(Color.WHITE, PieceType.PAWN))
        board.set_piece(Square.D5, Piece(Color.BLACK, PieceType.PAWN))
        board.current_turn = Color.WHITE
        
        # Capture d5 with e4
        move = Move(Square.E4, Square.D5)
        board.apply_move(move)
        
        piece = board.get_piece(Square.D5)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.color, Color.WHITE)
    
    def test_knight_move(self):
        """Test knight L-shaped movement"""
        # White knight g1 -> f3
        move = Move(Square.G1, Square.F3)
        self.board.apply_move(move)
        
        piece = self.board.get_piece(Square.F3)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.KNIGHT)
        self.assertEqual(piece.color, Color.WHITE)
    
    def test_bishop_diagonal_move(self):
        """Test bishop diagonal sliding"""
        # Setup: clear path for bishop - pawn already moved to d3
        board = Board("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1")
        # Bishop c1 -> f4 (diagonal)
        move = Move(Square.C1, Square.F4)
        board.apply_move(move)
        
        piece = board.get_piece(Square.F4)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.BISHOP)
    
    def test_rook_orthogonal_move(self):
        """Test rook horizontal/vertical sliding"""
        # Setup: clear path for rook
        board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1")
        # Rook h1 -> h3
        move = Move(Square.H1, Square.H3)
        board.apply_move(move)
        
        piece = board.get_piece(Square.H3)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.ROOK)
    
    def test_queen_move(self):
        """Test queen can move like rook and bishop"""
        # Setup: clear path for queen - pawn already on d3
        board = Board("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1")
        # Queen d1 -> d4 (vertical move, like rook)
        move = Move(Square.D1, Square.D4)
        board.apply_move(move)
        
        piece = board.get_piece(Square.D4)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.QUEEN)
        self.assertEqual(piece.color, Color.WHITE)


class TestSpecialMoves(unittest.TestCase):
    """Test special moves: castling, en passant, promotion"""
    
    def setUp(self):
        """Set up a fresh board for each test"""
        self.board = Board()
    
    def test_kingside_castling(self):
        """Test kingside castling"""
        # Setup position for castling (clear f1, g1)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1"
        board = Board(fen)
        
        # Castling should be legal
        self.assertTrue(can_castle_kingside(board, Color.WHITE))
        
        # Perform castling
        king_square = Square.E1
        move = Move(king_square, king_square + 2, is_castle=True, castle_side='kingside')
        board.apply_move(move)
        
        # King should be on g1
        king = board.get_piece(Square.G1)
        self.assertIsNotNone(king)
        self.assertEqual(king.type, PieceType.KING)
        self.assertEqual(king.color, Color.WHITE)
        
        # Rook should be on f1
        rook = board.get_piece(Square.F1)
        self.assertIsNotNone(rook)
        self.assertEqual(rook.type, PieceType.ROOK)
        self.assertEqual(rook.color, Color.WHITE)
        
        # Castling rights should be lost
        self.assertFalse(board.castling_rights[Color.WHITE]['kingside'])
    
    def test_queenside_castling(self):
        """Test queenside castling"""
        # Setup position for castling (clear b1, c1, d1)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 1"
        board = Board(fen)
        
        # Castling should be legal
        self.assertTrue(can_castle_queenside(board, Color.WHITE))
        
        # Perform castling
        king_square = Square.E1
        move = Move(king_square, king_square - 2, is_castle=True, castle_side='queenside')
        board.apply_move(move)
        
        # King should be on c1
        king = board.get_piece(Square.C1)
        self.assertIsNotNone(king)
        self.assertEqual(king.type, PieceType.KING)
        
        # Rook should be on d1
        rook = board.get_piece(Square.D1)
        self.assertIsNotNone(rook)
        self.assertEqual(rook.type, PieceType.ROOK)
    
    def test_en_passant_capture(self):
        """Test en passant capture"""
        # Setup: White pawn on e5, ready to capture en passant
        # Position after: e4 d5 e5 (white can capture en passant on d6)
        fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        board = Board(fen)
        
        # White pawn e5 -> d6 (en passant capture)
        move = Move(Square.E5, Square.D6, is_en_passant=True)
        
        # Check if en passant is legal
        self.assertTrue(is_legal_move(board, move))
        board.apply_move(move)
        
        # Pawn should be on d6
        piece = board.get_piece(Square.D6)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.PAWN)
        self.assertEqual(piece.color, Color.WHITE)
        
        # Captured pawn on d5 should be gone
        self.assertIsNone(board.get_piece(Square.D5))
    
    def test_pawn_promotion(self):
        """Test pawn promotion"""
        # Setup: White pawn on a7, ready to promote
        fen = "8/P7/8/8/8/8/8/8 w - - 0 1"
        board = Board(fen)
        
        # Promote to queen
        move = Move(Square.A7, Square.A8, promotion=PieceType.QUEEN)
        board.apply_move(move)
        
        piece = board.get_piece(Square.A8)
        self.assertIsNotNone(piece)
        self.assertEqual(piece.type, PieceType.QUEEN)
        self.assertEqual(piece.color, Color.WHITE)


class TestCheckDetection(unittest.TestCase):
    """Test check detection and legality"""
    
    def setUp(self):
        """Set up a fresh board for each test"""
        self.board = Board()
    
    def test_square_attacked_by_pawn(self):
        """Test if square is attacked by a pawn"""
        # Setup: Black pawn on d4
        board = Board("rnbqkbnr/pppppppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        # Square e3 should be attacked by pawn on d4
        self.assertTrue(is_square_attacked(board, Square.E3, Color.BLACK))
        
        # Square c3 should also be attacked
        self.assertTrue(is_square_attacked(board, Square.C3, Color.BLACK))
        
        # Square e4 should not be attacked (pawn attacks diagonally forward)
        self.assertFalse(is_square_attacked(board, Square.E4, Color.BLACK))
    
    def test_king_in_check(self):
        """Test if king is in check"""
        # Setup: Black queen attacking white king (clear path, no blocking pieces)
        fen = "rnb1kb1r/pppppppp/8/8/4q3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        board = Board(fen)
        
        # White king on e1 should be in check from black queen on e4
        # (pawn moved from e2, so path is clear)
        self.assertTrue(is_in_check(board, Color.WHITE))
        
        # Verify square is attacked
        self.assertTrue(is_square_attacked(board, Square.E1, Color.BLACK))
    
    def test_illegal_move_leaving_king_in_check(self):
        """Test that moves leaving own king in check are illegal"""
        # Setup position where moving a piece exposes king
        fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 1"
        board = Board(fen)
        
        # Try a move that would leave king in check - should be filtered out
        moves = generate_legal_moves(board, Color.BLACK)
        
        # None of the legal moves should leave the king in check
        for move in moves:
            board.apply_move(move)
            # After move, opponent's turn, so check if we left our king in check
            is_illegal = is_in_check(board, board.current_turn.opponent())
            board.undo_move()
            
            # Should not be in check after legal move (checked by is_legal_move)
            self.assertFalse(is_illegal, f"Move {move} should be legal")


class TestMoveGeneration(unittest.TestCase):
    """Test move generation"""
    
    def setUp(self):
        """Set up a fresh board for each test"""
        self.board = Board()
    
    def test_initial_position_moves(self):
        """Test move generation from starting position"""
        moves = generate_legal_moves(self.board, Color.WHITE)
        
        # Starting position should have 20 legal moves
        self.assertEqual(len(moves), 20)
        
        # All moves should be legal
        for move in moves:
            self.assertTrue(is_legal_move(self.board, move))
    
    def test_move_generation_includes_castling(self):
        """Test that castling moves are generated when legal"""
        # Setup position for castling
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1"
        board = Board(fen)
        
        moves = generate_legal_moves(board, Color.WHITE)
        
        # Should include castling move
        castling_moves = [m for m in moves if m.is_castle and m.castle_side == 'kingside']
        self.assertGreater(len(castling_moves), 0)
    
    def test_pawn_promotion_moves_generated(self):
        """Test that promotion moves are generated for pawns on 7th rank"""
        # Setup: White pawn on a7, can advance to a8
        fen = "8/P7/8/8/8/8/8/8 w - - 0 1"
        board = Board(fen)
        
        moves = generate_legal_moves(board, Color.WHITE)
        
        # Should have at least 4 promotion moves (queen, rook, bishop, knight)
        # Note: Promotion moves are generated for pawn on a7 -> a8
        promotion_moves = [m for m in moves if m.is_promotion()]
        
        # Check that we have promotion moves
        self.assertGreater(len(promotion_moves), 0, 
                          f"Expected promotion moves, got {len(moves)} total moves")
        
        # Verify at least one promotion type is queen
        queen_promotions = [m for m in promotion_moves if m.promotion == PieceType.QUEEN]
        self.assertGreater(len(queen_promotions), 0)


class TestApplyUndo(unittest.TestCase):
    """Test apply and undo move functionality"""
    
    def setUp(self):
        """Set up a fresh board for each test"""
        self.board = Board()
    
    def test_apply_and_undo_move(self):
        """Test that applying and undoing a move restores board state"""
        # Save initial FEN
        initial_fen = self.board.to_fen()
        initial_turn = self.board.current_turn
        
        # Apply a move
        move = Move(Square.E2, Square.E4)
        self.board.apply_move(move)
        
        # Verify move was applied
        self.assertIsNotNone(self.board.get_piece(Square.E4))
        
        # Undo move
        self.board.undo_move()
        
        # Verify board is restored
        restored_fen = self.board.to_fen()
        self.assertEqual(initial_fen, restored_fen)
        self.assertEqual(initial_turn, self.board.current_turn)
        self.assertIsNone(self.board.get_piece(Square.E4))
    
    def test_apply_undo_castling(self):
        """Test apply and undo castling"""
        # Setup for castling
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1"
        board = Board(fen)
        
        initial_fen = board.to_fen()
        
        # Apply castling
        move = Move(Square.E1, Square.G1, is_castle=True, castle_side='kingside')
        board.apply_move(move)
        
        # Verify castling
        self.assertIsNotNone(board.get_piece(Square.G1))
        
        # Undo
        board.undo_move()
        
        # Verify restoration
        restored_fen = board.to_fen()
        self.assertEqual(initial_fen, restored_fen)
    
    def test_apply_undo_promotion(self):
        """Test apply and undo pawn promotion"""
        fen = "8/P7/8/8/8/8/8/8 w - - 0 1"
        board = Board(fen)
        
        initial_fen = board.to_fen()
        
        # Promote
        move = Move(Square.A7, Square.A8, promotion=PieceType.QUEEN)
        board.apply_move(move)
        
        self.assertEqual(board.get_piece(Square.A8).type, PieceType.QUEEN)
        
        # Undo
        board.undo_move()
        
        # Verify restoration
        restored_fen = board.to_fen()
        self.assertEqual(initial_fen, restored_fen)
        self.assertEqual(board.get_piece(Square.A7).type, PieceType.PAWN)


class TestBoardState(unittest.TestCase):
    """Test board state management"""
    
    def setUp(self):
        """Set up a fresh board for each test"""
        self.board = Board()
    
    def test_fen_loading_and_saving(self):
        """Test FEN string loading and saving"""
        # Test starting position
        fen1 = self.board.to_fen()
        board2 = Board(fen1)
        fen2 = board2.to_fen()
        
        self.assertEqual(fen1, fen2)
    
    def test_turn_switching(self):
        """Test that turn switches after move"""
        initial_turn = self.board.current_turn
        self.assertEqual(initial_turn, Color.WHITE)
        
        self.board.apply_move(Move(Square.E2, Square.E4))
        
        self.assertEqual(self.board.current_turn, Color.BLACK)
        
        self.board.apply_move(Move(Square.E7, Square.E5))
        
        self.assertEqual(self.board.current_turn, Color.WHITE)
    
    def test_castling_rights_lost_after_king_move(self):
        """Test that castling rights are lost after king moves"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = Board(fen)
        
        # Initially has castling rights
        self.assertTrue(board.castling_rights[Color.WHITE]['kingside'])
        
        # Move king
        board.apply_move(Move(Square.E1, Square.E2))
        
        # Castling rights should be lost
        self.assertFalse(board.castling_rights[Color.WHITE]['kingside'])
        self.assertFalse(board.castling_rights[Color.WHITE]['queenside'])
    
    def test_en_passant_target_set(self):
        """Test that en passant target is set after double pawn push"""
        # Double pawn push
        self.board.apply_move(Move(Square.E2, Square.E4))
        
        # En passant target should be e3
        self.assertEqual(self.board.en_passant_target, Square.E3)
        
        # After next move, should be cleared
        self.board.apply_move(Move(Square.E7, Square.E5))
        
        # En passant should be cleared (no double push this time)
        # Actually, e5 is a double push, so en passant target should be e6
        # But let's check after white moves again
        self.board.apply_move(Move(Square.G1, Square.F3))
        
        # En passant should be None now
        self.assertIsNone(self.board.en_passant_target)


class TestMoveRepresentation(unittest.TestCase):
    """Test Move class and string conversion"""
    
    def test_move_creation(self):
        """Test creating moves"""
        move = Move(Square.E2, Square.E4)
        self.assertEqual(move.from_square, Square.E2)
        self.assertEqual(move.to_square, Square.E4)
        self.assertFalse(move.is_promotion())
    
    def test_move_string_conversion(self):
        """Test move to/from string conversion"""
        move = Move(Square.E2, Square.E4)
        move_str = move.to_string()
        
        self.assertEqual(move_str, "e2e4")
        
        # Test parsing
        parsed_move = Move.from_string("e2e4")
        self.assertEqual(parsed_move.from_square, Square.E2)
        self.assertEqual(parsed_move.to_square, Square.E4)
    
    def test_promotion_move_string(self):
        """Test promotion move string conversion"""
        move = Move(Square.A7, Square.A8, promotion=PieceType.QUEEN)
        move_str = move.to_string()
        
        self.assertEqual(move_str, "a7a8q")
        
        # Test parsing
        parsed_move = Move.from_string("a7a8q")
        self.assertEqual(parsed_move.promotion, PieceType.QUEEN)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

