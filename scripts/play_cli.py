"""
Command-line interface for playing chess.
Allows two players to play chess against each other via terminal.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.board import Board
from core.move import Move
from core.constants import Color, PieceType, Square
from core.move_generator import generate_legal_moves
from core.rules import is_in_check, is_checkmate, is_stalemate


class ChessCLI:
    """Command-line interface for chess game"""
    
    def __init__(self):
        self.board = Board()
        self.game_over = False
    
    def display_board(self):
        """Display the chess board in a readable format"""
        print("\n" + "=" * 50)
        print(f"Turn: {'White' if self.board.current_turn == Color.WHITE else 'Black'}")
        print("=" * 50)
        
        # Display board with coordinates
        print("\n   a b c d e f g h")
        print("  +" + "-" * 17 + "+")
        
        for rank in range(7, -1, -1):
            print(f"{rank + 1} |", end=" ")
            for file in range(8):
                square = rank * 8 + file
                piece = self.board.get_piece(square)
                if piece is None:
                    symbol = "."
                else:
                    symbols = {
                        (Color.WHITE, PieceType.PAWN): "P",
                        (Color.WHITE, PieceType.KNIGHT): "N",
                        (Color.WHITE, PieceType.BISHOP): "B",
                        (Color.WHITE, PieceType.ROOK): "R",
                        (Color.WHITE, PieceType.QUEEN): "Q",
                        (Color.WHITE, PieceType.KING): "K",
                        (Color.BLACK, PieceType.PAWN): "p",
                        (Color.BLACK, PieceType.KNIGHT): "n",
                        (Color.BLACK, PieceType.BISHOP): "b",
                        (Color.BLACK, PieceType.ROOK): "r",
                        (Color.BLACK, PieceType.QUEEN): "q",
                        (Color.BLACK, PieceType.KING): "k",
                    }
                    symbol = symbols.get((piece.color, piece.type), "?")
                print(symbol, end=" ")
            print(f"| {rank + 1}")
        
        print("  +" + "-" * 17 + "+")
        print("   a b c d e f g h")
        
        # Display game status
        if self.board.current_turn == Color.WHITE:
            opponent = Color.BLACK
        else:
            opponent = Color.WHITE
        
        if is_in_check(self.board, self.board.current_turn):
            print("\n⚠️  CHECK!")
        
        # Check for game over conditions
        legal_moves = generate_legal_moves(self.board, self.board.current_turn)
        if len(legal_moves) == 0:
            if is_in_check(self.board, self.board.current_turn):
                print(f"\n🎯 CHECKMATE! {opponent.name.capitalize()} wins!")
                self.game_over = True
            else:
                print(f"\n🤝 STALEMATE! Game is a draw.")
                self.game_over = True
        
        print()
    
    def parse_move(self, move_str: str):
        """
        Parse move input from user.
        Supports formats:
        - UCI: e2e4, e7e8q
        - Algebraic-like: e2-e4
        - Castling: O-O, O-O-O
        """
        move_str = move_str.strip().lower().replace("-", "").replace(" ", "")
        
        # Handle castling notation
        if move_str == "oo" or move_str == "0-0":
            # Kingside castling
            rank = 0 if self.board.current_turn == Color.WHITE else 7
            king_square = rank * 8 + 4  # e1 or e8
            moves = generate_legal_moves(self.board, self.board.current_turn)
            for move in moves:
                if move.is_castle and move.castle_side == 'kingside' and move.from_square == king_square:
                    return move
            return None
        elif move_str == "ooo" or move_str == "0-0-0":
            # Queenside castling
            rank = 0 if self.board.current_turn == Color.WHITE else 7
            king_square = rank * 8 + 4  # e1 or e8
            moves = generate_legal_moves(self.board, self.board.current_turn)
            for move in moves:
                if move.is_castle and move.castle_side == 'queenside' and move.from_square == king_square:
                    return move
            return None
        
        # Try UCI format: e2e4, e7e8q
        if len(move_str) >= 4:
            from_sq = Square.from_string(move_str[0:2])
            to_sq = Square.from_string(move_str[2:4])
            
            if from_sq is None or to_sq is None:
                return None
            
            # Check for promotion
            promotion = None
            if len(move_str) == 5:
                prom_map = {
                    'q': PieceType.QUEEN,
                    'r': PieceType.ROOK,
                    'b': PieceType.BISHOP,
                    'n': PieceType.KNIGHT
                }
                promotion = prom_map.get(move_str[4])
            
            # Try to find matching move in legal moves
            legal_moves = generate_legal_moves(self.board, self.board.current_turn)
            for move in legal_moves:
                if (move.from_square == from_sq and 
                    move.to_square == to_sq and 
                    move.promotion == promotion):
                    return move
            
            # If not found in legal moves, create move anyway (will be validated)
            return Move(from_sq, to_sq, promotion=promotion)
        
        return None
    
    def show_help(self):
        """Display help message"""
        print("\n" + "=" * 50)
        print("HOW TO PLAY")
        print("=" * 50)
        print("Enter moves in UCI format:")
        print("  - e2e4       (pawn from e2 to e4)")
        print("  - g1f3       (knight from g1 to f3)")
        print("  - e7e8q      (pawn promotes to queen)")
        print("  - O-O        (kingside castling)")
        print("  - O-O-O      (queenside castling)")
        print("\nOther commands:")
        print("  - help       (show this help)")
        print("  - moves      (show all legal moves)")
        print("  - undo       (undo last move)")
        print("  - fen        (show current position in FEN)")
        print("  - quit/exit  (exit game)")
        print("=" * 50 + "\n")
    
    def show_legal_moves(self):
        """Display all legal moves for current player"""
        moves = generate_legal_moves(self.board, self.board.current_turn)
        print(f"\nLegal moves ({len(moves)}):")
        
        # Group moves by from square for readability
        moves_by_square = {}
        for move in moves:
            from_str = Square.to_string(move.from_square)
            if from_str not in moves_by_square:
                moves_by_square[from_str] = []
            moves_by_square[from_str].append(move)
        
        # Display moves
        for from_sq in sorted(moves_by_square.keys()):
            to_squares = [Square.to_string(m.to_square) for m in moves_by_square[from_sq]]
            print(f"  {from_sq}: {', '.join(to_squares)}")
        print()
    
    def run(self):
        """Main game loop"""
        print("\n" + "=" * 50)
        print("WELCOME TO CHESS CLI")
        print("=" * 50)
        print("Type 'help' for instructions")
        
        while not self.game_over:
            self.display_board()
            
            # Get user input
            try:
                user_input = input(f"{'White' if self.board.current_turn == Color.WHITE else 'Black'} to move: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGame exited.")
                break
            
            if not user_input:
                continue
            
            user_input_lower = user_input.lower()
            
            # Handle special commands
            if user_input_lower in ['quit', 'exit', 'q']:
                print("\nThanks for playing!")
                break
            elif user_input_lower == 'help':
                self.show_help()
                continue
            elif user_input_lower == 'moves':
                self.show_legal_moves()
                continue
            elif user_input_lower == 'undo':
                if self.board.move_history:
                    self.board.undo_move()
                    print("Move undone.")
                else:
                    print("No moves to undo.")
                continue
            elif user_input_lower == 'fen':
                print(f"\nFEN: {self.board.to_fen()}\n")
                continue
            
            # Parse and execute move
            move = self.parse_move(user_input)
            
            if move is None:
                print("❌ Invalid move format. Type 'help' for instructions.")
                continue
            
            # Validate move is legal
            from core.rules import is_legal_move
            if not is_legal_move(self.board, move):
                print("❌ Illegal move! Try again.")
                # Show why it's illegal
                if is_in_check(self.board, self.board.current_turn):
                    print("   (Your king is in check - you must get out of check)")
                continue
            
            # Apply move
            try:
                self.board.apply_move(move)
                print(f"✓ Move played: {move}")
            except Exception as e:
                print(f"❌ Error applying move: {e}")
                continue
        
        # Final board display
        if self.game_over:
            self.display_board()


def main():
    """Entry point"""
    try:
        cli = ChessCLI()
        cli.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

