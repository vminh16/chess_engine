"""
Biểu diễn bàn cờ và quản lý trạng thái.
Xử lý áp dụng và hoàn tác nước đi.
"""
from typing import Optional, List
from .constants import (
    Color, PieceType, Piece, Square, Direction
)
from .move import Move


class Board:
    """Đại diện cho bàn cờ và trạng thái ván cờ"""
    
    # Vị trí khởi đầu ở định dạng FEN
    STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def __init__(self, fen: Optional[str] = None):
        """
        Khởi tạo bàn cờ từ chuỗi FEN hoặc vị trí khởi đầu.
        
        Args:
            fen: Chuỗi FEN biểu diễn bàn cờ
        """
        # 64 ô, mỗi ô có thể chứa một Piece hoặc None
        self.squares: List[Optional[Piece]] = [None] * 64
        
        # Trạng thái ván cờ
        self.current_turn: Color = Color.WHITE
        self.castling_rights: dict = {
            Color.WHITE: {'kingside': True, 'queenside': True},
            Color.BLACK: {'kingside': True, 'queenside': True}
        }
        self.en_passant_target: Optional[int] = None  # Chỉ số ô hoặc None
        self.halfmove_clock: int = 0  # Cho luật 50 nước
        self.fullmove_number: int = 1
        
        # Lịch sử nước đi để hoàn tác
        self.move_history: List[dict] = []
        
        # Vị trí vua (để phát hiện chiếu nhanh hơn)
        self.king_positions: dict = {Color.WHITE: None, Color.BLACK: None}
        
        if fen:
            self.load_from_fen(fen)
        else:
            self.load_from_fen(self.STARTING_POSITION)
    
    def get_piece(self, square: int) -> Optional[Piece]:
        """Lấy quân cờ tại ô cho trước"""
        if not (0 <= square < 64):
            return None
        return self.squares[square]
    
    def set_piece(self, square: int, piece: Optional[Piece]):
        """Đặt quân cờ lên bàn cờ"""
        if not (0 <= square < 64):
            return
        
        # Cập nhật theo dõi vị trí vua
        if piece and piece.type == PieceType.KING:
            self.king_positions[piece.color] = square
        elif self.squares[square] and self.squares[square].type == PieceType.KING:
            # Xóa vua
            if self.king_positions[self.squares[square].color] == square:
                self.king_positions[self.squares[square].color] = None
        
        self.squares[square] = piece
    
    def load_from_fen(self, fen: str):
        """Tải trạng thái bàn cờ từ chuỗi FEN"""
        parts = fen.split()
        if len(parts) < 1:
            return
        
        # Parse vị trí bàn cờ
        ranks = parts[0].split('/')
        square = 56  # Bắt đầu tại a8 (rank 8, file a)
        
        for rank_str in ranks:
            for char in rank_str:
                if char.isdigit():
                    # Ô trống
                    square += int(char)
                else:
                    # Quân cờ
                    color = Color.WHITE if char.isupper() else Color.BLACK
                    piece_chars = {
                        'P': PieceType.PAWN, 'N': PieceType.KNIGHT,
                        'B': PieceType.BISHOP, 'R': PieceType.ROOK,
                        'Q': PieceType.QUEEN, 'K': PieceType.KING
                    }
                    piece_type = piece_chars[char.upper()]
                    self.set_piece(square, Piece(color, piece_type))
                    square += 1
            
            square -= 16  # Chuyển sang rank tiếp theo (lùi 8, rồi tiến 8 nữa)
        
        # Parse màu đang đi
        if len(parts) > 1:
            self.current_turn = Color.WHITE if parts[1] == 'w' else Color.BLACK
        
        # Parse quyền nhập thành
        if len(parts) > 2:
            self.castling_rights[Color.WHITE]['kingside'] = 'K' in parts[2]
            self.castling_rights[Color.WHITE]['queenside'] = 'Q' in parts[2]
            self.castling_rights[Color.BLACK]['kingside'] = 'k' in parts[2]
            self.castling_rights[Color.BLACK]['queenside'] = 'q' in parts[2]
        
        # Parse ô đích en passant
        if len(parts) > 3 and parts[3] != '-':
            self.en_passant_target = Square.from_string(parts[3])
        else:
            self.en_passant_target = None
        
        # Parse halfmove clock và fullmove number
        if len(parts) > 4:
            self.halfmove_clock = int(parts[4])
        if len(parts) > 5:
            self.fullmove_number = int(parts[5])
    
    def to_fen(self) -> str:
        """Chuyển đổi trạng thái bàn cờ sang chuỗi FEN"""
        fen_parts = []
        
        # Board position
        for rank in range(7, -1, -1):
            rank_str = ""
            empty_count = 0
            
            for file in range(8):
                square = rank * 8 + file
                piece = self.get_piece(square)
                
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    
                    piece_chars = {
                        (Color.WHITE, PieceType.PAWN): 'P',
                        (Color.WHITE, PieceType.KNIGHT): 'N',
                        (Color.WHITE, PieceType.BISHOP): 'B',
                        (Color.WHITE, PieceType.ROOK): 'R',
                        (Color.WHITE, PieceType.QUEEN): 'Q',
                        (Color.WHITE, PieceType.KING): 'K',
                        (Color.BLACK, PieceType.PAWN): 'p',
                        (Color.BLACK, PieceType.KNIGHT): 'n',
                        (Color.BLACK, PieceType.BISHOP): 'b',
                        (Color.BLACK, PieceType.ROOK): 'r',
                        (Color.BLACK, PieceType.QUEEN): 'q',
                        (Color.BLACK, PieceType.KING): 'k',
                    }
                    rank_str += piece_chars[(piece.color, piece.type)]
            
            if empty_count > 0:
                rank_str += str(empty_count)
            fen_parts.append(rank_str)
        
        fen = '/'.join(fen_parts)
        
        # Màu đang đi
        fen += ' ' + ('w' if self.current_turn == Color.WHITE else 'b')
        
        # Quyền nhập thành
        castling = ""
        if self.castling_rights[Color.WHITE]['kingside']:
            castling += 'K'
        if self.castling_rights[Color.WHITE]['queenside']:
            castling += 'Q'
        if self.castling_rights[Color.BLACK]['kingside']:
            castling += 'k'
        if self.castling_rights[Color.BLACK]['queenside']:
            castling += 'q'
        fen += ' ' + (castling if castling else '-')
        
        # Ô đích en passant
        if self.en_passant_target is not None:
            fen += ' ' + Square.to_string(self.en_passant_target)
        else:
            fen += ' -'
        
        # Halfmove clock và fullmove number
        fen += f' {self.halfmove_clock} {self.fullmove_number}'
        
        return fen
    
    def apply_move(self, move: Move):
        """
        Áp dụng một nước đi lên bàn cờ.
        Lưu trạng thái để hoàn tác.
        """
        # Lưu trạng thái để hoàn tác
        history_entry = {
            'move': move,
            'captured_piece': self.get_piece(move.to_square),
            'old_en_passant': self.en_passant_target,
            'old_halfmove': self.halfmove_clock,
            'old_castling': {
                Color.WHITE: self.castling_rights[Color.WHITE].copy(),
                Color.BLACK: self.castling_rights[Color.BLACK].copy()
            },
            'old_king_positions': self.king_positions.copy(),
        }
        
        piece = self.get_piece(move.from_square)
        if piece is None:
            raise ValueError(f"No piece at square {move.from_square}")
        
        # Đặt lại ô đích en passant
        self.en_passant_target = None
        
        # Xử lý nhập thành
        if move.is_castle:
            self._apply_castling(move)
            history_entry['rook_from'] = None
            history_entry['rook_to'] = None
        else:
            # Nước đi thường
            captured_piece = self.get_piece(move.to_square)
            
            # Xử lý bắt tốt qua đường (en passant)
            if move.is_en_passant:
                # Xóa tốt bị bắt (một rank phía sau ô đích en passant)
                # Cho trắng: e5->d6 bắt d5, nên d6-8=d5
                # Cho đen: e4->d3 bắt d4, nên d3+8=d4
                capture_square = move.to_square + (-8 if piece.color == Color.WHITE else 8)
                self.set_piece(capture_square, None)
                history_entry['en_passant_capture_square'] = capture_square
            else:
                history_entry['en_passant_capture_square'] = None
            
            # Di chuyển quân cờ
            self.set_piece(move.to_square, piece)
            self.set_piece(move.from_square, None)
            
            # Xử lý phong cấp
            if move.promotion:
                promoted_piece = Piece(piece.color, move.promotion)
                self.set_piece(move.to_square, promoted_piece)
                history_entry['promoted_from'] = piece.type
            else:
                history_entry['promoted_from'] = None
            
            # Xử lý tốt nhảy 2 ô để tạo en passant
            if piece.type == PieceType.PAWN:
                rank_from, _ = Square.to_rank_file(move.from_square)
                rank_to, _ = Square.to_rank_file(move.to_square)
                if abs(rank_to - rank_from) == 2:
                    # Tốt nhảy 2 ô - đặt ô đích en passant
                    en_passant_rank = (rank_from + rank_to) // 2
                    self.en_passant_target = Square.from_rank_file(
                        en_passant_rank, Square.to_rank_file(move.from_square)[1]
                    )
        
        # Cập nhật quyền nhập thành
        self._update_castling_rights(move, piece)
        
        # Cập nhật đồng hồ nước đi
        if piece.type == PieceType.PAWN or history_entry['captured_piece']:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        if self.current_turn == Color.BLACK:
            self.fullmove_number += 1
        
        # Đổi lượt
        self.current_turn = self.current_turn.opponent()
        
        # Lưu lịch sử
        self.move_history.append(history_entry)
    
    def undo_move(self):
        """
        Hoàn tác nước đi cuối cùng.
        Khôi phục trạng thái bàn cờ từ lịch sử.
        """
        if not self.move_history:
            raise ValueError("No moves to undo")
        
        history_entry = self.move_history.pop()
        move = history_entry['move']
        
        # Khôi phục lượt (trước nước đi)
        self.current_turn = self.current_turn.opponent()
        
        # Khôi phục đồng hồ nước đi
        self.halfmove_clock = history_entry['old_halfmove']
        if self.current_turn == Color.BLACK:
            self.fullmove_number -= 1
        
        # Khôi phục en passant
        self.en_passant_target = history_entry['old_en_passant']
        
        # Khôi phục quyền nhập thành
        self.castling_rights = history_entry['old_castling']
        
        # Xử lý hoàn tác nhập thành
        if move.is_castle:
            self._undo_castling(move)
        else:
            # Lấy quân cờ từ ô đích
            piece = self.get_piece(move.to_square)
            if piece is None:
                raise ValueError(f"No piece at destination square {move.to_square}")
            
            # Xử lý hoàn tác phong cấp
            if history_entry['promoted_from']:
                piece = Piece(piece.color, history_entry['promoted_from'])
            
            # Di chuyển quân cờ về lại (tạm thời tắt theo dõi vua)
            # Xóa quân cờ từ ô đích trước
            dest_piece = self.get_piece(move.to_square)
            self.squares[move.to_square] = history_entry['captured_piece']
            
            # Đặt quân cờ tại ô xuất phát
            self.squares[move.from_square] = piece
            
            # Xử lý hoàn tác en passant
            if history_entry.get('en_passant_capture_square') is not None:
                # Khôi phục tốt bị bắt
                captured_pawn = Piece(
                    self.current_turn.opponent(),
                    PieceType.PAWN
                )
                self.squares[history_entry['en_passant_capture_square']] = captured_pawn
        
        # Tính lại vị trí vua từ trạng thái bàn cờ
        self._update_king_positions()
    
    def _apply_castling(self, move: Move):
        """Áp dụng nước nhập thành"""
        king = self.get_piece(move.from_square)
        if king is None or king.type != PieceType.KING:
            raise ValueError("Castling requires a king")
        
        color = king.color
        rank = Square.to_rank_file(move.from_square)[0]
        
        if move.castle_side == 'kingside':
            # Di chuyển vua
            self.set_piece(rank * 8 + 6, king)  # g1 hoặc g8
            self.set_piece(rank * 8 + 4, None)  # e1 hoặc e8
            
            # Di chuyển xe
            rook = self.get_piece(rank * 8 + 7)  # h1 hoặc h8
            self.set_piece(rank * 8 + 5, rook)  # f1 hoặc f8
            self.set_piece(rank * 8 + 7, None)  # h1 hoặc h8
        else:  # queenside
            # Di chuyển vua
            self.set_piece(rank * 8 + 2, king)  # c1 hoặc c8
            self.set_piece(rank * 8 + 4, None)  # e1 hoặc e8
            
            # Di chuyển xe
            rook = self.get_piece(rank * 8)  # a1 hoặc a8
            self.set_piece(rank * 8 + 3, rook)  # d1 hoặc d8
            self.set_piece(rank * 8, None)  # a1 hoặc a8
    
    def _undo_castling(self, move: Move):
        """Hoàn tác nước nhập thành"""
        king = self.get_piece(move.to_square)
        if king is None or king.type != PieceType.KING:
            raise ValueError("Cannot undo castling - no king at destination")
        
        color = king.color
        rank = Square.to_rank_file(move.from_square)[0]
        
        if move.castle_side == 'kingside':
            # Khôi phục vua
            self.set_piece(rank * 8 + 4, king)  # e1 hoặc e8
            self.set_piece(rank * 8 + 6, None)  # g1 hoặc g8
            
            # Khôi phục xe
            rook = self.get_piece(rank * 8 + 5)  # f1 hoặc f8
            self.set_piece(rank * 8 + 7, rook)  # h1 hoặc h8
            self.set_piece(rank * 8 + 5, None)  # f1 hoặc f8
        else:  # queenside
            # Khôi phục vua
            self.set_piece(rank * 8 + 4, king)  # e1 hoặc e8
            self.set_piece(rank * 8 + 2, None)  # c1 hoặc c8
            
            # Khôi phục xe
            rook = self.get_piece(rank * 8 + 3)  # d1 hoặc d8
            self.set_piece(rank * 8, rook)  # a1 hoặc a8
            self.set_piece(rank * 8 + 3, None)  # d1 hoặc d8
    
    def _update_king_positions(self):
        """Tính lại vị trí vua từ trạng thái bàn cờ"""
        self.king_positions[Color.WHITE] = None
        self.king_positions[Color.BLACK] = None
        
        for square in range(64):
            piece = self.get_piece(square)
            if piece and piece.type == PieceType.KING:
                self.king_positions[piece.color] = square
    
    def _update_castling_rights(self, move: Move, piece: Piece):
        """Cập nhật quyền nhập thành sau một nước đi"""
        # Nếu vua di chuyển, mất tất cả quyền nhập thành
        if piece.type == PieceType.KING:
            self.castling_rights[piece.color]['kingside'] = False
            self.castling_rights[piece.color]['queenside'] = False
        
        # Nếu xe di chuyển, mất quyền nhập thành ở phía đó
        if piece.type == PieceType.ROOK:
            rank, file = Square.to_rank_file(move.from_square)
            if piece.color == Color.WHITE:
                if rank == 0:
                    if file == 0:  # a1
                        self.castling_rights[Color.WHITE]['queenside'] = False
                    elif file == 7:  # h1
                        self.castling_rights[Color.WHITE]['kingside'] = False
            else:  # ĐEN
                if rank == 7:
                    if file == 0:  # a8
                        self.castling_rights[Color.BLACK]['queenside'] = False
                    elif file == 7:  # h8
                        self.castling_rights[Color.BLACK]['kingside'] = False
        
        # Nếu xe bị bắt, mất quyền nhập thành ở phía đó
        captured = self.get_piece(move.to_square)
        if captured and captured.type == PieceType.ROOK:
            rank, file = Square.to_rank_file(move.to_square)
            if captured.color == Color.WHITE:
                if rank == 0:
                    if file == 0:  # a1
                        self.castling_rights[Color.WHITE]['queenside'] = False
                    elif file == 7:  # h1
                        self.castling_rights[Color.WHITE]['kingside'] = False
            else:  # ĐEN
                if rank == 7:
                    if file == 0:  # a8
                        self.castling_rights[Color.BLACK]['queenside'] = False
                    elif file == 7:  # h8
                        self.castling_rights[Color.BLACK]['kingside'] = False
    
    def __repr__(self):
        """Biểu diễn chuỗi của bàn cờ"""
        result = "\n   a b c d e f g h\n"
        for rank in range(7, -1, -1):
            result += f"{rank + 1}  "
            for file in range(8):
                square = rank * 8 + file
                piece = self.get_piece(square)
                if piece is None:
                    result += ". "
                else:
                    piece_chars = {
                        (Color.WHITE, PieceType.PAWN): 'P',
                        (Color.WHITE, PieceType.KNIGHT): 'N',
                        (Color.WHITE, PieceType.BISHOP): 'B',
                        (Color.WHITE, PieceType.ROOK): 'R',
                        (Color.WHITE, PieceType.QUEEN): 'Q',
                        (Color.WHITE, PieceType.KING): 'K',
                        (Color.BLACK, PieceType.PAWN): 'p',
                        (Color.BLACK, PieceType.KNIGHT): 'n',
                        (Color.BLACK, PieceType.BISHOP): 'b',
                        (Color.BLACK, PieceType.ROOK): 'r',
                        (Color.BLACK, PieceType.QUEEN): 'q',
                        (Color.BLACK, PieceType.KING): 'k',
                    }
                    result += piece_chars[(piece.color, piece.type)] + " "
            result += f" {rank + 1}\n"
        result += "   a b c d e f g h\n"
        result += f"Turn: {'White' if self.current_turn == Color.WHITE else 'Black'}\n"
        return result

