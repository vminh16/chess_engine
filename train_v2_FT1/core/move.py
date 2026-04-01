"""
Biểu diễn nước đi cho chess engine.
Định nghĩa lớp Move và các loại nước đi.
"""
from typing import Optional
from .constants import Square, PieceType


class Move:
    """Đại diện cho một nước đi cờ"""
    
    def __init__(self, from_square: int, to_square: int, 
                 promotion: Optional[PieceType] = None,
                 is_en_passant: bool = False,
                 is_castle: bool = False,
                 castle_side: Optional[str] = None):
        """
        Khởi tạo một nước đi.
        
        Args:
            from_square: Ô xuất phát (0-63)
            to_square: Ô đích (0-63)
            promotion: Loại quân cờ cho phong cấp tốt (None nếu không phải phong cấp)
            is_en_passant: True nếu đây là nước bắt tốt qua đường
            is_castle: True nếu đây là nước nhập thành
            castle_side: 'kingside' hoặc 'queenside' cho nước nhập thành
        """
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion
        self.is_en_passant = is_en_passant
        self.is_castle = is_castle
        self.castle_side = castle_side
    def __str__(self):
        """Chuyển đổi nước đi sang định dạng chuỗi UCI"""
        return self.to_string()
    def __eq__(self, other):
        """Hai nước đi bằng nhau nếu tất cả thuộc tính khớp"""
        if not isinstance(other, Move):
            return False
        return (self.from_square == other.from_square and
                self.to_square == other.to_square and
                self.promotion == other.promotion and
                self.is_en_passant == other.is_en_passant and
                self.is_castle == other.is_castle and
                self.castle_side == other.castle_side)
    
    def __hash__(self):
        """Hash để dùng trong sets/dicts"""
        return hash((self.from_square, self.to_square, self.promotion,
                    self.is_en_passant, self.is_castle, self.castle_side))
    
    def __repr__(self):
        """Biểu diễn chuỗi của nước đi"""
        from_str = Square.to_string(self.from_square)
        to_str = Square.to_string(self.to_square)
        
        if self.promotion:
            prom_char = ['', 'P', 'N', 'B', 'R', 'Q', 'K'][self.promotion]
            return f"{from_str}{to_str}{prom_char}"
        
        if self.is_en_passant:
            return f"{from_str}{to_str}ep"
        
        if self.is_castle:
            side = 'O-O' if self.castle_side == 'kingside' else 'O-O-O'
            return f"{side} ({from_str}{to_str})"
        
        return f"{from_str}{to_str}"
    
    @classmethod
    def from_string(cls, move_str: str) -> Optional['Move']:
        """
        Parse một nước đi từ định dạng UCI (ví dụ: 'e2e4', 'e7e8q')
        hoặc ký hiệu đại số (ví dụ: 'e4', 'Nf3', 'O-O')
        
        Hiện tại chỉ hỗ trợ định dạng UCI.
        """
        if len(move_str) < 4:
            return None
        
        # Định dạng UCI: e2e4, e7e8q
        if len(move_str) >= 4:
            from_sq = Square.from_string(move_str[0:2])
            to_sq = Square.from_string(move_str[2:4])
            
            if from_sq is None or to_sq is None:
                return None
            
            promotion = None
            if len(move_str) == 5:
                prom_chars = {'q': PieceType.QUEEN, 'r': PieceType.ROOK,
                             'b': PieceType.BISHOP, 'n': PieceType.KNIGHT}
                promotion = prom_chars.get(move_str[4].lower())
            
            return cls(from_sq, to_sq, promotion=promotion)
        
        return None
    
    def to_string(self) -> str:
        """Chuyển đổi nước đi sang định dạng chuỗi UCI"""
        from_str = Square.to_string(self.from_square)
        to_str = Square.to_string(self.to_square)
        
        if self.promotion:
            prom_chars = {PieceType.QUEEN: 'q', PieceType.ROOK: 'r',
                         PieceType.BISHOP: 'b', PieceType.KNIGHT: 'n'}
            return f"{from_str}{to_str}{prom_chars[self.promotion]}"
        
        return f"{from_str}{to_str}"
    
    def is_promotion(self) -> bool:
        """Kiểm tra xem đây có phải nước phong cấp không"""
        return self.promotion is not None
    
    def is_capture(self, board) -> bool:
        """
        Kiểm tra xem nước đi này có bắt quân không.
        Cần board để kiểm tra ô đích.
        """
        if self.is_en_passant:
            return True
        return board.get_piece(self.to_square) is not None


class MoveType:
    """Hằng số cho các loại nước đi"""
    NORMAL = 0
    PROMOTION = 1
    CASTLING = 2
    EN_PASSANT = 3
    DOUBLE_PAWN_PUSH = 4

