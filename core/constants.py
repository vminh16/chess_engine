"""
Hằng số và enum cho chess engine.
Định nghĩa loại quân cờ, màu sắc, và hướng di chuyển.
"""
from enum import IntEnum, Enum


class Color(IntEnum):
    """Màu sắc quân cờ"""
    WHITE = 0
    BLACK = 1
    
    def opponent(self):
        """Trả về màu đối thủ"""
        return Color.BLACK if self == Color.WHITE else Color.WHITE


class PieceType(IntEnum):
    """Loại quân cờ"""
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6
    EMPTY = 0


class Piece:
    """Đại diện cho một quân cờ với màu sắc và loại"""
    def __init__(self, color: Color, piece_type: PieceType):
        self.color = color
        self.type = piece_type
    
    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.color == other.color and self.type == other.type
    
    def __repr__(self):
        color_str = "W" if self.color == Color.WHITE else "B"
        type_str = ["", "P", "N", "B", "R", "Q", "K"][self.type]
        return f"{color_str}{type_str}"


class Square(IntEnum):
    """Các ô trên bàn cờ (0-63)"""
    A1, B1, C1, D1, E1, F1, G1, H1 = range(0, 8)
    A2, B2, C2, D2, E2, F2, G2, H2 = range(8, 16)
    A3, B3, C3, D3, E3, F3, G3, H3 = range(16, 24)
    A4, B4, C4, D4, E4, F4, G4, H4 = range(24, 32)
    A5, B5, C5, D5, E5, F5, G5, H5 = range(32, 40)
    A6, B6, C6, D6, E6, F6, G6, H6 = range(40, 48)
    A7, B7, C7, D7, E7, F7, G7, H7 = range(48, 56)
    A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)
    
    @classmethod
    def from_rank_file(cls, rank: int, file: int) -> int:
        """Chuyển đổi rank (0-7) và file (0-7) sang chỉ số ô"""
        if not (0 <= rank < 8 and 0 <= file < 8):
            return None
        return rank * 8 + file
    
    @classmethod
    def to_rank_file(cls, square: int) -> tuple:
        """Chuyển đổi chỉ số ô sang (rank, file)"""
        if not (0 <= square < 64):
            return None
        return (square // 8, square % 8)
    
    @classmethod
    def from_string(cls, square_str: str) -> int:
        """Chuyển đổi ký hiệu đại số (ví dụ: 'e4') sang chỉ số ô"""
        if len(square_str) != 2:
            return None
        file = ord(square_str[0].lower()) - ord('a')
        rank = int(square_str[1]) - 1
        return cls.from_rank_file(rank, file)
    
    @classmethod
    def to_string(cls, square: int) -> str:
        """Chuyển đổi chỉ số ô sang ký hiệu đại số"""
        if not (0 <= square < 64):
            return None
        rank, file = cls.to_rank_file(square)
        return chr(ord('a') + file) + str(rank + 1)


class Direction:
    """Hướng di chuyển trên bàn cờ"""
    # Hướng trực giao (ngang/dọc)
    NORTH = 8
    SOUTH = -8
    EAST = 1
    WEST = -1
    
    # Hướng chéo
    NORTHEAST = 9
    NORTHWEST = 7
    SOUTHEAST = -7
    SOUTHWEST = -9
    
    # Nước đi của mã (knight)
    KNIGHT_MOVES = [
        NORTH + NORTH + EAST,   # 17
        NORTH + NORTH + WEST,   # 15
        EAST + EAST + NORTH,    # 10
        EAST + EAST + SOUTH,    # -6
        SOUTH + SOUTH + EAST,   # -15
        SOUTH + SOUTH + WEST,   # -17
        WEST + WEST + NORTH,    # 6
        WEST + WEST + SOUTH,    # -10
    ]
    
    # Tất cả 8 hướng
    ALL_DIRECTIONS = [NORTH, NORTHEAST, EAST, SOUTHEAST, 
                      SOUTH, SOUTHWEST, WEST, NORTHWEST]
    
    # Chỉ hướng trực giao
    ORTHOGONAL = [NORTH, EAST, SOUTH, WEST]
    
    # Chỉ hướng chéo
    DIAGONAL = [NORTHEAST, SOUTHEAST, SOUTHWEST, NORTHWEST]


# Giá trị quân cờ cho đánh giá (tùy chọn, dùng trong evaluation)
PIECE_VALUES = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 300,
    PieceType.BISHOP: 300,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 20000,
}

