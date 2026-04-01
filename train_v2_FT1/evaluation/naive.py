"""
Đánh giá naive cho bàn cờ dựa trên material và side-to-move.
Trả về giá trị trong khoảng [-1, 1] từ góc nhìn của bên hiện tại.
"""
from core.board import Board
from core.constants import Color, PieceType


def piece_value(piece_type: PieceType) -> int:
    """Trả về giá trị material của loại quân"""
    if piece_type == PieceType.PAWN:
        return 1
    elif piece_type == PieceType.KNIGHT:
        return 3
    elif piece_type == PieceType.BISHOP:
        return 3
    elif piece_type == PieceType.ROOK:
        return 5
    elif piece_type == PieceType.QUEEN:
        return 9
    elif piece_type == PieceType.KING:
        return 0
    else:
        return 0


def calculate_material(board: Board, color: Color) -> int:
    """Tính tổng material cho một màu"""
    material = 0
    for square in range(64):
        piece = board.squares[square]
        if piece and piece.color == color:
            material += piece_value(piece.type)
    return material


def evaluate(board: Board) -> float:
    """
    Đánh giá bàn cờ từ góc nhìn của bên hiện tại.
    
    Chỉ dựa trên material difference và side-to-move.
    Giá trị được normalize về khoảng [-1, 1].
    
    Args:
        board: Bàn cờ cần đánh giá
        
    Returns:
        Giá trị float trong [-1, 1], dương nếu bên hiện tại có lợi
    """
    my_color = board.current_turn
    opp_color = my_color.opponent()
    
    my_material = calculate_material(board, my_color)
    opp_material = calculate_material(board, opp_color)
    
    # Material advantage, normalized by max possible material (39 per side)
    material_advantage = my_material - opp_material
    value = material_advantage / 39.0
    
    # Clamp to [-1, 1] to be safe
    return max(-1.0, min(1.0, value))