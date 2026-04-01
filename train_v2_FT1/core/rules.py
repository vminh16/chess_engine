"""
Luật cờ và kiểm tra tính hợp lệ.
Xử lý nhập thành, en passant, và phát hiện chiếu.
"""
from typing import List, Optional
from .constants import Color, PieceType, Piece, Square, Direction
from .move import Move
from .board import Board


def is_square_attacked(board: Board, square: int, by_color: Color) -> bool:
    """
    Kiểm tra xem một ô có bị tấn công bởi bất kỳ quân cờ nào của màu cho trước không.
    
    Args:
        board: Bàn cờ
        square: Ô cần kiểm tra (0-63)
        by_color: Màu của quân cờ để kiểm tra tấn công
    
    Returns:
        True nếu ô bị tấn công bởi quân cờ của màu cho trước
    """
    # Kiểm tra tấn công của tốt
    # Đối với tốt trắng: chúng tấn công chéo về phía trước (bắc), nên chúng ở các ô phía nam của mục tiêu
    # Đối với tốt đen: chúng tấn công chéo về phía trước (nam), nên chúng ở các ô phía bắc của mục tiêu
    for pawn_offset in [7, 9] if by_color == Color.WHITE else [-7, -9]:
        pawn_square = square - pawn_offset
        if pawn_square < 0 or pawn_square >= 64:
            continue
        
        # Kiểm tra xem tốt có ở rank và file đúng không
        rank, file = Square.to_rank_file(square)
        attack_rank, attack_file = Square.to_rank_file(pawn_square)
        
        if attack_rank < 0 or attack_rank >= 8 or attack_file < 0 or attack_file >= 8:
            continue
        
        # Tốt tấn công chéo (một rank và một file khác biệt)
        if abs(file - attack_file) != 1:
            continue
        
        # Đối với trắng: tốt phải ở một rank phía nam (số rank thấp hơn)
        # Đối với đen: tốt phải ở một rank phía bắc (số rank cao hơn)
        if by_color == Color.WHITE and attack_rank >= rank:
            continue
        if by_color == Color.BLACK and attack_rank <= rank:
            continue
        
        piece = board.get_piece(pawn_square)
        if piece and piece.color == by_color and piece.type == PieceType.PAWN:
            return True
    
    # Kiểm tra tấn công của mã (mã có thể tấn công từ 8 vị trí hình chữ L)
    for offset in Direction.KNIGHT_MOVES:
        knight_square = square + offset  # Kiểm tra các ô mà mã có thể tấn công TỪ
        if knight_square < 0 or knight_square >= 64:
            continue
        
        # Kiểm tra xem nước đi của mã có bọc quanh bàn cờ không
        rank, file = Square.to_rank_file(square)
        knight_rank, knight_file = Square.to_rank_file(knight_square)
        
        # Mã di chuyển chính xác 2 ô theo một hướng và 1 ô theo hướng kia
        rank_diff = abs(rank - knight_rank)
        file_diff = abs(file - knight_file)
        if not ((rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2)):
            continue
        
        piece = board.get_piece(knight_square)
        if piece and piece.color == by_color and piece.type == PieceType.KNIGHT:
            return True
    
    # Kiểm tra tấn công của quân trượt (xe, tượng, hậu)
    # Xe/Hậu (trực giao)
    for direction in Direction.ORTHOGONAL:
        for distance in range(1, 8):
            attack_square = square + direction * distance
            if attack_square < 0 or attack_square >= 64:
                break
            
            # Kiểm tra xem nước đi có bọc quanh bàn cờ không
            rank, file = Square.to_rank_file(square)
            attack_rank, attack_file = Square.to_rank_file(attack_square)
            
            # Xác thực hướng (phải ở cùng rank hoặc cùng file)
            rank_diff = abs(rank - attack_rank)
            file_diff = abs(file - attack_file)
            if not ((rank_diff == distance and file_diff == 0) or 
                   (rank_diff == 0 and file_diff == distance)):
                break
            
            piece = board.get_piece(attack_square)
            if piece:
                if (piece.color == by_color and 
                    (piece.type == PieceType.ROOK or piece.type == PieceType.QUEEN)):
                    return True
                break  # Bị chặn bởi bất kỳ quân cờ nào
    
    # Tượng/Hậu (chéo)
    for direction in Direction.DIAGONAL:
        for distance in range(1, 8):
            attack_square = square + direction * distance
            if attack_square < 0 or attack_square >= 64:
                break
            
            # Kiểm tra xem nước đi có bọc quanh bàn cờ không
            rank, file = Square.to_rank_file(square)
            attack_rank, attack_file = Square.to_rank_file(attack_square)
            
            # Xác thực hướng chéo
            rank_diff = abs(rank - attack_rank)
            file_diff = abs(file - attack_file)
            if rank_diff != distance or file_diff != distance:
                break
            
            piece = board.get_piece(attack_square)
            if piece:
                if (piece.color == by_color and 
                    (piece.type == PieceType.BISHOP or piece.type == PieceType.QUEEN)):
                    return True
                break  # Bị chặn bởi bất kỳ quân cờ nào
    
    # Kiểm tra tấn công của vua (các ô kề cận)
    for direction in Direction.ALL_DIRECTIONS:
        king_square = square + direction
        if king_square < 0 or king_square >= 64:
            continue
        
        # Kiểm tra xem nước đi có bọc quanh bàn cờ không
        rank, file = Square.to_rank_file(square)
        king_rank, king_file = Square.to_rank_file(king_square)
        
        if abs(rank - king_rank) > 1 or abs(file - king_file) > 1:
            continue
        
        piece = board.get_piece(king_square)
        if piece and piece.color == by_color and piece.type == PieceType.KING:
            return True
    
    return False


def is_in_check(board: Board, color: Color) -> bool:
    """
    Kiểm tra xem vua của màu cho trước có bị chiếu không.
    
    Args:
        board: Bàn cờ
        color: Màu cần kiểm tra
    
    Returns:
        True nếu vua bị chiếu
    """
    king_square = board.king_positions[color]
    if king_square is None:
        return False
    
    opponent_color = color.opponent()
    return is_square_attacked(board, king_square, opponent_color)


def is_legal_move(board: Board, move: Move) -> bool:
    """
    Kiểm tra xem một nước đi có hợp lệ không (không để vua của mình trong chiếu).
    
    Args:
        board: Bàn cờ
        move: Nước đi cần kiểm tra
    
    Returns:
        True nếu nước đi hợp lệ
    """
    # Áp dụng nước đi
    board.apply_move(move)
    
    # Kiểm tra xem vua của mình có bị chiếu không
    legal = not is_in_check(board, board.current_turn.opponent())
    
    # Hoàn tác nước đi
    board.undo_move()
    
    return legal


def can_castle_kingside(board: Board, color: Color) -> bool:
    """
    Kiểm tra xem nhập thành cánh vua có hợp lệ cho màu cho trước không.
    
    Args:
        board: Bàn cờ
        color: Màu cần kiểm tra
    
    Returns:
        True nếu nhập thành cánh vua hợp lệ
    """
    # Kiểm tra quyền nhập thành
    if not board.castling_rights[color]['kingside']:
        return False
    
    # Kiểm tra xem vua và xe có ở vị trí khởi đầu không
    rank = 0 if color == Color.WHITE else 7
    king_square = rank * 8 + 4  # e1 hoặc e8
    rook_square = rank * 8 + 7  # h1 hoặc h8
    
    king = board.get_piece(king_square)
    rook = board.get_piece(rook_square)
    
    if (not king or king.type != PieceType.KING or king.color != color):
        return False
    if (not rook or rook.type != PieceType.ROOK or rook.color != color):
        return False
    
    # Kiểm tra xem các ô giữa vua và xe có trống không
    if (board.get_piece(rank * 8 + 5) is not None or  # f1/f8
        board.get_piece(rank * 8 + 6) is not None):   # g1/g8
        return False
    
    # Kiểm tra xem vua có bị chiếu không
    if is_in_check(board, color):
        return False
    
    # Kiểm tra xem vua có đi qua hoặc đặt chân lên các ô bị tấn công không
    for square_offset in [5, 6]:  # f1/f8, g1/g8
        square = rank * 8 + square_offset
        if is_square_attacked(board, square, color.opponent()):
            return False
    
    return True


def can_castle_queenside(board: Board, color: Color) -> bool:
    """
    Kiểm tra xem nhập thành cánh hậu có hợp lệ cho màu cho trước không.
    
    Args:
        board: Bàn cờ
        color: Màu cần kiểm tra
    
    Returns:
        True nếu nhập thành cánh hậu hợp lệ
    """
    # Kiểm tra quyền nhập thành
    if not board.castling_rights[color]['queenside']:
        return False
    
    # Kiểm tra xem vua và xe có ở vị trí khởi đầu không
    rank = 0 if color == Color.WHITE else 7
    king_square = rank * 8 + 4  # e1 hoặc e8
    rook_square = rank * 8  # a1 hoặc a8
    
    king = board.get_piece(king_square)
    rook = board.get_piece(rook_square)
    
    if (not king or king.type != PieceType.KING or king.color != color):
        return False
    if (not rook or rook.type != PieceType.ROOK or rook.color != color):
        return False
    
    # Kiểm tra xem các ô giữa vua và xe có trống không
    if (board.get_piece(rank * 8 + 1) is not None or  # b1/b8
        board.get_piece(rank * 8 + 2) is not None or  # c1/c8
        board.get_piece(rank * 8 + 3) is not None):   # d1/d8
        return False
    
    # Kiểm tra xem vua có bị chiếu không
    if is_in_check(board, color):
        return False
    
    # Kiểm tra xem vua có đi qua hoặc đặt chân lên các ô bị tấn công không
    for square_offset in [2, 3, 4]:  # c1/c8, d1/d8, e1/e8
        square = rank * 8 + square_offset
        if is_square_attacked(board, square, color.opponent()):
            return False
    
    return True


def is_en_passant_legal(board: Board, move: Move) -> bool:
    """
    Kiểm tra xem nước bắt tốt qua đường có hợp lệ không.
    
    Args:
        board: Bàn cờ
        move: Nước đi cần kiểm tra (nên có is_en_passant=True)
    
    Returns:
        True nếu en passant hợp lệ
    """
    if not move.is_en_passant:
        return False
    
    piece = board.get_piece(move.from_square)
    if not piece or piece.type != PieceType.PAWN:
        return False
    
    # Kiểm tra xem ô đích en passant có được đặt không
    if board.en_passant_target != move.to_square:
        return False
    
    # Lấy ô của tốt bị bắt
    rank, file = Square.to_rank_file(move.to_square)
    captured_pawn_rank = rank + (1 if piece.color == Color.WHITE else -1)
    captured_pawn_square = Square.from_rank_file(captured_pawn_rank, file)
    
    # Kiểm tra xem có thực sự có tốt ở đó không
    captured_pawn = board.get_piece(captured_pawn_square)
    if (not captured_pawn or 
        captured_pawn.type != PieceType.PAWN or 
        captured_pawn.color == piece.color):
        return False
    
    # Áp dụng en passant và kiểm tra xem nó có để vua của mình trong chiếu không
    board.apply_move(move)
    legal = not is_in_check(board, board.current_turn.opponent())
    board.undo_move()
    
    return legal


def is_checkmate(board: Board, color: Color) -> bool:
    """
    Kiểm tra xem màu cho trước có bị chiếu hết không.
    
    Args:
        board: Bàn cờ
        color: Màu cần kiểm tra
    
    Returns:
        True nếu màu đó bị chiếu hết
    """
    # Phải bị chiếu
    if not is_in_check(board, color):
        return False
    
    # Phải không có nước đi hợp lệ nào
    # Điều này sẽ được kiểm tra bởi move_generator, nhưng chúng ta không thể import nó ở đây
    # để tránh phụ thuộc vòng tròn. Vì vậy chúng ta chỉ trả về False ở đây
    # và để move_generator xử lý việc phát hiện chiếu hết đầy đủ.
    return False  # Sẽ được triển khai đúng cách khi chúng ta có move generation


def is_stalemate(board: Board, color: Color) -> bool:
    """
    Kiểm tra xem màu cho trước có bị hòa do hết nước không.
    
    Args:
        board: Bàn cờ
        color: Màu cần kiểm tra
    
    Returns:
        True nếu màu đó bị hòa do hết nước
    """
    # Phải không bị chiếu
    if is_in_check(board, color):
        return False
    
    # Phải không có nước đi hợp lệ nào
    # Tương tự như chiếu hết, sẽ cần move_generator để triển khai đầy đủ
    return False

