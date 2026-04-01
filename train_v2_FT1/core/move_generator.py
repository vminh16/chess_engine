"""
Sinh nước đi cho chess engine.
Sinh tất cả nước đi hợp lệ từ một vị trí bàn cờ cho trước.
"""
from typing import List, Optional
from .constants import Color, PieceType, Piece, Square, Direction
from .move import Move
from .board import Board
from .rules import (
    is_legal_move, can_castle_kingside, can_castle_queenside,
    is_en_passant_legal, is_in_check
)


class MoveGenerator:
    """Sinh các nước đi hợp lệ"""
    
    def __init__(self, board: Board):
        self.board = board
    
    def generate_legal_moves(self, color: Optional[Color] = None) -> List[Move]:
        """
        Sinh tất cả nước đi hợp lệ cho màu cho trước.
        
        Args:
            color: Màu để sinh nước đi (mặc định là lượt hiện tại)
        
        Returns:
            Danh sách nước đi hợp lệ
        """
        if color is None:
            color = self.board.current_turn
        
        moves = []
        
        # Sinh nước đi cho mỗi quân cờ của màu cho trước
        for square in range(64):
            piece = self.board.get_piece(square)
            if piece and piece.color == color:
                moves.extend(self._generate_moves_for_piece(square, piece))
        
        # Lọc chỉ lấy nước đi hợp lệ (không để vua trong chiếu)
        legal_moves = [move for move in moves if is_legal_move(self.board, move)]
        
        return legal_moves
    
    def _generate_moves_for_piece(self, square: int, piece: Piece) -> List[Move]:
        """
        Sinh tất cả nước đi giả hợp lệ cho một quân cờ tại ô cho trước.
        
        Args:
            square: Chỉ số ô (0-63)
            piece: Quân cờ để sinh nước đi
        
        Returns:
            Danh sách nước đi giả hợp lệ (có thể để vua trong chiếu)
        """
        moves = []
        
        if piece.type == PieceType.PAWN:
            moves.extend(self._generate_pawn_moves(square, piece))
        elif piece.type == PieceType.KNIGHT:
            moves.extend(self._generate_knight_moves(square, piece))
        elif piece.type == PieceType.BISHOP:
            moves.extend(self._generate_bishop_moves(square, piece))
        elif piece.type == PieceType.ROOK:
            moves.extend(self._generate_rook_moves(square, piece))
        elif piece.type == PieceType.QUEEN:
            moves.extend(self._generate_queen_moves(square, piece))
        elif piece.type == PieceType.KING:
            moves.extend(self._generate_king_moves(square, piece))
        
        return moves
    
    def _generate_pawn_moves(self, square: int, piece: Piece) -> List[Move]:
        """Sinh nước đi của tốt bao gồm phong cấp, en passant, và nhảy 2 ô"""
        moves = []
        rank, file = Square.to_rank_file(square)
        
        # Xác định hướng tiến
        forward = 8 if piece.color == Color.WHITE else -8
        start_rank = 1 if piece.color == Color.WHITE else 6
        promotion_rank = 7 if piece.color == Color.WHITE else 0
        
        # Nước đi tiến một ô
        target_square = square + forward
        if 0 <= target_square < 64:
            target_piece = self.board.get_piece(target_square)
            if target_piece is None:
                target_rank, _ = Square.to_rank_file(target_square)
                if target_rank == promotion_rank:
                    # Promotion
                    for prom_type in [PieceType.QUEEN, PieceType.ROOK, 
                                     PieceType.BISHOP, PieceType.KNIGHT]:
                        moves.append(Move(square, target_square, promotion=prom_type))
                else:
                    moves.append(Move(square, target_square))
        
        # Nước đi tiến 2 ô từ vị trí khởi đầu
        if rank == start_rank:
            target_square = square + forward * 2
            if 0 <= target_square < 64:
                intermediate_square = square + forward
                if (self.board.get_piece(intermediate_square) is None and
                    self.board.get_piece(target_square) is None):
                    moves.append(Move(square, target_square))
        
        # Nước đi bắt (chéo)
        capture_offsets = [forward + 1, forward - 1] if piece.color == Color.WHITE else [forward + 1, forward - 1]
        for offset in capture_offsets:
            target_square = square + offset
            if target_square < 0 or target_square >= 64:
                continue
            
            target_rank, target_file = Square.to_rank_file(target_square)
            if target_rank < 0 or target_rank >= 8 or target_file < 0 or target_file >= 8:
                continue
            
            # Kiểm tra xem nước đi chéo có hợp lệ không (một file khác biệt)
            if abs(file - target_file) != 1:
                continue
            
            target_piece = self.board.get_piece(target_square)
            if target_piece and target_piece.color != piece.color:
                # Capture
                target_rank, _ = Square.to_rank_file(target_square)
                if target_rank == promotion_rank:
                    # Promotion capture
                    for prom_type in [PieceType.QUEEN, PieceType.ROOK,
                                     PieceType.BISHOP, PieceType.KNIGHT]:
                        moves.append(Move(square, target_square, promotion=prom_type))
                else:
                    moves.append(Move(square, target_square))
        
        # En passant
        if self.board.en_passant_target is not None:
            ep_square = self.board.en_passant_target
            ep_rank, ep_file = Square.to_rank_file(ep_square)
            
            # Ô đích en passant là một ô chéo về phía trước từ vị trí hiện tại
            # Kiểm tra xem chúng ta có ở rank đúng không (3 cho trắng, 4 cho đen khi bắt)
            expected_rank = 4 if piece.color == Color.WHITE else 3
            if rank == expected_rank and abs(file - ep_file) == 1:
                # Kiểm tra xem có tốt để bắt không (phía sau ô đích en passant)
                capture_square = ep_square - forward  # Ô có tốt để bắt
                if 0 <= capture_square < 64:
                    captured_pawn = self.board.get_piece(capture_square)
                    if (captured_pawn and 
                        captured_pawn.type == PieceType.PAWN and
                        captured_pawn.color != piece.color):
                        ep_move = Move(square, ep_square, is_en_passant=True)
                        # Lưu ý: is_legal_move sẽ được gọi sau để lọc nước đi
                        moves.append(ep_move)
        
        return moves
    
    def _generate_knight_moves(self, square: int, piece: Piece) -> List[Move]:
        """Generate knight moves"""
        moves = []
        rank, file = Square.to_rank_file(square)
        
        for offset in Direction.KNIGHT_MOVES:
            target_square = square + offset
            if target_square < 0 or target_square >= 64:
                continue
            
            target_rank, target_file = Square.to_rank_file(target_square)
            if target_rank < 0 or target_rank >= 8 or target_file < 0 or target_file >= 8:
                continue
            
            # Xác thực nước đi của mã (hình chữ L)
            rank_diff = abs(rank - target_rank)
            file_diff = abs(file - target_file)
            if not ((rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2)):
                continue
            
            target_piece = self.board.get_piece(target_square)
            if target_piece is None or target_piece.color != piece.color:
                moves.append(Move(square, target_square))
        
        return moves
    
    def _generate_bishop_moves(self, square: int, piece: Piece) -> List[Move]:
        """Sinh nước đi của tượng (trượt chéo)"""
        moves = []
        
        for direction in Direction.DIAGONAL:
            for distance in range(1, 8):
                target_square = square + direction * distance
                if target_square < 0 or target_square >= 64:
                    break
                
                # Kiểm tra xem nước đi có bọc quanh bàn cờ không
                rank, file = Square.to_rank_file(square)
                target_rank, target_file = Square.to_rank_file(target_square)
                if abs(rank - target_rank) != distance or abs(file - target_file) != distance:
                    break
                
                target_piece = self.board.get_piece(target_square)
                if target_piece is None:
                    moves.append(Move(square, target_square))
                elif target_piece.color != piece.color:
                    moves.append(Move(square, target_square))
                    break  # Can capture, but can't continue
                else:
                    break  # Blocked by own piece
        
        return moves
    
    def _generate_rook_moves(self, square: int, piece: Piece) -> List[Move]:
        """Sinh nước đi của xe (trượt trực giao)"""
        moves = []
        
        for direction in Direction.ORTHOGONAL:
            for distance in range(1, 8):
                target_square = square + direction * distance
                if target_square < 0 or target_square >= 64:
                    break
                
                # Kiểm tra xem nước đi có bọc quanh bàn cờ không
                rank, file = Square.to_rank_file(square)
                target_rank, target_file = Square.to_rank_file(target_square)
                
                # Nước đi trực giao: cùng rank hoặc cùng file
                rank_diff = abs(rank - target_rank)
                file_diff = abs(file - target_file)
                if not ((rank_diff == distance and file_diff == 0) or 
                       (rank_diff == 0 and file_diff == distance)):
                    break
                
                target_piece = self.board.get_piece(target_square)
                if target_piece is None:
                    moves.append(Move(square, target_square))
                elif target_piece.color != piece.color:
                    moves.append(Move(square, target_square))
                    break  # Can capture, but can't continue
                else:
                    break  # Blocked by own piece
        
        return moves
    
    def _generate_queen_moves(self, square: int, piece: Piece) -> List[Move]:
        """Sinh nước đi của hậu (trượt trực giao + chéo)"""
        moves = []
        # Hậu di chuyển như cả xe và tượng
        moves.extend(self._generate_rook_moves(square, piece))
        moves.extend(self._generate_bishop_moves(square, piece))
        return moves
    
    def _generate_king_moves(self, square: int, piece: Piece) -> List[Move]:
        """Sinh nước đi của vua bao gồm nhập thành"""
        moves = []
        
        # Nước đi vua thường (một ô theo bất kỳ hướng nào)
        for direction in Direction.ALL_DIRECTIONS:
            target_square = square + direction
            if target_square < 0 or target_square >= 64:
                continue
            
            # Check if move wraps around board
            rank, file = Square.to_rank_file(square)
            target_rank, target_file = Square.to_rank_file(target_square)
            
            if abs(rank - target_rank) > 1 or abs(file - target_file) > 1:
                continue
            
            target_piece = self.board.get_piece(target_square)
            if target_piece is None or target_piece.color != piece.color:
                moves.append(Move(square, target_square))
        
        # Nước đi nhập thành
        if can_castle_kingside(self.board, piece.color):
            moves.append(Move(square, square + 2, is_castle=True, castle_side='kingside'))
        
        if can_castle_queenside(self.board, piece.color):
            moves.append(Move(square, square - 2, is_castle=True, castle_side='queenside'))
        
        return moves
    def generate_pseudo_captures(self) -> List[Move]:
        """
        Sinh các nước AN QUÂN (Captures) và PHONG CẤP (Promotions).
        Tối ưu hóa tốc độ cho Quiescence Search.
        Lưu ý: Đây là nước đi Pseudo-legal (chưa kiểm tra vua bị chiếu).
        """
        moves = []
        color = self.board.current_turn
        
        for square in range(64):
            piece = self.board.get_piece(square)
            if piece and piece.color == color:
                if piece.type == PieceType.PAWN:
                    moves.extend(self._gen_pawn_captures(square, piece))
                elif piece.type == PieceType.KNIGHT:
                    moves.extend(self._gen_step_captures(square, piece, Direction.KNIGHT_MOVES))
                elif piece.type == PieceType.BISHOP:
                    moves.extend(self._gen_slide_captures(square, piece, Direction.DIAGONAL))
                elif piece.type == PieceType.ROOK:
                    moves.extend(self._gen_slide_captures(square, piece, Direction.ORTHOGONAL))
                elif piece.type == PieceType.QUEEN:
                    # Hậu = Xe + Tượng
                    moves.extend(self._gen_slide_captures(square, piece, Direction.ORTHOGONAL))
                    moves.extend(self._gen_slide_captures(square, piece, Direction.DIAGONAL))
                elif piece.type == PieceType.KING:
                    moves.extend(self._gen_step_captures(square, piece, Direction.ALL_DIRECTIONS))
        
        return moves

    def _gen_pawn_captures(self, square: int, piece: Piece) -> List[Move]:
        """Chỉ sinh nước tốt ăn quân hoặc phong cấp"""
        moves = []
        rank, file = Square.to_rank_file(square)
        
        forward = 8 if piece.color == Color.WHITE else -8
        promotion_rank = 7 if piece.color == Color.WHITE else 0
        
        # 1. Kiểm tra ăn quân chéo (Captures)
        capture_offsets = [forward + 1, forward - 1]
        for offset in capture_offsets:
            target_square = square + offset
            
            # Kiểm tra biên bàn cờ
            if not (0 <= target_square < 64): continue
            
            target_rank, target_file = Square.to_rank_file(target_square)
            # Kiểm tra lệch dòng (ví dụ từ cột h ăn sang cột a bên kia)
            if abs(file - target_file) != 1: continue
            
            target_piece = self.board.get_piece(target_square)
            
            # Nếu có quân địch -> ĂN
            if target_piece and target_piece.color != piece.color:
                if target_rank == promotion_rank:
                    # Luôn lấy Queen promotion cho QS để đơn giản
                    for prom in [PieceType.QUEEN, PieceType.KNIGHT]: 
                        moves.append(Move(square, target_square, promotion=prom))
                else:
                    moves.append(Move(square, target_square))
                    
        # 2. En Passant (Luôn được coi là Capture)
        if self.board.en_passant_target is not None:
            ep_sq = self.board.en_passant_target
            ep_rank, ep_file = Square.to_rank_file(ep_sq)
            
            # Logic kiểm tra vị trí tốt để bắt En Passant
            expected_rank = 4 if piece.color == Color.WHITE else 3
            if rank == expected_rank and abs(file - ep_file) == 1:
                # Kiểm tra lại (đề phòng)
                if (ep_sq - square) in capture_offsets:
                     moves.append(Move(square, ep_sq, is_en_passant=True))

        # 3. Phong cấp (Promotions) - Kể cả không ăn quân cũng quan trọng trong QS
        target_square = square + forward
        if 0 <= target_square < 64:
            target_rank, _ = Square.to_rank_file(target_square)
            if target_rank == promotion_rank:
                if self.board.get_piece(target_square) is None:
                    # Phong cấp (không ăn quân)
                    for prom in [PieceType.QUEEN, PieceType.KNIGHT]:
                         moves.append(Move(square, target_square, promotion=prom))
        
        return moves

    def _gen_step_captures(self, square: int, piece: Piece, offsets: List[int]) -> List[Move]:
        """Dùng cho Mã và Vua: Chỉ lấy nước nhảy vào ô có quân địch"""
        moves = []
        rank, file = Square.to_rank_file(square)
        
        for offset in offsets:
            target_square = square + offset
            if not (0 <= target_square < 64): continue
            
            # Kiểm tra biên cho Mã và Vua
            t_rank, t_file = Square.to_rank_file(target_square)
            
            # Logic kiểm tra bước nhảy hợp lệ (tránh wrap-around)
            if piece.type == PieceType.KNIGHT:
                if abs(rank - t_rank) + abs(file - t_file) != 3: continue
            else: # King
                if abs(rank - t_rank) > 1 or abs(file - t_file) > 1: continue

            target_piece = self.board.get_piece(target_square)
            
            # TỐI ƯU: Chỉ thêm nếu ô đích có quân ĐỊCH
            if target_piece and target_piece.color != piece.color:
                moves.append(Move(square, target_square))
                
        return moves

    def _gen_slide_captures(self, square: int, piece: Piece, directions: List[int]) -> List[Move]:
        """Dùng cho Tượng, Xe, Hậu: Raycasting dừng lại khi gặp quân"""
        moves = []
        rank, file = Square.to_rank_file(square)
        
        for direction in directions:
            for distance in range(1, 8):
                target_square = square + direction * distance
                if not (0 <= target_square < 64): break
                
                # Kiểm tra biên (wrap-around)
                t_rank, t_file = Square.to_rank_file(target_square)
                
                # Logic kiểm tra đường thẳng/chéo
                if abs(rank - t_rank) != distance and abs(file - t_file) != distance:
                    # Nếu không phải đường thẳng (cho Xe) cũng không phải chéo đều (cho Tượng) -> Break
                    # (Logic wrap-around của bạn trong file gốc dùng abs diff check, ta giữ nguyên logic tương tự)
                    # Tuy nhiên cách check đơn giản nhất cho ray là:
                    r_diff = abs(rank - t_rank)
                    f_diff = abs(file - t_file)
                    is_diagonal = (r_diff == distance and f_diff == distance)
                    is_straight = (r_diff == distance and f_diff == 0) or (r_diff == 0 and f_diff == distance)
                    if not (is_diagonal or is_straight):
                        break

                target_piece = self.board.get_piece(target_square)
                
                if target_piece is None:
                    # Ô trống -> Bỏ qua (đây là điểm khác biệt với generate_moves thường)
                    continue 
                elif target_piece.color != piece.color:
                    # Gặp quân địch -> ĂN và DỪNG LẠI
                    moves.append(Move(square, target_square))
                    break
                else:
                    # Gặp quân mình -> DỪNG LẠI (Bị chặn)
                    break
        return moves


def generate_legal_moves(board: Board, color: Optional[Color] = None) -> List[Move]:
    """
    Hàm tiện ích để sinh nước đi hợp lệ.
    
    Args:
        board: Bàn cờ
        color: Màu để sinh nước đi (mặc định là lượt hiện tại)
    
    Returns:
        Danh sách nước đi hợp lệ
    """
    generator = MoveGenerator(board)
    return generator.generate_legal_moves(color)
def generate_capture_moves(board: Board, color: Optional[Color] = None) -> List[Move]:
    """
    Hàm tiện ích tối ưu để sinh nước đi bắt hợp lệ (cho Quiescence Search).
    """
    generator = MoveGenerator(board)
    
    # 1. Sinh các nước ăn quân thô (Pseudo-legal captures) - Rất nhanh
    pseudo_captures = generator.generate_pseudo_captures()
    
    # 2. Lọc lại tính hợp lệ (Is Legal)
    # Vì số lượng nước ăn quân thường rất ít (2-5 nước), việc check legal ở đây nhanh hơn nhiều
    # so với việc sinh 40 nước rồi lọc.
    legal_captures = [move for move in pseudo_captures if is_legal_move(board, move)]
    
    return legal_captures

