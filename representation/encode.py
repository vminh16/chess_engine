import numpy as np
from core.constants import PieceType, Color

def encode_board(board):
    """
    Chuyển đổi trạng thái bàn cờ thành Tensor 8x8x18.
    Góc nhìn: Luôn xoay về phía người chơi hiện tại (Current Player).
    """
    # Khởi tạo tensor 18 lớp, kích thước 8x8
    # Shape: (18, 8, 8) cho PyTorch (Channels First) hoặc (8, 8, 18) cho Keras
    # Ở đây tôi dùng (18, 8, 8) theo chuẩn PyTorch phổ biến
    X = np.zeros((18, 8, 8), dtype=np.float32)
    
    current_color = board.current_turn
    opponent_color = Color.BLACK if current_color == Color.WHITE else Color.WHITE
    
    # Channel 0-5: Quân mình (P, N, B, R, Q, K)
    # Channel 6-11: Quân địch
    
    piece_types = [PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP, 
                   PieceType.ROOK, PieceType.QUEEN, PieceType.KING]
    
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            piece = board.squares[square]
            
            if piece:
                # Tính toán channel index
                plane_idx = -1
                if piece.type in piece_types:
                    type_idx = piece_types.index(piece.type)
                    
                    if piece.color == current_color:
                        plane_idx = type_idx       # 0-5
                    else:
                        plane_idx = type_idx + 6   # 6-11
                
                if plane_idx != -1:
                    #  Nếu là phe Đen, cần lật bàn cờ (Flip board)
                    # để Model luôn nhìn thấy quân mình ở phía dưới.
                    if current_color == Color.BLACK:
                        r_idx = 7 - rank # Lật hàng 0->7, 7->0
                        f_idx = file     # Hoặc lật cả file (7-file) tùy cách bạn train
                    else:
                        r_idx = rank
                        f_idx = file
                        
                    X[plane_idx][r_idx][f_idx] = 1.0


    
    # Helper để fill full channel
    def fill_plane(idx, value):
        if value:
            X[idx].fill(1.0)
            
    # Channel 12: Lượt đi (Optional nếu đã flip board, nhưng giữ lại cũng tốt)
    # 1 nếu là Trắng, 0 nếu là Đen
    fill_plane(12, 1.0 if current_color == Color.WHITE else 0.0)
    
    # Channel 13-16: Castling Rights
    # Cần map đúng theo góc nhìn (My King, My Queen, Opp King, Opp Queen)
    rights = board.castling_rights
    
    # Quyền của Mình
    fill_plane(13, rights[current_color]['kingside'])
    fill_plane(14, rights[current_color]['queenside'])
    
    # Quyền của Địch
    fill_plane(15, rights[opponent_color]['kingside'])
    fill_plane(16, rights[opponent_color]['queenside'])
    
    # Channel 17: En Passant Target
    if board.en_passant_target is not None:
        ep_rank, ep_file = divmod(board.en_passant_target, 8)
        if current_color == Color.BLACK:
            ep_rank = 7 - ep_rank
        X[17][ep_rank][ep_file] = 1.0
        
    return X