import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import random
from core.constants import Color
from core.board import Board


import random
from core.constants import Color, PieceType

class TranspositionTable:
    
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

    
    def __init__(self, size_limit=1000000):
        self.table = {}
        self.size_limit = size_limit
        
        # 1. Piece-Square: 64 ô x 12 loại quân
        self.zobrist_pieces = [[random.getrandbits(64) for _ in range(12)] for _ in range(64)]
        # 2. Side to move
        self.zobrist_side = random.getrandbits(64)
        # 3. Castling Rights: 16 trạng thái
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]
        # 4. En Passant: 8 cột (files)
        self.zobrist_en_passant = [random.getrandbits(64) for _ in range(8)]

    def _get_castling_index(self, castling_rights):
        """Chuyển đổi dict lồng nhau của Board sang bitmask 0-15"""
        idx = 0
        # Truy cập theo cấu trúc: board.castling_rights[Color.WHITE]['kingside']
        if castling_rights[Color.WHITE]['kingside']: idx |= 1
        if castling_rights[Color.WHITE]['queenside']: idx |= 2
        if castling_rights[Color.BLACK]['kingside']: idx |= 4
        if castling_rights[Color.BLACK]['queenside']: idx |= 8
        return idx

    def compute_hash(self, board):
        h = 0
        # 1. Hash quân cờ
        for i, piece in enumerate(board.squares):
            if piece is not None:
                # piece.type (int), piece.color (Enum Color)
                p_type = piece.type.value if hasattr(piece.type, 'value') else piece.type
                idx = (p_type - 1) + (6 if piece.color == Color.BLACK else 0)
                h ^= self.zobrist_pieces[i][idx]
        
        # 2. Hash lượt đi
        if board.current_turn == Color.BLACK:
            h ^= self.zobrist_side
            
        # 3.  Chuyển dict lồng nhau thành index
        c_idx = self._get_castling_index(board.castling_rights)
        h ^= self.zobrist_castling[c_idx]
        
        # 4. Hash En Passant (sử dụng en_passant_target là index ô)
        if board.en_passant_target is not None:
            file_idx = board.en_passant_target % 8
            h ^= self.zobrist_en_passant[file_idx]
            
        return h

    def store(self, hash_key, depth, value, flag, age, best_move=None, static_eval=None):
        """
        Lưu trạng thái vào bảng băm.
        static_eval: Giá trị đánh giá tĩnh (NN output) để cache lại.
        """
        if len(self.table) >= self.size_limit:
            self.table.clear() 

        existing = self.table.get(hash_key)
        
        # Logic cập nhật:
        # 1. Luôn ưu tiên lưu nếu depth mới >= depth cũ
        # 2. Hoặc nếu entry cũ đã quá cũ (age chênh lệch)
        # 3. Đối với static_eval: Nếu entry cũ chưa có static_eval mà entry mới có, thì cập nhật thêm vào.
        
        if existing is None:
            self.table[hash_key] = {
                'value': value,
                'depth': depth,
                'flag': flag,
                'age': age, 
                'best_move': best_move,
                'static_eval': static_eval # Lưu thêm cache NN
            }
        else:
            # Nếu chỉ muốn cập nhật static_eval mà không ghi đè search result quan trọng
            if static_eval is not None:
                existing['static_eval'] = static_eval
            
            # Chỉ ghi đè thông tin search (score, flag, best_move) nếu kết quả mới "xịn" hơn (depth cao hơn)
            if depth >= existing['depth'] or age > existing['age']:
                existing['value'] = value
                existing['depth'] = depth
                existing['flag'] = flag
                existing['age'] = age
                if best_move is not None:
                    existing['best_move'] = best_move

    def lookup(self, hash_key):
        return self.table.get(hash_key)