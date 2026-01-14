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
        self.current_age = 0  # Dùng để đánh dấu thế hệ tìm kiếm
        
        # Seed cố định để Zobrist keys nhất quán giữa các lần chạy
        random.seed(12345)
        
        # 1. Piece-Square: 64 ô x 12 loại quân
        self.zobrist_pieces = [[random.getrandbits(64) for _ in range(12)] for _ in range(64)]
        # 2. Side to move
        self.zobrist_side = random.getrandbits(64)
        # 3. Castling Rights: 16 trạng thái
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]
        # 4. En Passant: 8 cột (files)
        self.zobrist_en_passant = [random.getrandbits(64) for _ in range(8)]
        
        # Reset seed để không ảnh hưởng random khác
        random.seed()

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
        Lưu trạng thái vào bảng băm với replacement policy thông minh.
        """
        # --- REPLACEMENT POLICY CẢI TIẾN ---
        # Khi đầy, xóa 25% entries cũ nhất thay vì xóa hết
        if len(self.table) >= self.size_limit:
            self._evict_old_entries()

        existing = self.table.get(hash_key)
        
        if existing is None:
            # Entry mới - lưu ngay
            self.table[hash_key] = {
                'value': value,
                'depth': depth,
                'flag': flag,
                'age': age, 
                'best_move': best_move,
                'static_eval': static_eval
            }
        else:
            # Entry đã tồn tại - quyết định có ghi đè không
            
            # Luôn cập nhật static_eval nếu có
            if static_eval is not None and existing['static_eval'] is None:
                existing['static_eval'] = static_eval
            
            # Điều kiện ghi đè search result:
            # 1. Depth mới >= depth cũ (thông tin chất lượng hơn)
            # 2. Entry cũ từ search trước (age cũ hơn 2 thế hệ)
            # 3. Flag EXACT luôn được ưu tiên hơn BOUND
            should_replace = False
            
            if depth > existing['depth']:
                should_replace = True
            elif depth == existing['depth']:
                # Cùng depth: EXACT > BOUND
                if flag == TranspositionTable.EXACT and existing['flag'] != TranspositionTable.EXACT:
                    should_replace = True
                # Hoặc age mới hơn
                elif age > existing['age'] + 1:
                    should_replace = True
            elif age > existing['age'] + 2:
                # Entry quá cũ, ghi đè dù depth thấp hơn
                should_replace = True
            
            if should_replace:
                existing['value'] = value
                existing['depth'] = depth
                existing['flag'] = flag
                existing['age'] = age
                if best_move is not None:
                    existing['best_move'] = best_move
    
    def _evict_old_entries(self):
        """Xóa 25% entries cũ nhất để giải phóng bộ nhớ"""
        if not self.table:
            return
        
        # Tìm entries cũ nhất dựa trên age và depth
        entries = [(k, v['age'], v['depth']) for k, v in self.table.items()]
        # Sắp xếp: age thấp (cũ) trước, depth thấp trước
        entries.sort(key=lambda x: (x[1], x[2]))
        
        # Xóa 25% entries đầu tiên
        num_to_remove = len(entries) // 4
        for i in range(num_to_remove):
            del self.table[entries[i][0]]
    
    def new_search(self):
        """Gọi khi bắt đầu search mới (iterative deepening loop)"""
        self.current_age += 1

    def lookup(self, hash_key):
        return self.table.get(hash_key)