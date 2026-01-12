# search/see.py
from core.constants import PieceType, Color

# Bảng giá trị cho SEE (đơn giản hóa)
SEE_VALUES = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 300,
    PieceType.BISHOP: 300,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 10000
}

def get_val(piece):
    if piece is None: return 0
    # Xử lý tương thích nếu piece.type là Enum hoặc int
    pt = piece.type.value if hasattr(piece.type, 'value') else piece.type
    lookup = [0, 100, 300, 300, 500, 900, 10000]
    return lookup[pt] if 0 <= pt < len(lookup) else 0

def get_smallest_attacker(board, square, side, removed_squares):
    """
    Tìm quân có giá trị thấp nhất của phe 'side' đang tấn công vào 'square'.
    Bỏ qua các ô nằm trong 'removed_squares' (các quân đã bị ăn giả định).
    """
    
    # 1. Kiểm tra Tốt (Pawns) - Ưu tiên số 1 vì giá trị thấp nhất
    # Lưu ý: Pawns tấn công ngược chiều di chuyển của nó.
    # Nếu phe tấn công là Trắng (đi lên, index tăng), nó phải ở dưới (index nhỏ hơn).

    # Nếu phe tấn công là Đen (đi xuống, index giảm), nó phải ở trên (index lớn hơn).
    
    pawn_offsets = [-7, -9] if side == Color.WHITE else [7, 9]
    for offset in pawn_offsets:
        target = square + offset # Vị trí tiềm năng của con tốt tấn công
        if 0 <= target < 64:
            # Kiểm tra biên: Tránh wrap-around (VD: H1 tấn công A2)
            if abs((square % 8) - (target % 8)) > 1:
                continue
                
            if target in removed_squares: continue
            
            p = board.squares[target]
            if p and p.color == side and p.type == PieceType.PAWN:
                return target, 100

    # 2. Kiểm tra Mã (Knights)
    knight_offsets = [-17, -15, -10, -6, 6, 10, 15, 17]
    for offset in knight_offsets:
        target = square + offset
        if 0 <= target < 64:
            if abs((square % 8) - (target % 8)) > 2: continue # Lệch quá 2 cột là sai
            if target in removed_squares: continue

            p = board.squares[target]
            if p and p.color == side and p.type == PieceType.KNIGHT:
                return target, 300

    # 3. Kiểm tra Tượng (Bishops) & Hậu (Queens) - Đường chéo
    diag_dirs = [-9, -7, 7, 9]
    attacker = _scan_sliders(board, square, side, removed_squares, diag_dirs, [PieceType.BISHOP, PieceType.QUEEN])
    if attacker: return attacker

    # 4. Kiểm tra Xe (Rooks) & Hậu (Queens) - Đường thẳng
    ortho_dirs = [-8, -1, 1, 8]
    attacker = _scan_sliders(board, square, side, removed_squares, ortho_dirs, [PieceType.ROOK, PieceType.QUEEN])
    if attacker: return attacker

    # 5. Kiểm tra Vua (King)
    king_offsets = [-9, -8, -7, -1, 1, 7, 8, 9]
    for offset in king_offsets:
        target = square + offset
        if 0 <= target < 64:
            if abs((square % 8) - (target % 8)) > 1: continue
            if target in removed_squares: continue
            
            p = board.squares[target]
            if p and p.color == side and p.type == PieceType.KING:
                return target, 10000

    return None, 0

def _scan_sliders(board, square, side, removed_squares, directions, allowed_types):
    """Hàm phụ trợ để quét các hướng (Ray-casting) cho quân trượt"""
    for d in directions:
        curr = square
        while True:
            curr += d
            if not (0 <= curr < 64): break
            # Kiểm tra wrap-around cột cho nước đi ngang/chéo
            if abs((curr % 8) - ((curr - d) % 8)) > 1: break 
            
            if curr in removed_squares:
                continue # Coi như ô trống, đi tiếp (X-ray qua quân đã bị ăn)
            
            p = board.squares[curr]
            if p:
                if p.color == side and p.type in allowed_types:
                    return curr, get_val(p)
                else:
                    break # Gặp quân chặn (của địch hoặc quân mình nhưng ko phải loại đang tìm)
    return None

def see_capture(board, move, threshold=0):
    from_sq = move.from_square
    to_sq = move.to_square
    
    # ---  XỬ LÝ EN PASSANT ---
    if move.is_en_passant:
        # Nếu là En Passant, quân bị ăn nằm khác hàng với to_sq
        # Ví dụ: Trắng đi e5->d6 (ăn tốt đen ở d5)
        # to_sq là d6, nhưng victim ở d5.
        if board.current_turn == Color.WHITE:
            victim_sq = to_sq - 8 # Victim ở dưới to_sq
        else:
            victim_sq = to_sq + 8 # Victim ở trên to_sq
        
        victim = board.squares[victim_sq]
        # Giá trị En Passant luôn là Tốt (100)
        value_victim = 100 
    else:
        # Nước ăn quân bình thường
        victim_sq = to_sq
        victim = board.squares[victim_sq]
        value_victim = get_val(victim)

    # --- Aggressor ---
    aggressor = board.squares[from_sq]
    value_aggressor = get_val(aggressor)
    
    # Gain list
    gain = [value_victim]
    
    # --- Logic loại bỏ quân (Removed Squares) ---
    # Khi En Passant, phải đánh dấu ô chứa Tốt bị ăn là đã loại bỏ
    removed_squares = {from_sq}
    if move.is_en_passant:
        removed_squares.add(victim_sq)
        
    curr_aggressor_val = value_aggressor
    side_to_move = Color.BLACK if board.current_turn == Color.WHITE else Color.WHITE
    
    
    # Lưu ý: Trong loop get_smallest_attacker, phải truyền to_sq (ô đích)
    # chứ không phải victim_sq, vì các quân khác sẽ tấn công vào ô đích (to_sq).
    while True:
        attacker_sq, attacker_val = get_smallest_attacker(board, to_sq, side_to_move, removed_squares)
        if attacker_sq is None: break
        
        removed_squares.add(attacker_sq)
        gain.append(curr_aggressor_val - gain[-1])
        curr_aggressor_val = attacker_val
        side_to_move = Color.BLACK if side_to_move == Color.WHITE else Color.WHITE
        
    while len(gain) > 1:
        gain[-2] = -max(-gain[-2], gain[-1])
        gain.pop()
        
    return gain[0] >= threshold