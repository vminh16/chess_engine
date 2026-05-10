"""
CRANE-v0 Board Encoding
=======================
Encode schema: crane_v0_stm_spatial18_scalar5

Convention: Row 0 = STM back rank, Row 7 = OPP back rank
  - When STM = White: rank 0 (a1) → row 0 (no flip)
  - When STM = Black: rank 7 (a8) → row 0 (flip vertically)
  - Piece planes 0-5 = STM pieces, 6-11 = OPP pieces (auto-swap)
  - Castling 13-14 = STM, 15-16 = OPP (auto-swap)
"""

from typing import TYPE_CHECKING, Tuple, Dict, List, Optional

import numpy as np
from core.constants import PieceType, Color

if TYPE_CHECKING:
    from core.board import Board


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIECE_INDEX: Dict[PieceType, int] = {
    PieceType.PAWN: 0,
    PieceType.KNIGHT: 1,
    PieceType.BISHOP: 2,
    PieceType.ROOK: 3,
    PieceType.QUEEN: 4,
    PieceType.KING: 5,
}

_PIECE_SYMBOLS: Dict[int, str] = {
    0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",  # STM
    6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",  # OPP
}

_MATERIAL_VALUES: Dict[PieceType, int] = {
    PieceType.PAWN: 1,
    PieceType.KNIGHT: 3,
    PieceType.BISHOP: 3,
    PieceType.ROOK: 5,
    PieceType.QUEEN: 9,
    PieceType.KING: 0,
}

_PLANE_NAMES: List[str] = [
    "P_stm", "N_stm", "B_stm", "R_stm", "Q_stm", "K_stm",
    "P_opp", "N_opp", "B_opp", "R_opp", "Q_opp", "K_opp",
    "STM", "Castle_K_stm", "Castle_Q_stm", "Castle_K_opp", "Castle_Q_opp",
    "EnPassant",
]

_SCALAR_NAMES: List[str] = [
    "rule50", "phase", "mat_self", "mat_opp", "mat_delta",
]


# ---------------------------------------------------------------------------
# Core encode function
# ---------------------------------------------------------------------------

def encode_crane_v0(board: "Board") -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode board state per CRANE-v0 spec.

    Returns:
        X: (18, 8, 8) float32 spatial tensor
        s: (5,) float32 scalar vector

    Convention:
        Row 0 = STM back rank, Row 7 = OPP back rank.
        square = rank * 8 + file, where rank 0 = a1 (White back rank).
    """
    X = np.zeros((18, 8, 8), dtype=np.float32)

    current_color = board.current_turn
    opponent_color = Color.BLACK if current_color == Color.WHITE else Color.WHITE
    is_black_stm = (current_color == Color.BLACK)

    # Counters for scalar vector
    material_self = 0.0
    material_opp = 0.0
    count_knight = 0
    count_bishop = 0
    count_rook = 0
    count_queen = 0

    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            piece = board.squares[square]

            if piece is None:
                continue

            # --- Piece planes ---
            type_idx = _PIECE_INDEX.get(piece.type)
            if type_idx is not None:
                # Determine plane: 0-5 for STM, 6-11 for OPP
                if piece.color == current_color:
                    plane_idx = type_idx
                else:
                    plane_idx = type_idx + 6

                # Flip: Row 0 = STM back rank
                #   White=STM: rank 0 (a1) is already row 0 → no flip
                #   Black=STM: rank 7 (a8) should become row 0 → flip
                if is_black_stm:
                    r_idx = 7 - rank
                else:
                    r_idx = rank
                f_idx = file

                X[plane_idx, r_idx, f_idx] = 1.0

            # --- Material counting ---
            value = _MATERIAL_VALUES.get(piece.type, 0)
            if value > 0:
                if piece.color == current_color:
                    material_self += value
                else:
                    material_opp += value

            # --- Phase counting (both sides) ---
            if piece.type == PieceType.KNIGHT:
                count_knight += 1
            elif piece.type == PieceType.BISHOP:
                count_bishop += 1
            elif piece.type == PieceType.ROOK:
                count_rook += 1
            elif piece.type == PieceType.QUEEN:
                count_queen += 1

    # --- Constant planes ---
    def fill_plane(idx: int, value: float) -> None:
        X[idx].fill(value)

    # Plane 12: Side-to-move (1.0 = White, 0.0 = Black)
    fill_plane(12, 1.0 if current_color == Color.WHITE else 0.0)

    # Planes 13-16: Castling rights (STM first, then OPP)
    rights = board.castling_rights
    fill_plane(13, 1.0 if rights[current_color]["kingside"] else 0.0)
    fill_plane(14, 1.0 if rights[current_color]["queenside"] else 0.0)
    fill_plane(15, 1.0 if rights[opponent_color]["kingside"] else 0.0)
    fill_plane(16, 1.0 if rights[opponent_color]["queenside"] else 0.0)

    # Plane 17: En passant target
    if board.en_passant_target is not None:
        ep_rank, ep_file = divmod(board.en_passant_target, 8)
        if is_black_stm:
            ep_rank = 7 - ep_rank
        X[17, ep_rank, ep_file] = 1.0

    # --- Scalar vector ---
    rule50 = min(1.0, board.halfmove_clock / 100.0)
    phase_raw = count_knight + count_bishop + 2 * count_rook + 4 * count_queen
    phase = min(1.0, phase_raw / 20.0)
    mat_self = material_self / 39.0
    mat_opp = material_opp / 39.0
    mat_delta = float(np.clip(mat_self - mat_opp, -1.0, 1.0))

    s = np.array([rule50, phase, mat_self, mat_opp, mat_delta], dtype=np.float32)

    return X, s


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_spatial(X: np.ndarray, title: str = "Spatial Tensor") -> None:
    """
    Print ASCII visualization of the 18x8x8 spatial tensor.

    Shows:
    1. Combined piece map (all pieces on one board)
    2. Individual non-empty planes
    3. Constant planes (STM, castling) value
    """
    assert X.shape == (18, 8, 8), f"Expected (18,8,8), got {X.shape}"

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

    # --- Combined piece map ---
    print(f"\n  Combined board (Row 0 = STM back rank):")
    print(f"       a  b  c  d  e  f  g  h")
    print(f"      {'─'*27}")

    board_chars = [["." for _ in range(8)] for _ in range(8)]
    for plane in range(12):
        for r in range(8):
            for c in range(8):
                if X[plane, r, c] > 0.5:
                    board_chars[r][c] = _PIECE_SYMBOLS[plane]

    for r in range(7, -1, -1):
        row_label = r + 1  # Display as 1-indexed from bottom
        # But we print top-to-bottom: row 7 first (OPP side)
        row_str = "  ".join(board_chars[r])
        side = "OPP" if r >= 6 else ("STM" if r <= 1 else "   ")
        print(f"  {row_label}  {row_str}   {side}")

    print(f"      {'─'*27}")
    print(f"       a  b  c  d  e  f  g  h")
    print(f"       ^STM back rank = row 0 (bottom)")

    # --- En passant ---
    ep_positions = []
    for r in range(8):
        for c in range(8):
            if X[17, r, c] > 0.5:
                file_ch = chr(ord('a') + c)
                ep_positions.append(f"{file_ch}{r+1}")

    # --- Constant planes info ---
    stm_val = X[12, 0, 0]
    stm_str = "White" if stm_val > 0.5 else "Black"
    castle_str = ""
    for i, name in zip([13, 14, 15, 16], ["K_stm", "Q_stm", "K_opp", "Q_opp"]):
        if X[i, 0, 0] > 0.5:
            castle_str += f" {name}"

    print(f"\n  Side to move: {stm_str}")
    print(f"  Castling:    {castle_str if castle_str else '  (none)'}")
    print(f"  En passant:  {', '.join(ep_positions) if ep_positions else '(none)'}")

    # --- Individual non-empty piece planes ---
    print(f"\n  Non-empty piece planes:")
    for plane in range(12):
        count = int(X[plane].sum())
        if count > 0:
            positions = []
            for r in range(8):
                for c in range(8):
                    if X[plane, r, c] > 0.5:
                        positions.append(f"{chr(ord('a')+c)}{r+1}")
            print(f"    Plane {plane:2d} ({_PLANE_NAMES[plane]:12s}): {count} pieces  {', '.join(positions)}")


def visualize_scalar(s: np.ndarray, title: str = "Scalar Vector") -> None:
    """Print scalar vector values with labels."""
    assert s.shape == (5,), f"Expected (5,), got {s.shape}"

    print(f"\n  {title}:")
    for i, (name, val) in enumerate(zip(_SCALAR_NAMES, s)):
        print(f"    s[{i}] {name:12s} = {val:+.4f}")


def visualize_encoding(X: np.ndarray, s: np.ndarray, title: str = "CRANE-v0 Encoding") -> None:
    """Full visualization of both spatial tensor and scalar vector."""
    visualize_spatial(X, title=title)
    visualize_scalar(s)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class EncodeVerificationError(Exception):
    """Raised when encode verification fails."""
    pass


def verify_encoding(X: np.ndarray, s: np.ndarray, board: "Board",
                    strict: bool = True) -> List[str]:
    """
    Verify encoding against CRANE-v0 spec.

    Returns list of error messages. Empty list = all checks passed.

    Checks:
    1. Shape and dtype
    2. Piece counts match board
    3. No duplicate placement (2 pieces on same square in same plane)
    4. STM/OPP King count = 1 each
    5. En passant at correct position
    6. Castling rights match board
    7. Scalar values match computed values
    8. Binary planes are truly binary (0.0 or 1.0)
    9. Constant planes are constant across spatial dims
    10. No overlap between piece planes (at most 1 piece per square)
    """
    errors: List[str] = []

    current_color = board.current_turn
    opponent_color = Color.BLACK if current_color == Color.WHITE else Color.WHITE
    is_black_stm = (current_color == Color.BLACK)

    # --- 1. Shape & dtype ---
    if X.shape != (18, 8, 8):
        errors.append(f"X shape = {X.shape}, expected (18,8,8)")
    if X.dtype != np.float32:
        errors.append(f"X dtype = {X.dtype}, expected float32")
    if s.shape != (5,):
        errors.append(f"s shape = {s.shape}, expected (5,)")
    if s.dtype != np.float32:
        errors.append(f"s dtype = {s.dtype}, expected float32")

    # --- 2. Piece counts ---
    # Count pieces from board
    board_stm_counts = {}  # piece_type -> count
    board_opp_counts = {}
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            piece = board.squares[square]
            if piece is not None and piece.type in _PIECE_INDEX:
                if piece.color == current_color:
                    board_stm_counts[piece.type] = board_stm_counts.get(piece.type, 0) + 1
                else:
                    board_opp_counts[piece.type] = board_opp_counts.get(piece.type, 0) + 1

    for ptype, expected_count in board_stm_counts.items():
        plane = _PIECE_INDEX[ptype]
        actual_count = int(X[plane].sum())
        if actual_count != expected_count:
            errors.append(f"STM {ptype.name}: plane {plane} has {actual_count} pieces, expected {expected_count}")

    for ptype, expected_count in board_opp_counts.items():
        plane = _PIECE_INDEX[ptype] + 6
        actual_count = int(X[plane].sum())
        if actual_count != expected_count:
            errors.append(f"OPP {ptype.name}: plane {plane} has {actual_count} pieces, expected {expected_count}")

    # --- 3. No duplicate placement ---
    for plane in range(12):
        vals = X[plane]
        if np.any((vals > 0.5) & (vals < 1.5) & (vals != 1.0)):
            errors.append(f"Plane {plane}: non-binary values found")

    # --- 4. Piece count sanity ---
    # STM King must exist (exactly 1)
    stm_king_plane = 5  # King is plane 5 for STM
    king_count = int(X[stm_king_plane].sum())
    if king_count != 1:
        if king_count == 0:
            errors.append("STM King not found on board")
        else:
            errors.append(f"Multiple STM Kings found: {king_count}")

    # OPP King must exist (exactly 1)
    opp_king_plane = 11
    opp_king_count = int(X[opp_king_plane].sum())
    if opp_king_count != 1:
        if opp_king_count == 0:
            errors.append("OPP King not found on board")
        else:
            errors.append(f"Multiple OPP Kings found: {opp_king_count}")

    # --- 5. En passant ---
    if board.en_passant_target is not None:
        ep_rank_raw, ep_file_raw = divmod(board.en_passant_target, 8)
        if is_black_stm:
            expected_ep_row = 7 - ep_rank_raw
        else:
            expected_ep_row = ep_rank_raw
        expected_ep_col = ep_file_raw

        ep_count = int(X[17].sum())
        if ep_count != 1:
            errors.append(f"En passant plane has {ep_count} active squares, expected 1")

        ep_positions = np.argwhere(X[17] > 0.5)
        if len(ep_positions) == 1:
            actual_r, actual_c = ep_positions[0]
            if actual_r != expected_ep_row or actual_c != expected_ep_col:
                errors.append(
                    f"En passant at ({actual_r},{actual_c}), "
                    f"expected ({expected_ep_row},{expected_ep_col})"
                )
    else:
        ep_count = int(X[17].sum())
        if ep_count != 0:
            errors.append(f"En passant plane has {ep_count} active squares, expected 0 (no EP)")

    # --- 6. Castling rights ---
    rights = board.castling_rights
    expected_castle = {
        13: rights[current_color]["kingside"],
        14: rights[current_color]["queenside"],
        15: rights[opponent_color]["kingside"],
        16: rights[opponent_color]["queenside"],
    }
    for plane, expected in expected_castle.items():
        actual = X[plane, 0, 0] > 0.5
        if actual != expected:
            errors.append(f"Plane {plane} ({_PLANE_NAMES[plane]}): {actual}, expected {expected}")

    # --- 7. Scalar values ---
    # rule50
    expected_rule50 = min(1.0, board.halfmove_clock / 100.0)
    if abs(s[0] - expected_rule50) > 1e-5:
        errors.append(f"s[0] rule50 = {s[0]}, expected {expected_rule50}")

    # phase — recount from board (counters are local to encode_crane_v0)
    recount_n = recount_b = recount_r = recount_q = 0
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            piece = board.squares[square]
            if piece is None:
                continue
            if piece.type == PieceType.KNIGHT:
                recount_n += 1
            elif piece.type == PieceType.BISHOP:
                recount_b += 1
            elif piece.type == PieceType.ROOK:
                recount_r += 1
            elif piece.type == PieceType.QUEEN:
                recount_q += 1
    phase_raw = recount_n + recount_b + 2 * recount_r + 4 * recount_q
    expected_phase = min(1.0, phase_raw / 20.0)
    if abs(s[1] - expected_phase) > 1e-5:
        errors.append(f"s[1] phase = {s[1]}, expected {expected_phase}")

    # material
    expected_mat_self = 0.0
    expected_mat_opp = 0.0
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file
            piece = board.squares[square]
            if piece is None:
                continue
            val = _MATERIAL_VALUES.get(piece.type, 0)
            if val > 0:
                if piece.color == current_color:
                    expected_mat_self += val
                else:
                    expected_mat_opp += val
    expected_mat_self /= 39.0
    expected_mat_opp /= 39.0
    expected_delta = float(np.clip(expected_mat_self - expected_mat_opp, -1.0, 1.0))

    if abs(s[2] - expected_mat_self) > 1e-5:
        errors.append(f"s[2] mat_self = {s[2]}, expected {expected_mat_self}")
    if abs(s[3] - expected_mat_opp) > 1e-5:
        errors.append(f"s[3] mat_opp = {s[3]}, expected {expected_mat_opp}")
    if abs(s[4] - expected_delta) > 1e-5:
        errors.append(f"s[4] mat_delta = {s[4]}, expected {expected_delta}")

    # --- 8. Binary planes (0-11, 17) are truly binary ---
    for plane in list(range(12)) + [17]:
        vals = X[plane]
        nonzero = vals[vals != 0.0]
        if len(nonzero) > 0 and not np.allclose(nonzero, 1.0, atol=1e-6):
            errors.append(f"Plane {plane}: non-binary values {nonzero[nonzero != 1.0]}")

    # --- 9. Constant planes (12-16) are spatially constant ---
    for plane in range(12, 17):
        if not np.allclose(X[plane], X[plane, 0, 0]):
            errors.append(f"Plane {plane} ({_PLANE_NAMES[plane]}): not constant across spatial dims")

    # --- 10. No overlap between piece planes ---
    for r in range(8):
        for c in range(8):
            piece_count = sum(1 for p in range(12) if X[p, r, c] > 0.5)
            if piece_count > 1:
                errors.append(f"Square ({r},{c}): {piece_count} pieces overlap")

    return errors


def assert_encoding_correct(X: np.ndarray, s: np.ndarray, board: "Board") -> None:
    """Raise EncodeVerificationError if encoding is incorrect."""
    errors = verify_encoding(X, s, board)
    if errors:
        msg = f"Encoding verification failed with {len(errors)} error(s):\n"
        for i, e in enumerate(errors, 1):
            msg += f"  {i}. {e}\n"
        raise EncodeVerificationError(msg)


# ---------------------------------------------------------------------------
# Perspective consistency test
# ---------------------------------------------------------------------------

def verify_perspective_consistency(board: "Board") -> List[str]:
    """
    Verify that encoding produces a valid STM-relative representation.

    Checks:
    1. STM King exists (exactly 1)
    2. OPP King exists (exactly 1)
    3. No overlap between piece planes (at most 1 piece per square)
    4. Scalar delta sign matches spatial material difference

    Note: We cannot check that the STM King is on rows 0-1 because
    the King may have moved anywhere on the board in a real game.
    The "Row 0 = STM back rank" convention defines orientation, not
    a constraint on piece positions.
    """
    X, s = encode_crane_v0(board)
    errors = []

    # STM King must exist
    stm_king_plane = 5
    king_positions = np.argwhere(X[stm_king_plane] > 0.5)
    if len(king_positions) != 1:
        errors.append(f"STM King count: {len(king_positions)}, expected 1")

    # OPP King must exist
    opp_king_plane = 11
    opp_king_positions = np.argwhere(X[opp_king_plane] > 0.5)
    if len(opp_king_positions) != 1:
        errors.append(f"OPP King count: {len(opp_king_positions)}, expected 1")

    # No overlap between piece planes
    for r in range(8):
        for c in range(8):
            piece_count = sum(1 for p in range(12) if X[p, r, c] > 0.5)
            if piece_count > 1:
                errors.append(f"Square ({r},{c}): {piece_count} pieces overlap")

    # Material delta sign consistency with spatial encoding
    mat_self_spatial = sum(int(X[p].sum()) * _MATERIAL_VALUES.get(
        [_k for _k, _v in _PIECE_INDEX.items() if _v == p][0], 0
    ) for p in range(6))
    mat_opp_spatial = sum(int(X[p].sum()) * _MATERIAL_VALUES.get(
        [_k for _k, _v in _PIECE_INDEX.items() if _v == p - 6][0], 0
    ) for p in range(6, 12))
    if mat_self_spatial > mat_opp_spatial and s[4] < 0:
        errors.append(f"Material delta sign mismatch: spatial shows self>opp but s[4]={s[4]:.4f}")
    elif mat_self_spatial < mat_opp_spatial and s[4] > 0:
        errors.append(f"Material delta sign mismatch: spatial shows self<opp but s[4]={s[4]:.4f}")

    return errors
