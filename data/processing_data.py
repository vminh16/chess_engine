import zstandard as zst
import chess.pgn
import chess
import chess.engine
import random
import io
import numpy as np
import sys
sys.path.append('.')
from representation.encode import encode_board
from core.board import Board
from core.constants import Color, PieceType, Piece

def chess_piece_to_piece(piece: chess.Piece) -> 'Piece':
    """Convert chess.Piece to Piece object."""
    color = Color.WHITE if piece.color == chess.WHITE else Color.BLACK
    if piece.piece_type == chess.PAWN:
        ptype = PieceType.PAWN
    elif piece.piece_type == chess.KNIGHT:
        ptype = PieceType.KNIGHT
    elif piece.piece_type == chess.BISHOP:
        ptype = PieceType.BISHOP
    elif piece.piece_type == chess.ROOK:
        ptype = PieceType.ROOK
    elif piece.piece_type == chess.QUEEN:
        ptype = PieceType.QUEEN
    elif piece.piece_type == chess.KING:
        ptype = PieceType.KING
    else:
        raise ValueError(f"Unknown piece type: {piece.piece_type}")
    return Piece(color, ptype)

def convert_chess_board_to_board(chess_board: chess.Board) -> Board:
    """Convert chess.Board to Board object."""
    board = Board()
    board.squares = [None] * 64
    for square in chess.SQUARES:
        piece = chess_board.piece_at(square)
        if piece is not None:
            board.squares[square] = chess_piece_to_piece(piece)
    board.current_turn = Color.WHITE if chess_board.turn == chess.WHITE else Color.BLACK
    board.castling_rights = {
        Color.WHITE: {
            'kingside': chess_board.has_kingside_castling_rights(chess.WHITE),
            'queenside': chess_board.has_queenside_castling_rights(chess.WHITE)
        },
        Color.BLACK: {
            'kingside': chess_board.has_kingside_castling_rights(chess.BLACK),
            'queenside': chess_board.has_queenside_castling_rights(chess.BLACK)
        }
    }
    board.en_passant_target = chess_board.ep_square
    return board

def decompress_and_parse_pgn(file_path, stockfish_path, target_samples=10000, batch_size=1000):
    """
    Giải nén file .zst và parse từng game PGN.
    Cho mỗi game, chọn 2-5 positions ngẫu nhiên, eval với Stockfish, encode và lưu dataset.
    Dừng khi đạt target_samples.
    Ghi theo batch.
    """
    X = []
    y = []
    fens = []
    batch_num = 0
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        # Set Stockfish options
        engine.configure({"Threads": 4, "Hash": 128})
        
        with open(file_path, 'rb') as f:
            dctx = zst.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    process_game(game, engine, X, y, fens)
                    
                    # Check if enough samples
                    if len(X) >= target_samples:
                        break
                    
                    # Ghi batch nếu đủ
                    if len(X) >= (batch_num + 1) * batch_size:
                        save_batch(X, y, fens, batch_num, batch_size)
                        batch_num += 1
        
        # Ghi batch cuối
        if X:
            save_batch(X, y, fens, batch_num, batch_size)
        
        print(f"Total samples: {len(X)}")
    # Engine đóng tự động

def save_batch(X, y, fens, batch_num, batch_size):
    """Ghi batch dataset."""
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, len(X))
    batch_X = np.array(X[start_idx:end_idx])
    batch_y = np.array(y[start_idx:end_idx])
    batch_fens = fens[start_idx:end_idx]
    np.savez(f'dataset_batch_{batch_num}.npz', X=batch_X, y=batch_y, fens=batch_fens)
    print(f"Saved batch {batch_num} with {len(batch_X)} samples")

def process_game(game, engine, X, y, fens):
    """
    Xử lý một game: chọn 2-5 ply ngẫu nhiên cách nhau >=2, eval với Stockfish, encode và lưu.
    """
    board = game.board()
    moves = list(game.mainline_moves())
    n = len(moves)
    if n < 7:  # Ít nhất 4 ply + 3 để check mate-in
        return

    start_ply = 12
    end_ply = min(n - 3, 80)
    if end_ply < start_ply:
        return

    # Chọn số ply: 2-5
    k = random.randint(4, 9)
    # Chọn k ply distinct, cách nhau >=2
    available = list(range(start_ply, end_ply + 1))
    if len(available) < k:
        k = len(available)
    
    selected_plies = []
    while len(selected_plies) < k and available:
        ply = random.choice(available)
        if all(abs(ply - s) >= 2 for s in selected_plies):
            selected_plies.append(ply)
        available.remove(ply)
    
    selected_plies.sort()

    for ply in selected_plies:
        # Reset board
        temp_board = game.board()
        for move in moves[:ply]:
            temp_board.push(move)
        
        # Gọi Stockfish để eval position tại depth = 8
        result = engine.analyse(temp_board, chess.engine.Limit(depth=12))
        score = result['score']
        
        # Bỏ qua positions mate
        if score.is_mate():
            continue
        
        # Label với tanh
        cp = score.pov(chess.WHITE).score()
        eval_score = np.tanh(cp / 600.0)

        # Encode board dùng code đã có
        board_obj = convert_chess_board_to_board(temp_board)
        encoded = encode_board(board_obj)

        # Lưu dataset
        X.append(encoded)
        y.append(eval_score)
        fens.append(temp_board.fen())


def main():
    file_path = "data/lichess_db_standard_rated_2017-08.pgn.zst"
    stockfish_path = r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # Adjust path if needed
    decompress_and_parse_pgn(file_path, stockfish_path, target_samples=1000000, batch_size=100000)

if __name__ == "__main__":
    main()