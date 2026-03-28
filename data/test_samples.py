import chess.pgn
import chess.engine
import numpy as np
import io
import sys
sys.path.append('..')
import chess.pgn
import chess.engine
import numpy as np
import io

# Copy process_game logic without encode
def process_game_simple(game, engine, X, y, fens):
    # Simplified version for testing
    board = game.board()
    moves = list(game.mainline_moves())
    n = len(moves)
    if n < 7:
        return

    start_ply = 4
    end_ply = min(n - 3, 80)
    if end_ply < start_ply:
        return

    k = 2  # Fixed for test
    available = list(range(start_ply, end_ply + 1))
    selected_plies = [10, 15]  # Fixed for test

    for ply in selected_plies:
        temp_board = game.board()
        for move in moves[:ply]:
            temp_board.push(move)
        
        result = engine.analyse(temp_board, chess.engine.Limit(depth=8))
        score = result['score']
        
        if score.is_mate() and abs(score.mate()) <= 3:
            continue
        
        if score.is_mate():
            continue
        
        cp = score.relative.score()
        eval_score = np.tanh(cp / 600.0)

        X.append(None)  # Dummy
        y.append(eval_score)
        fens.append(temp_board.fen())

def test_samples():
    # Tạo game mẫu
    pgn_text = '''
[Event "Test Game"]
[Site "?"]
[Date "2023.01.01"]
[Round "?"]
[White "White"]
[Black "Black"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 15. a4 c5 16. d5 c4 17. Bg5 h6 18. Be3 Nc5 19. Qd2 Kg7 20. axb5 axb5 21. Ra5 Qb6 22. Rea1 Rxa5 23. Rxa5 Ra8 24. Rxa8 Bxa8 25. Qa5 Qxa5 26. Bxa5 Nd3 27. Bxd3 cxd3 28. Bc7 Bc8 29. Ne1 d2 30. Nxd3 Nxd5 31. exd5 Bxd5 32. Nf3 f6 33. Kf1 Kf7 34. Ke2 Ke6 35. Kd1 Kd5 36. Kxd2 Kc4 37. Ke3 Kb3 38. Nd4+ Kxb2 39. Nxb5 Kxc3 40. Nd6 Kd4 41. Nf7 Ke5 42. Ng5 hxg5 43. Bxf6+ Kf5 44. Bxg5 Ke6 45. h4 Kd7 46. h5 gxh5 47. Bf6 Ke8 48. Bg7 Bf7 49. Bf8 Kd7 50. Bh6 Ke8 51. Bg7 *
'''
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    
    # Mở engine
    stockfish_path = r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"Threads": 4, "Hash": 128})
        
        X = []
        y = []
        fens = []
        process_game_simple(game, engine, X, y, fens)
        
        # In 2-3 samples
        for i in range(min(3, len(X))):
            print(f"Sample {i+1}:")
            print(f"FEN: {fens[i]}")
            cp = np.arctanh(y[i]) * 600  # Tính ngược cp từ tanh
            print(f"cp: {cp}")
            print(f"tanh(cp / 600): {y[i]}")
            print()

if __name__ == "__main__":
    test_samples()