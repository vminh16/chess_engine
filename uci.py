import sys
import os
import traceback # Thư viện để in lỗi chi tiết

# Đảm bảo python nhìn thấy thư mục gốc
sys.path.append(os.getcwd())

try:
    from core.board import Board
    from search.negamax import find_best_move
    from evaluation.nn import NeuralNetwork
except ImportError as e:
    print(f"info string CRITICAL ERROR: Import failed - {e}")
    sys.stdout.flush()

# Load model an toàn
model = NeuralNetwork()
PATH_MODEL = os.path.join(os.getcwd(), "model", "nn_parameters.pth") # Đường dẫn tuyệt đối an toàn

if os.path.exists(PATH_MODEL):
    try:
        model.load_model(PATH_MODEL)
        # print("info string Model loaded successfully") 
    except Exception as e:
        print(f"info string Error loading model: {e}")
else:
    print(f"info string Warning: Model not found at {PATH_MODEL}")
sys.stdout.flush()

def main():
    board = Board()
    
    while True:
        try:
            command = sys.stdin.readline()
            if not command:
                break
            command = command.strip()
            
            if command == "uci":
                print("id name MyPythonEngine")
                print("id author You")
                print("uciok")
                sys.stdout.flush()

            elif command == "isready":
                print("readyok")
                sys.stdout.flush()

            elif command == "ucinewgame":
                board = Board()

            elif command.startswith("position"):
                params = command.split()
                idx = 0
                if "startpos" in params:
                    board = Board()
                    if "moves" in params:
                        idx = params.index("moves") + 1
                        for move_str in params[idx:]:
                            # Logic parse move đơn giản để tránh lỗi
                            # (Cần class MoveGenerator hoàn chỉnh để parse chuẩn hơn)
                            from core.move_generator import MoveGenerator
                            mg = MoveGenerator(board)
                            found = False
                            for m in mg.generate_legal_moves():
                                if str(m) == move_str:
                                    board.apply_move(m)
                                    found = True
                                    break
                            if not found:
                                print(f"info string Warning: Illegal move received {move_str}")

            elif command.startswith("go"):
                # BẮT LỖI TRONG QUÁ TRÌNH TÌM KIẾM
                try:
                    # In ra info để biết engine đang bắt đầu tính
                    print("info string Engine starting search...") 
                    sys.stdout.flush()
                    
                    # Gọi hàm tìm kiếm (Đảm bảo verbose=False để không in linh tinh)
                    # Depth 4 để test nhanh
                    best_move = find_best_move(board, depth=4, epsilon=0.7, model=model, verbose=False)
                    
                    if best_move:
                        # Fix format UCI
                        move_str = str(best_move).replace("ep", "")
                        if len(move_str) == 5: move_str = move_str.lower()
                        
                        print(f"bestmove {move_str}")
                    else:
                        print("bestmove 0000") # Null move nếu không tìm thấy
                        
                except Exception as e:
                    # NẾU CÓ LỖI, IN RA CHO CUTECHESS THẤY
                    error_msg = traceback.format_exc().replace('\n', ' | ')
                    print(f"info string SEARCH CRASHED: {error_msg}")
                    print("bestmove 0000") # Trả về move rác để không treo GUI
                
                sys.stdout.flush()

            elif command == "quit":
                break
                
        except Exception as global_e:
            # Bắt lỗi logic chung (parse lệnh...)
            err = str(global_e).replace('\n', ' ')
            print(f"info string GLOBAL ERROR: {err}")
            sys.stdout.flush()

if __name__ == "__main__":
    main()