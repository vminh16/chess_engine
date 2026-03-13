# app.py
from flask import Flask, render_template, request, jsonify
import sys
import os

# Đảm bảo python nhìn thấy các module của bạn
sys.path.append(os.getcwd())

from core.board import Board
from evaluation.nn import NeuralNetwork
from search.negamax import find_best_move
from model.architecture.model import PhantomChessNet

app = Flask(__name__, template_folder='templates')

# --- KHỞI TẠO MODEL MỘT LẦN DUY NHẤT (Để tiết kiệm thời gian) ---
print("Loading Neural Network...")
model = PhantomChessNet()
# Đảm bảo đường dẫn này đúng với máy bạn
PATH_MODEL = "model/param_model/PhantomChessNet.pth" 
if os.path.exists(PATH_MODEL):
    model.load_model(PATH_MODEL)
else:
    print(f"Warning: Model not found at {PATH_MODEL}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_move', methods=['POST'])
def get_move():
    try:
        data = request.json
        fen = data.get('fen')
        
        # 1. Tái tạo bàn cờ từ FEN nhận được từ browser
        board = Board(fen)
        
        # 2. Gọi Engine của bạn
        # Lưu ý: depth có thể điều chỉnh, depth=4 là hợp lý cho web nhanh
        best_move = find_best_move(board, depth=4, epsilon=0.0, model=model)
        
        if best_move:
            # Chuyển đổi move object thành string (ví dụ: "e2e4")
            # Giả định class Move của bạn có __str__ hoặc bạn tự convert
            move_str = str(best_move) 
            
            return jsonify({
                'success': True,
                'move': move_str,
                'from': move_str[:2],
                'to': move_str[2:4],
                'promotion': move_str[4] if len(move_str) > 4 else 'q' # Mặc định phong Hậu
            })
        else:
            return jsonify({'success': False, 'message': 'No move found (Mate/Stalemate?)'})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)