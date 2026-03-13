"""
Benchmark script for chess engine.
Measures NPS (Nodes Per Second) and Time per Inference.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import torch
import numpy as np
import onnxruntime as ort
from core.board import Board
from core.constants import Color
from model.architecture.model import PhantomChessNet
from representation.encode import encode_board
from search.negamax import negamax, quiescence_search, hybrid_evaluate, TT, move_ordering_capture
from search.transition_table import TranspositionTable
from evaluation.static_eval import material_evaluate
from core.move_generator import MoveGenerator
from core.rules import is_in_check
from search.ordering import sort_moves_priority
from search.see import see_capture
from search.utils import is_forced_move, calculate_lmr_reduction
import json

# Model path
PATH_MODEL = "model/param_model/PhantomChessNet_int8.onnx"

# 10 chess positions: Opening (3), Middle Game (4), Endgame (3)
TEST_POSITIONS = [
    # Opening positions
    {
        "name": "Starting Position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "category": "Opening"
    },
    {
        "name": "Italian Game",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "category": "Opening"
    },
    {
        "name": "Sicilian Defense",
        "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "category": "Opening"
    },
    
    # Middle game positions
    {
        "name": "Complex Middle Game 1",
        "fen": "r2qkb1r/pp2nppp/2n1p3/3pP3/3P1P2/2N2N2/PPP3PP/R1BQKB1R w KQkq - 0 8",
        "category": "Middle Game"
    },
    {
        "name": "Complex Middle Game 2",
        "fen": "r1bqkb1r/pp3ppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
        "category": "Middle Game"
    },
    {
        "name": "Tactical Position",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
        "category": "Middle Game"
    },
    {
        "name": "Attacking Position",
        "fen": "r2qkb1r/pp2nppp/2n1p3/3pP3/3P1P2/2N2N2/PPP3PP/R1BQKB1R w KQkq - 0 8",
        "category": "Middle Game"
    },
    
    # Endgame positions
    {
        "name": "King and Pawn Endgame",
        "fen": "8/8/8/4k3/4P3/8/4K3/8 w - - 0 1",
        "category": "Endgame"
    },
    {
        "name": "Rook Endgame",
        "fen": "8/8/8/8/2k5/8/2K5/4R3 w - - 0 1",
        "category": "Endgame"
    },
    {
        "name": "Queen Endgame",
        "fen": "8/8/8/3k4/3Q4/8/3K4/8 w - - 0 1",
        "category": "Endgame"
    }
]

# Global node counter
node_count = 0

def reset_node_count():
    """Reset the global node counter"""
    global node_count
    node_count = 0

def get_node_count():
    """Get the current node count"""
    return node_count

def increment_node_count():
    """Increment the node counter"""
    global node_count
    node_count += 1

def benchmark_negamax_with_counting(board: Board, depth: int, alpha: float, beta: float,
                                     epsilon: float, model, ply: int) -> float:
    """
    Modified negamax that counts nodes.
    Uses the actual negamax logic but wraps it to count nodes.
    """
    increment_node_count()
    
    MATE_SCORE = 10000
    MATE_THRESHOLD = 9000
    
    # Check transposition table
    hash_key = TT.compute_hash(board)
    entry = TT.lookup(hash_key)
    
    static_eval = entry['static_eval'] if entry is not None else None
    if static_eval is None:
        static_eval = material_evaluate(board)
        if board.current_turn == Color.BLACK:
            static_eval = -static_eval
    
    best_move_tt = entry['best_move'] if entry is not None else None
    if entry is not None and entry['depth'] >= depth:
        val = entry['value']
        if val > MATE_THRESHOLD:
            val -= ply
        elif val < -MATE_THRESHOLD:
            val += ply
        
        if entry['flag'] == TranspositionTable.EXACT:
            return val
        elif entry['flag'] == TranspositionTable.LOWERBOUND:
            alpha = max(alpha, val)
        elif entry['flag'] == TranspositionTable.UPPERBOUND:
            beta = min(beta, val)
        
        if alpha >= beta:
            return val
    
    if depth <= 0:
        return benchmark_quiescence_with_counting(board, alpha, beta, epsilon, model, ply)
    
    move_gen = MoveGenerator(board)
    moves = move_gen.generate_legal_moves()
    moves = sort_moves_priority(moves, board, tt_move=best_move_tt)
    
    if not moves:
        if is_in_check(board, board.current_turn):
            return -(MATE_SCORE - ply)
        return 0.0
    
    max_score = -float('inf')
    alpha_orig = alpha
    in_check = is_in_check(board, board.current_turn)
    
    for i, move in enumerate(moves):
        is_capture = move.is_capture(board)
        gives_check = False
        
        # Simplified LMR for benchmarking
        R = 0
        if depth >= 3 and i >= 3 and not in_check and not move.is_promotion():
            board.apply_move(move)
            gives_check = is_in_check(board, board.current_turn)
            board.undo_move()
            
            if not gives_check:
                see_result = -1
                if is_capture:
                    if see_capture(board, move, threshold=0):
                        see_result = 1
                R = calculate_lmr_reduction(i, depth, move, board, is_capture,
                                            see_result, in_check, static_eval, alpha, beta)
        
        board.apply_move(move)
        score = -float('inf')
        
        if i == 0:
            score = -benchmark_negamax_with_counting(board, depth - 1, -beta, -alpha, epsilon, model, ply + 1)
        else:
            score = -benchmark_negamax_with_counting(board, depth - 1 - R, -alpha - 1, -alpha, epsilon, model, ply + 1)
            if score > alpha and R > 0:
                score = -benchmark_negamax_with_counting(board, depth - 1, -alpha - 1, -alpha, epsilon, model, ply + 1)
            if score > alpha and score < beta:
                score = -benchmark_negamax_with_counting(board, depth - 1, -beta, -alpha, epsilon, model, ply + 1)
        
        board.undo_move()
        
        if score > max_score:
            max_score = score
        if max_score > alpha:
            alpha = max_score
        if alpha >= beta:
            break
    
    # Store in transposition table
    if max_score <= alpha_orig:
        flag = TranspositionTable.UPPERBOUND
    elif max_score >= beta:
        flag = TranspositionTable.LOWERBOUND
    else:
        flag = TranspositionTable.EXACT
    
    tt_val = max_score
    if max_score > MATE_THRESHOLD:
        tt_val = max_score + ply
    elif max_score < -MATE_THRESHOLD:
        tt_val = max_score - ply
    
    TT.store(hash_key, depth, tt_val, flag, board.fullmove_number)
    return max_score

def benchmark_quiescence_with_counting(board: Board, alpha: float, beta: float,
                                       epsilon: float, model, ply: int) -> float:
    """Quiescence search with node counting"""
    increment_node_count()
    
    MAX_QS_PLY = 100
    MATE_THRESHOLD = 9000
    
    if ply > MAX_QS_PLY:
        eval_score = material_evaluate(board)
        if board.current_turn == Color.BLACK:
            eval_score = -eval_score
        return eval_score
    
    hash_key = TT.compute_hash(board)
    entry = TT.lookup(hash_key)
    
    stand_pat = hybrid_evaluate(board, material_evaluate(board), epsilon, model, hash_key, entry)
    if stand_pat >= beta:
        return stand_pat
    if alpha < stand_pat:
        alpha = stand_pat
    
    move_gen = MoveGenerator(board)
    capture_moves = move_gen.generate_pseudo_captures()
    move_ordered = move_ordering_capture(capture_moves, board, tt=entry['best_move'] if entry else None)
    
    for move in move_ordered[:10]:  # Limit captures for benchmarking
        board.apply_move(move)
        score = -benchmark_quiescence_with_counting(board, -beta, -alpha, epsilon, model, ply + 1)
        board.undo_move()
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    
    return alpha

def measure_nps(board: Board, depth: int, epsilon: float = 0.5, model=None):
    """
    Measure Nodes Per Second (NPS) for a given position.
    
    Args:
        board: Chess board position
        depth: Search depth
        epsilon: Weight for neural network evaluation
        model: Neural network model
    
    Returns:
        tuple: (nps, total_nodes, elapsed_time)
    """
    global TT
    TT = TranspositionTable()  # Reset TT for each test
    reset_node_count()
    
    alpha = -float('inf')
    beta = float('inf')
    
    start_time = time.time()
    benchmark_negamax_with_counting(board, depth, alpha, beta, epsilon, model, ply=0)
    elapsed_time = time.time() - start_time
    
    total_nodes = get_node_count()
    nps = total_nodes / elapsed_time if elapsed_time > 0 else 0
    
    return nps, total_nodes, elapsed_time

def measure_inference_time(onnx_wrapper, batch_sizes=[1]):
    """
    Measure time per inference for different batch sizes using ONNX model.
    
    NOTE: Many ONNX models are exported with batch_size=1 only, especially if they
    contain operations that don't support batching (e.g., hardcoded dimensions in heads).
    This function will test each batch size and skip unsupported ones.
    
    Args:
        onnx_wrapper: ONNXModelWrapper instance
        batch_sizes: List of batch sizes to test
    
    Returns:
        dict: Dictionary mapping batch_size to time_per_inference (seconds)
    """
    # Create a dummy board position
    board = Board()
    encoded = encode_board(board)  # Shape: (18, 8, 8) - (C, H, W) from encode.py
    
    # Get expected input shape from ONNX model
    input_shape = onnx_wrapper.session.get_inputs()[0].shape
    print(f"ONNX model expects input shape: {input_shape}")
    print(f"Encoded board shape: {encoded.shape}")
    
    # Verify encoded shape is (18, 8, 8)
    if encoded.shape != (18, 8, 8):
        raise ValueError(
            f"Unexpected encoded shape: {encoded.shape}, expected (18, 8, 8). "
            f"Check encode_board() function."
        )
    
    results = {}
    unsupported_batches = []
    
    for batch_size in batch_sizes:
        # Prepare batch: encoded is already (C, H, W) = (18, 8, 8)
        # Just need to stack to get (N, C, H, W) = (N, 18, 8, 8)
        batch_list = [encoded.copy() for _ in range(batch_size)]
        batch = np.stack(batch_list, axis=0).astype(np.float32)  # (N, 18, 8, 8)
        
        # Verify shape
        if batch.shape != (batch_size, 18, 8, 8):
            raise ValueError(
                f"Batch shape mismatch: got {batch.shape}, expected ({batch_size}, 18, 8, 8)"
            )
        
        # Try to run inference with this batch size
        try:
            # Warmup
            _ = onnx_wrapper.session.run(
                [onnx_wrapper.output_name],
                {onnx_wrapper.input_name: batch}
            )
            
            # Measure time
            num_runs = 100
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = onnx_wrapper.session.run(
                    [onnx_wrapper.output_name],
                    {onnx_wrapper.input_name: batch}
                )
            
            elapsed_time = time.time() - start_time
            time_per_inference = elapsed_time / num_runs
            
            results[batch_size] = time_per_inference
            print(f"  Batch Size {batch_size}: {time_per_inference * 1000:.4f} ms - OK")
            
        except Exception as e:
            # Model doesn't support this batch size
            unsupported_batches.append(batch_size)
            error_msg = str(e)
            if "Reshape" in error_msg or "shape" in error_msg.lower():
                print(f"  Batch Size {batch_size}: NOT SUPPORTED (model likely exported with batch_size=1)")
            else:
                print(f"  Batch Size {batch_size}: ERROR - {error_msg[:100]}")
            continue
    
    if unsupported_batches:
        print(f"\nWarning: Model does not support batch sizes: {unsupported_batches}")
        print("This is common for ONNX models exported with fixed batch_size=1.")
        print("Only batch_size=1 may be supported for this model.")
    
    if not results:
        raise RuntimeError(
            "No batch sizes were successful. Model may only support batch_size=1. "
            "Try testing with batch_sizes=[1] only."
        )
    
    return results

def run_benchmark(depth=4, epsilon=0.5, output_file="benchmark_results.txt"):
    """
    Run benchmark on all test positions.
    
    Args:
        depth: Search depth for NPS measurement
        epsilon: Weight for neural network evaluation
        output_file: Output file path (supports .txt, .json, .csv)
    """
    print("Loading ONNX model...")
    if not os.path.exists(PATH_MODEL):
        raise FileNotFoundError(f"ONNX model not found: {PATH_MODEL}")
    
    # Load ONNX model with ONNX Runtime
    sess = ort.InferenceSession(
        PATH_MODEL,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # Get input/output names from ONNX model
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"Model loaded: {PATH_MODEL}")
    print(f"Input name: {input_name}, Output name: {output_name}")
    print(f"Providers: {sess.get_providers()}")

    # Wrap ONNX model for compatibility with existing code
    class ONNXModelWrapper:
        def __init__(self, session, input_name, output_name):
            self.session = session
            self.input_name = input_name
            self.output_name = output_name
        
        def predict(self, board_encoded):
            """
            Predict using ONNX model.
            Args:
                board_encoded: numpy array with shape (18, 8, 8) from encode_board()
                                or (8, 8, 18) if from other source
            Returns:
                float: evaluation score
            """
            # Convert to numpy if needed
            if isinstance(board_encoded, torch.Tensor):
                board_encoded = board_encoded.numpy()
            
            # Ensure correct shape: (1, 18, 8, 8)
            if board_encoded.ndim == 3:
                # Check if shape is (C, H, W) = (18, 8, 8) or (H, W, C) = (8, 8, 18)
                if board_encoded.shape[0] == 18:
                    # Already (C, H, W) = (18, 8, 8) - correct format
                    board_encoded = board_encoded[np.newaxis, ...]  # (1, 18, 8, 8)
                elif board_encoded.shape[2] == 18:
                    # Shape is (H, W, C) = (8, 8, 18) -> convert to (C, H, W) = (18, 8, 8)
                    board_encoded = board_encoded.transpose(2, 0, 1)  # (18, 8, 8)
                    board_encoded = board_encoded[np.newaxis, ...]  # (1, 18, 8, 8)
                else:
                    raise ValueError(
                        f"Unexpected 3D shape: {board_encoded.shape}. "
                        f"Expected (18, 8, 8) or (8, 8, 18)"
                    )
            elif board_encoded.ndim == 4:
                # Already batch format, ensure (N, C, H, W) = (N, 18, 8, 8)
                if board_encoded.shape[1] == 18:
                    # Already (N, C, H, W) - correct
                    pass
                elif board_encoded.shape[-1] == 18:
                    # Shape is (N, H, W, C) -> convert to (N, C, H, W)
                    board_encoded = board_encoded.transpose(0, 3, 1, 2)
                else:
                    raise ValueError(
                        f"Unexpected 4D shape: {board_encoded.shape}. "
                        f"Expected (N, 18, 8, 8) or (N, 8, 8, 18)"
                    )
            else:
                raise ValueError(
                    f"Unexpected number of dimensions: {board_encoded.ndim}. "
                    f"Expected 3 or 4 dimensions"
                )
            
            # Ensure float32
            board_encoded = board_encoded.astype(np.float32)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: board_encoded})
            output = outputs[0]
            
            # Extract scalar value
            if isinstance(output, np.ndarray):
                return float(output.item() if output.size == 1 else output[0])
            return float(output)
    
    wrapped_model = ONNXModelWrapper(sess, input_name, output_name)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    # Get device info from ONNX providers
    providers = sess.get_providers()
    device_info = "CUDA" if 'CUDAExecutionProvider' in providers else "CPU"
    
    results = {
        "baseline": "ONNX Quantized Model (INT8)",
        "model_path": PATH_MODEL,
        "search_depth": depth,
        "epsilon": epsilon,
        "device": device_info,
        "providers": providers,
        "positions": [],
        "inference_times": {}
    }
    
    # Measure NPS for each position
    print("\nMeasuring NPS (Nodes Per Second)...")
    print("-" * 60)
    
    total_nps = 0
    for i, pos in enumerate(TEST_POSITIONS, 1):
        print(f"\n[{i}/{len(TEST_POSITIONS)}] {pos['name']} ({pos['category']})")
        print(f"FEN: {pos['fen']}")
        
        board = Board(pos['fen'])
        nps, nodes, elapsed = measure_nps(board, depth, epsilon, wrapped_model)
        
        total_nps += nps
        
        result_entry = {
            "name": pos['name'],
            "category": pos['category'],
            "fen": pos['fen'],
            "nps": round(nps, 2),
            "total_nodes": nodes,
            "elapsed_time": round(elapsed, 4)
        }
        results["positions"].append(result_entry)
        
        print(f"  NPS: {nps:.2f} nodes/s")
        print(f"  Total Nodes: {nodes}")
        print(f"  Elapsed Time: {elapsed:.4f}s")
    
    avg_nps = total_nps / len(TEST_POSITIONS)
    results["average_nps"] = round(avg_nps, 2)
    
    print("\n" + "-" * 60)
    print(f"Average NPS: {avg_nps:.2f} nodes/s")
    
    # Measure inference time
    print("\nMeasuring Time per Inference...")
    print("-" * 60)
    inference_results = measure_inference_time(wrapped_model)
    results["inference_times"] = {str(k): round(v * 1000, 4) for k, v in inference_results.items()}
    
    for batch_size, time_ms in inference_results.items():
        print(f"  Batch Size {batch_size}: {time_ms * 1000:.4f} ms")
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    
    if output_file.endswith('.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    elif output_file.endswith('.xlsx') or output_file.endswith('.xls'):
        try:
            import openpyxl
            from openpyxl import Workbook
            
            wb = Workbook()
            ws = wb.active
            ws.title = "Benchmark Results"
            
            # Header info
            ws.append(['Baseline', results['baseline']])
            ws.append(['Model Path', results['model_path']])
            ws.append(['Search Depth', results['search_depth']])
            ws.append(['Epsilon', results['epsilon']])
            ws.append(['Device', results['device']])
            ws.append([])
            ws.append(['Average NPS', f"{results['average_nps']} nodes/s"])
            ws.append([])
            
            # NPS Results
            ws.append(['Position', 'Category', 'NPS (nodes/s)', 'Total Nodes', 'Time (s)', 'FEN'])
            for pos in results['positions']:
                ws.append([
                    pos['name'],
                    pos['category'],
                    pos['nps'],
                    pos['total_nodes'],
                    pos['elapsed_time'],
                    pos['fen']
                ])
            
            ws.append([])
            ws.append(['Batch Size', 'Time per Inference (ms)'])
            for batch_size, time_ms in results['inference_times'].items():
                ws.append([batch_size, time_ms])
            
            wb.save(output_file)
        except ImportError:
            print("Warning: openpyxl not installed. Saving as CSV instead.")
            output_file = output_file.replace('.xlsx', '.csv').replace('.xls', '.csv')
            # Fall through to CSV
    if output_file.endswith('.csv'):
        import csv
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Baseline', f'{results["baseline"]}'])
            writer.writerow(['Average NPS', f'{results["average_nps"]} nodes/s'])
            writer.writerow([])
            writer.writerow(['Position', 'Category', 'NPS (nodes/s)', 'Total Nodes', 'Time (s)'])
            for pos in results['positions']:
                writer.writerow([
                    pos['name'],
                    pos['category'],
                    pos['nps'],
                    pos['total_nodes'],
                    pos['elapsed_time']
                ])
            writer.writerow([])
            writer.writerow(['Batch Size', 'Time per Inference (ms)'])
            for batch_size, time_ms in results['inference_times'].items():
                writer.writerow([batch_size, time_ms])
    else:  # .txt or default
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CHESS ENGINE BENCHMARK RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Baseline: {results['baseline']}\n")
            f.write(f"Model Path: {results['model_path']}\n")
            f.write(f"Search Depth: {results['search_depth']}\n")
            f.write(f"Epsilon: {results['epsilon']}\n")
            f.write(f"Device: {results['device']}\n\n")
            
            f.write("="*60 + "\n")
            f.write("NPS (Nodes Per Second) Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Average NPS: {results['average_nps']} nodes/s\n\n")
            
            for pos in results['positions']:
                f.write(f"{pos['name']} ({pos['category']})\n")
                f.write(f"  FEN: {pos['fen']}\n")
                f.write(f"  NPS: {pos['nps']} nodes/s\n")
                f.write(f"  Total Nodes: {pos['total_nodes']}\n")
                f.write(f"  Elapsed Time: {pos['elapsed_time']}s\n\n")
            
            f.write("="*60 + "\n")
            f.write("Time per Inference Results\n")
            f.write("="*60 + "\n\n")
            for batch_size, time_ms in results['inference_times'].items():
                f.write(f"Batch Size {batch_size}: {time_ms} ms\n")
    
    print("Benchmark completed!")
    print(f"\nSummary:")
    print(f"  Baseline: {results['baseline']}")
    print(f"  Average NPS: {results['average_nps']} nodes/s")
    print(f"  Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark chess engine')
    parser.add_argument('--depth', type=int, default=4, help='Search depth (default: 4)')
    parser.add_argument('--epsilon', type=float, default=0.5, help='NN weight (default: 0.5)')
    parser.add_argument('--output', type=str, default='benchmark_results_onnc.txt', 
                       help='Output file (default: benchmark_results.txt)')
    
    args = parser.parse_args()
    
    run_benchmark(depth=args.depth, epsilon=args.epsilon, output_file=args.output)

