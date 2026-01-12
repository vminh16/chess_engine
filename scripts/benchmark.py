import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.nn import NeuralNetwork
import torch
import numpy as np
import time
from scipy.stats import spearmanr
PATH_MODEL = "model/nn_parameters.pth"
PATH_DATASET = 'data/data_set_v1/dataset_full.npz'
# Benchmarking script for the NeuralNetwork model
def benchmark_model():
    np.random.seed(42)

    model = NeuralNetwork()
    model.load_model(PATH_MODEL)

    data = np.load(PATH_DATASET)
    total_samples = len(data['X'])
    indicate = np.random.choice(total_samples, min(10000, total_samples), replace=False)
    X = data['X'][indicate]
    y = data['y'][indicate]

    
    start_time = time.time()
    y_pred = np.array([model.evaluate(board) for board in X])
    end_time = time.time()
    num_evaluations = len(X)
    time_per_eval = (end_time - start_time) / num_evaluations
    print(f"Time per evaluation: {time_per_eval * 1000:.2f} ms")
    print(f"Total time for {num_evaluations} evaluations: {end_time - start_time:.4f} seconds")

    # Calculate metrics
    mse = np.mean((y_pred - y)**2)
    print(f"MSE: {mse:.4f}")
    
    # Compare with material
    piece_values = np.array([1, 3, 3, 5, 9, 0])  # P, N, B, R, Q, K
    white_material = np.sum(X[..., :6] * piece_values, axis=(1, 2, 3))
    black_material = np.sum(X[..., 6:12] * piece_values, axis=(1, 2, 3))
    material_scores = white_material - black_material
    
    # Adjust for side to move: eval from current player's perspective
    side_to_move = X[:, 0, 0, 12]  # 1 if white to move, 0 if black
    current_material_scores = material_scores * (2 * side_to_move - 1)
    
    # Spearman correlation with naive material
    spearman_naive_material, _ = spearmanr(y, current_material_scores)
    print(f"Spearman correlation of targets with naive material: {spearman_naive_material:.4f}")
    
    # Spearman correlation
    spearman_corr, _ = spearmanr(y_pred, y)
    print(f"Spearman correlation with targets: {spearman_corr:.4f}")
    
    spearman_material, _ = spearmanr(y_pred, current_material_scores)
    print(f"Spearman correlation with material: {spearman_material:.4f}")
    
    print(f'Material scores sample: {current_material_scores[:5]}')
    print(f'Neural network predictions sample: {y_pred[:5]}')

if __name__ == "__main__":
    benchmark_model()
