import numpy as np
import os
import glob

def concatenate_datasets(folder_path):
    """
    Concatenate all dataset_batch_*.npz files in the given folder.
    Saves the concatenated data to dataset_full.npz in the same folder.
    """
    # Find all batch files
    pattern = os.path.join(folder_path, 'dataset_batch_*.npz')
    batch_files = glob.glob(pattern)
    
    if not batch_files:
        print(f"No dataset_batch_*.npz files found in {folder_path}")
        return
    
    # Sort files by batch number
    batch_files.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
    
    # Filter to only batches 0 to 9
    filtered_files = [f for f in batch_files if 0 <= int(os.path.basename(f).split('_')[2].split('.')[0]) <= 9]
    
    # Initialize lists
    all_X = []
    all_y = []
    all_fens = []
    
    for file_path in filtered_files:
        data = np.load(file_path)
        all_X.append(data['X'])
        all_y.append(data['y'])
        all_fens.extend(data['fens'])  # fens is array of strings
    
    # Concatenate
    X_concat = np.concatenate(all_X, axis=0)
    y_concat = np.concatenate(all_y, axis=0)
    fens_concat = np.array(all_fens)
    
    # Save
    output_path = os.path.join(folder_path, 'dataset_full.npz')
    np.savez(output_path, X=X_concat, y=y_concat, fens=fens_concat)
    print(f"Concatenated data saved to {output_path}")
    print(f"Total samples: {len(X_concat)}")

if __name__ == "__main__":
    folder_path = './data/dataset'  # Directory containing the batch files
    concatenate_datasets(folder_path)