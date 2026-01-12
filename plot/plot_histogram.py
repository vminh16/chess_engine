import numpy as np
import matplotlib.pyplot as plt
import os

# Load data from dataset_batch_0.npz
data = np.load('dataset_batch_0.npz')
y = data['y']

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(y, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Evaluation Scores')
plt.xlabel('Evaluation Score')
plt.ylabel('Frequency')
plt.grid(True)

# Ensure directories exist
os.makedirs('plot/visual', exist_ok=True)

# Save the plot
plt.savefig('plot/visual/histogram.png', dpi=300, bbox_inches='tight')
print("Histogram saved to plot/visual/histogram.png")