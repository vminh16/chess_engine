import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.model_selection import train_test_split
from evaluation.nn import NeuralNetwork
import matplotlib.pyplot as plt


def visualize_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./plot/visual/loss_plot.png')
    plt.show()
    


PATH = 'data/data_set_v1/dataset_full.npz'

# data = np.load(PATH)
# X = data['X']
# y = data['y']
# print(data.files)
# print(data['X'][0])
def main():
    np.random.seed(42)
    data = np.load(PATH)
    X = data['X']
    y = data['y']
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_1, y_1, test_size=0.1, random_state=42)

    print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    """
        Mạng nơ-ron nhân tạo để đánh giá bàn cờ.

        Args:
            board_representation: Biểu diễn bàn cờ dưới dạng mảng numpy với shape (8, 8, 18)
            epochs: 20
            LR: 0.001
            batch_size: 32
            Loss function: Mean Squared Error (MSE)
        Returns:
            parameters: Tham số đã được huấn luyện của mạng nơ-ron.
        """
    nn = NeuralNetwork()
    #vizualize_network(nn)

    train_losses, val_losses = nn.train_model(X_train, y_train, X_val=X_val, y_val=y_val, learning_rate=0.01, epochs=16, batch_size=32, clip_norm=1.0)

    visualize_losses(train_losses, val_losses)

    # Lưu tham số đã huấn luyện
    os.makedirs('model', exist_ok=True)
    nn.save_model('model/nn_parameters.pth')
    print("Model parameters saved to 'model/nn_parameters.pth'")
    print("Training complete.")

if __name__ == "__main__":
    main()