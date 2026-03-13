import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_shape = (8, 8, 18)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(32, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure input is tensor and correct shape (N, C, H, W)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 3:  # (H, W, C) -> (C, H, W) -> (1, C, H, W)
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4 and x.shape[1] != 18:  # Assume (N, H, W, C) -> (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
        # If already (N, C, H, W), do nothing
        
        # Convolutional layer 1
        x = self.relu(self.conv1(x))
        
        # Convolutional layer 2
        x = self.relu(self.conv2(x))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layer
        x = self.tanh(self.fc(x))
        
        return x.squeeze()  # Remove batch dimension if single sample

    def train_model(self, X, y, X_val=None, y_val=None, learning_rate=0.001, epochs=10, batch_size=32, clip_norm=1.0):
        """
        Train the neural network using PyTorch's autograd.
        
        Args:
            X: List of input boards (numpy arrays)
            y: List of target values
            X_val: Validation inputs
            y_val: Validation targets
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size
            clip_norm: Gradient clipping norm
        """
        self.train()  # Set to training mode
        
        # Convert to tensors
        X_tensor = torch.stack([torch.from_numpy(x).float() for x in X])
        X_tensor = X_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.stack([torch.from_numpy(x).float() for x in X_val])
            X_val_tensor = X_val_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        n_samples = len(X)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0.0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_y = y_shuffled[start:end]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
                
                # Update parameters
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / (n_samples // batch_size + 1))
            
            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    val_outputs = self(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor.squeeze())
                    val_losses.append(val_loss.item())
        
        return train_losses, val_losses

    def evaluate(self, board_representation):
        """
        Đánh giá bàn cờ sử dụng mạng nơ-ron nhân tạo.

        Args:
            board_representation: Biểu diễn bàn cờ dưới dạng mảng numpy với shape (8, 8, 18)
        Returns:
            Giá trị float trong [-1, 1], dương nếu bên hiện tại có lợi
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            input_tensor = torch.from_numpy(board_representation).float().unsqueeze(0)
            output = self(input_tensor)
            return output.item()

    def save_model(self, path):
        """Save model state"""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """Load model state"""
        self.load_state_dict(torch.load(path))
        self.eval()






    