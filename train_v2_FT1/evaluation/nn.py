import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, output_mode: str = "tanh"):
        super(NeuralNetwork, self).__init__()
        self.input_shape = (8, 8, 18)
        self.output_mode = output_mode
        
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
        x = self.fc(x)
        if self.output_mode == "tanh":
            x = self.tanh(x)
        elif self.output_mode != "linear":
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        
        return x.squeeze()  # Remove batch dimension if single sample

    def _compute_region_weights(self, targets, region_weights):
        """Compute light region weights without changing sampling/distribution."""
        abs_targets = torch.abs(targets)
        center_mask = abs_targets <= 0.1
        mid_mask = (abs_targets > 0.1) & (abs_targets <= 0.5)
        decisive_mask = abs_targets > 0.5

        weights = torch.ones_like(targets)
        weights[center_mask] = region_weights["center"]
        weights[mid_mask] = region_weights["mid"]
        weights[decisive_mask] = region_weights["decisive"]
        return weights

    def train_model(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        learning_rate=0.001,
        epochs=10,
        batch_size=32,
        clip_norm=1.0,
        loss_name="huber",
        huber_delta=0.1,
        use_region_weighting=True,
        region_weights=None,
        scheduler_name="cosine_warm_restarts",
        scheduler_t0=5,
        scheduler_t_mult=2,
        scheduler_eta_min=1e-6,
    ):
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
        
        if region_weights is None:
            # Balanced-first objective: keep center strongest, de-emphasize decisive.
            region_weights = {"center": 1.0, "mid": 0.7, "decisive": 0.4}

        # Loss function and optimizer
        if loss_name.lower() == "huber":
            criterion = nn.HuberLoss(delta=huber_delta, reduction="none")
        elif loss_name.lower() == "mse":
            criterion = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = None
        if scheduler_name == "cosine_warm_restarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, scheduler_t0),
                T_mult=max(1, scheduler_t_mult),
                eta_min=scheduler_eta_min,
            )
        
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
                targets = batch_y.squeeze()
                base_loss = criterion(outputs, targets)
                if use_region_weighting:
                    sample_weights = self._compute_region_weights(targets, region_weights)
                    loss = (base_loss * sample_weights).mean()
                else:
                    loss = base_loss.mean()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
                
                # Update parameters
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / (n_samples // batch_size + 1))

            if scheduler is not None:
                scheduler.step(epoch + 1)
            
            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    val_outputs = self(X_val_tensor)
                    y_val_squeezed = y_val_tensor.squeeze()
                    if loss_name.lower() == "huber":
                        val_loss_value = F.huber_loss(
                            val_outputs, y_val_squeezed, delta=huber_delta, reduction="mean"
                        ).item()
                    else:
                        val_loss_value = torch.mean((val_outputs - y_val_squeezed) ** 2).item()
                    val_losses.append(val_loss_value)

                    # Track center-region MSE explicitly (balanced-first goal).
                    val_err = val_outputs - y_val_squeezed
                    center_mask = torch.abs(y_val_squeezed) <= 0.1
                    if torch.any(center_mask):
                        center_mse = torch.mean((val_err[center_mask]) ** 2).item()
                    else:
                        center_mse = float("nan")

                    decisive_mask = torch.abs(y_val_squeezed) > 0.5
                    if torch.any(decisive_mask):
                        decisive_mse = torch.mean((val_err[decisive_mask]) ** 2).item()
                    else:
                        decisive_mse = float("nan")

                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  lr={current_lr:.2e} val_loss={val_loss_value:.6f} "
                        f"center_mse={center_mse:.6f} decisive_mse={decisive_mse:.6f}"
                    )
        
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

    def predict(self, board_representation):
        """Search code uses predict(); keep this alias for compatibility."""
        return self.evaluate(board_representation)

    def save_model(self, path):
        """Save model state"""
        payload = {
            "state_dict": self.state_dict(),
            "output_mode": self.output_mode,
        }
        torch.save(payload, path)
    
    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.output_mode = checkpoint.get("output_mode", self.output_mode)
            self.load_state_dict(checkpoint["state_dict"])
        else:
            # Backward compatibility for old pure state_dict checkpoints.
            self.load_state_dict(checkpoint)
        self.eval()






    