"""
PyTorch CNN Implementation
==========================

CNN using PyTorch for comparison with the from-scratch version.

Architecture matches the NumPy implementation:
    Input -> Conv2d(32) -> BN -> ReLU -> Conv2d(32) -> BN -> ReLU -> MaxPool -> Dropout2D
    -> Conv2d(64) -> BN -> ReLU -> Conv2d(64) -> BN -> ReLU -> MaxPool -> Dropout2D
    -> Flatten -> Dense(128) -> BN -> ReLU -> Dropout -> Dense(10) -> Softmax
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class CNNPyTorch(nn.Module):
    """
    PyTorch CNN with same architecture as NumPy version.
    
    Example:
        >>> model = CNNPyTorch(input_channels=1, num_classes=10)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 10])
    """

    def __init__(self, input_channels=1, num_classes=10, dropout_rate=0.25, 
                 dense_dropout=0.5, use_batchnorm=True):
        """
        Initialize PyTorch CNN.
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for conv blocks
            dense_dropout: Dropout rate for dense layers
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()

        self.use_batchnorm = use_batchnorm

        # Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout2d_1 = nn.Dropout2d(dropout_rate)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2d_2 = nn.Dropout2d(dropout_rate)

        # Flatten dimension: 64 * 7 * 7 = 3136 for 28x28 input
        # After two 2x2 poolings: 28 -> 14 -> 7
        self.flatten_dim = 64 * 7 * 7

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout(dense_dropout)
        self.fc2 = nn.Linear(128, num_classes)

        # Store feature maps for visualization
        self.feature_maps = None

    def forward(self, x):
        """Forward pass."""
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout2d_1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Store feature maps for visualization
        self.feature_maps = x

        x = self.pool2(x)
        x = self.dropout2d_2(x)

        # Flatten
        x = self.flatten(x)

        # FC layers
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        """Make predictions (with softmax)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict_classes(self, x):
        """Predict class labels."""
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)

    def get_feature_maps(self, x, layer_name='conv4'):
        """
        Get feature maps from a specific layer.
        
        Args:
            x: Input tensor
            layer_name: Name of the layer
        
        Returns:
            Feature maps tensor
        """
        self.eval()

        # Hook to capture feature maps
        features = {}

        def hook(module, input, output):
            features['output'] = output

        # Register hook
        layer = getattr(self, layer_name)
        handle = layer.register_forward_hook(hook)

        with torch.no_grad():
            _ = self.forward(x)

        handle.remove()

        return features.get('output')

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_pytorch_model(model, train_loader, val_loader=None, epochs=10,
                       learning_rate=0.001, device='cpu', verbose=True):
    """
    Train PyTorch model.
    
    Args:
        model: CNNPyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'
        verbose: Print progress
    
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {
        'loss': [], 'accuracy': [],
        'val_loss': [], 'val_accuracy': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        history['loss'].append(avg_train_loss)
        history['accuracy'].append(avg_train_acc)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_x.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            avg_val_loss = val_loss / val_total
            avg_val_acc = val_correct / val_total

            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(avg_val_acc)

        if verbose:
            msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Acc: {avg_train_acc:.4f}"
            if val_loader:
                msg += f" - Val Loss: {avg_val_loss:.4f} - Val Acc: {avg_val_acc:.4f}"
            print(msg)

    return history


def evaluate_pytorch_model(model, data_loader, device='cpu'):
    """
    Evaluate PyTorch model.
    
    Args:
        model: CNNPyTorch model
        data_loader: Data loader
        device: Device
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return total_loss / total, correct / total


def benchmark_pytorch_inference(model, sample_input, n_runs=100, device='cpu'):
    """
    Benchmark PyTorch inference time.
    
    Args:
        model: CNNPyTorch model
        sample_input: Sample input tensor
        n_runs: Number of runs
        device: Device
    
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    sample_input = sample_input.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)

    # Synchronize if on GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(sample_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'n_samples': sample_input.shape[0],
        'per_sample_ms': np.mean(times) / sample_input.shape[0]
    }


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Create PyTorch data loaders from NumPy arrays.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        batch_size: Batch size
    
    Returns:
        train_loader, test_loader
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
