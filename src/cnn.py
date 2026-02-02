"""
CNN (Convolutional Neural Network) Main Class
==============================================

This is the main class that ties everything together:
- Layer stacking
- Forward pass
- Backward pass (backpropagation)
- Training loop
- Prediction
- Model saving/loading

Architecture used:
    Input (28x28x1) -> Conv2D(32) -> BatchNorm -> ReLU -> Conv2D(32) -> BatchNorm -> ReLU -> MaxPool -> Dropout2D(0.25)
    -> Conv2D(64) -> BatchNorm -> ReLU -> Conv2D(64) -> BatchNorm -> ReLU -> MaxPool -> Dropout2D(0.25)
    -> Flatten -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.5) -> Dense(10) -> Softmax
"""

import numpy as np
from tqdm import tqdm
import time

from .layers import (Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, BatchNorm1D,
                    Dense, Flatten, Dropout, Dropout2D, Activation)
from .losses import CrossEntropyLoss, get_loss
from .optimizers import Adam, SGD, get_optimizer
from .utils import create_batches, one_hot_encode, accuracy_score


class CNN:
    """
    Convolutional Neural Network.
    
    A flexible CNN implementation with configurable architecture.
    Uses only NumPy for all computations.
    
    Default Architecture:
        Conv2D(32) -> BN -> ReLU -> Conv2D(32) -> BN -> ReLU -> MaxPool -> Dropout2D
        Conv2D(64) -> BN -> ReLU -> Conv2D(64) -> BN -> ReLU -> MaxPool -> Dropout2D
        Flatten -> Dense(128) -> BN -> ReLU -> Dropout -> Dense(10) -> Softmax
    
    Example:
        >>> from src.cnn import CNN
        >>> from src.utils import load_mnist
        >>> # Load data
        >>> X_train, y_train, X_test, y_test = load_mnist('mnist_dataset', subset_size=(1000, 200))
        >>> # Create and train model
        >>> model = CNN(input_shape=(1, 28, 28), num_classes=10)
        >>> history = model.fit(X_train, y_train, epochs=10, batch_size=32)
        >>> # Evaluate
        >>> accuracy = model.score(X_test, y_test)
        >>> print(f"Test Accuracy: {accuracy:.2%}")
    """

    def __init__(self, input_shape=(1, 28, 28), num_classes=10, 
                 use_batchnorm=True, dropout_rate=0.25, dense_dropout=0.5):
        """
        Initialize CNN.
        
        Args:
            input_shape: Input shape (channels, height, width)
            num_classes: Number of output classes
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout rate for conv blocks
            dense_dropout: Dropout rate for dense layers
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.dense_dropout = dense_dropout

        # Build the network
        self.layers = self._build_network()

        # Training state
        self.loss_fn = None
        self.optimizer = None
        self.history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'lr': []
        }

        # Store indices of conv layers for feature extraction
        self._conv_layer_indices = [i for i, layer in enumerate(self.layers) 
                                    if isinstance(layer, Conv2D)]

    def _build_network(self):
        """
        Build the CNN architecture.
        
        Architecture:
            Block 1: Conv(32) -> BN -> ReLU -> Conv(32) -> BN -> ReLU -> MaxPool -> Dropout2D
            Block 2: Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool -> Dropout2D
            FC: Flatten -> Dense(128) -> BN -> ReLU -> Dropout -> Dense(10) -> Softmax
        """
        layers = []
        channels, height, width = self.input_shape

        # ============= Block 1 =============
        # Conv2D(32, 3x3, same)
        layers.append(Conv2D(channels, 32, kernel_size=3, padding='same'))
        if self.use_batchnorm:
            layers.append(BatchNorm2D(32))
        layers.append(Activation('relu'))

        # Conv2D(32, 3x3, same)
        layers.append(Conv2D(32, 32, kernel_size=3, padding='same'))
        if self.use_batchnorm:
            layers.append(BatchNorm2D(32))
        layers.append(Activation('relu'))

        # MaxPool(2x2)
        layers.append(MaxPool2D(pool_size=2))

        # Dropout2D
        if self.dropout_rate > 0:
            layers.append(Dropout2D(self.dropout_rate))

        # After pooling: height/2, width/2
        h1 = height // 2
        w1 = width // 2

        # ============= Block 2 =============
        # Conv2D(64, 3x3, same)
        layers.append(Conv2D(32, 64, kernel_size=3, padding='same'))
        if self.use_batchnorm:
            layers.append(BatchNorm2D(64))
        layers.append(Activation('relu'))

        # Conv2D(64, 3x3, same)
        layers.append(Conv2D(64, 64, kernel_size=3, padding='same'))
        if self.use_batchnorm:
            layers.append(BatchNorm2D(64))
        layers.append(Activation('relu'))

        # MaxPool(2x2)
        layers.append(MaxPool2D(pool_size=2))

        # Dropout2D
        if self.dropout_rate > 0:
            layers.append(Dropout2D(self.dropout_rate))

        # After pooling: h1/2, w1/2
        h2 = h1 // 2
        w2 = w1 // 2
        flatten_dim = 64 * h2 * w2

        # ============= Fully Connected =============
        # Flatten
        layers.append(Flatten())

        # Dense(128)
        layers.append(Dense(flatten_dim, 128))
        if self.use_batchnorm:
            layers.append(BatchNorm1D(128))
        layers.append(Activation('relu'))

        # Dropout
        if self.dense_dropout > 0:
            layers.append(Dropout(self.dense_dropout))

        # Output: Dense(num_classes) + Softmax
        layers.append(Dense(128, self.num_classes))
        layers.append(Activation('softmax'))

        return layers

    def forward(self, x, training=True):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor, shape (batch, channels, height, width)
            training: Whether in training mode (affects dropout, batchnorm)
        
        Returns:
            Output predictions, shape (batch, num_classes)
        """
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        """
        Backward pass through the network.
        
        Propagates gradients backwards through each layer,
        computing gradients for all trainable parameters.
        
        Args:
            grad: Gradient from loss function
        
        Returns:
            Gradient w.r.t. input
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.001,
            optimizer='adam', validation_data=None, verbose=True,
            lr_scheduler=None, early_stopping_patience=None):
        """
        Train the CNN.
        
        Args:
            X: Training images, shape (N, C, H, W)
            y: Training labels, shape (N,) or (N, num_classes)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Initial learning rate
            optimizer: 'adam' or 'sgd'
            validation_data: Tuple (X_val, y_val) for validation
            verbose: Print progress
            lr_scheduler: Learning rate scheduler function
            early_stopping_patience: Stop if no improvement for this many epochs
        
        Returns:
            Training history dictionary
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Set up loss function
        self.loss_fn = CrossEntropyLoss()

        # Set up optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)

        if lr_scheduler is not None:
            self.optimizer.set_lr_scheduler(lr_scheduler)

        # One-hot encode labels if needed
        if y.ndim == 1:
            y_onehot = one_hot_encode(y, self.num_classes)
        else:
            y_onehot = y

        # Validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Reset history
        self.history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'lr': []
        }

        # Training loop
        n_batches = (len(X) + batch_size - 1) // batch_size

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            n_samples = 0
    
            # Progress bar for batches
            if verbose:
                pbar = tqdm(create_batches(X, y_onehot, batch_size, shuffle=True),
                           total=n_batches, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                pbar = create_batches(X, y_onehot, batch_size, shuffle=True)
    
            for X_batch, y_batch in pbar:
                # Forward pass
                predictions = self.forward(X_batch, training=True)

                # Compute loss
                loss = self.loss_fn(predictions, y_batch)
                epoch_loss += loss * len(X_batch)

                # Track accuracy
                pred_labels = np.argmax(predictions, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(pred_labels == true_labels)
                n_samples += len(X_batch)

                # Backward pass
                grad = self.loss_fn.backward(predictions, y_batch)
                self.backward(grad)

                # Update weights
                self.optimizer.step(self.layers)

                # Update progress bar
                if verbose and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'loss': f'{epoch_loss/n_samples:.4f}',
                        'acc': f'{epoch_correct/n_samples:.4f}'
                    })

            # Epoch metrics
            avg_loss = epoch_loss / len(X)
            avg_accuracy = epoch_correct / len(X)

            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_accuracy)
            self.history['lr'].append(self.optimizer.get_lr())

            # Validation
            if validation_data is not None:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

                # Early stopping
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"\nEarly stopping at epoch {epoch + 1}")
                            break

            # Print epoch summary
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_accuracy:.4f}"
                if validation_data is not None:
                    msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}"
                msg += f" - LR: {self.optimizer.get_lr():.6f}"
                print(msg)

        return self.history

    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input images, shape (N, C, H, W)
        
        Returns:
            Predicted probabilities, shape (N, num_classes)
        """
        X = np.asarray(X, dtype=np.float64)
        return self.forward(X, training=False)

    def predict_classes(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input images
        
        Returns:
            Predicted class indices, shape (N,)
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        """
        Evaluate model on data.
        
        Args:
            X: Images
            y: Labels
        
        Returns:
            Tuple of (loss, accuracy)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        predictions = self.predict(X)

        if y.ndim == 1:
            y_onehot = one_hot_encode(y, self.num_classes)
        else:
            y_onehot = y

        if self.loss_fn is None:
            self.loss_fn = CrossEntropyLoss()

        loss = self.loss_fn(predictions, y_onehot)
        accuracy = accuracy_score(y, predictions)

        return loss, accuracy

    def score(self, X, y):
        """Compute accuracy."""
        _, accuracy = self.evaluate(X, y)
        return accuracy

    def get_feature_maps(self, X, layer_index=None):
        """
        Get feature maps from convolutional layers.
        
        Args:
            X: Input image(s), shape (N, C, H, W) or (C, H, W)
            layer_index: Specific conv layer index, or None for all
        
        Returns:
            List of feature maps from conv layers
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X = X[np.newaxis, ...]

        feature_maps = []
        output = X

        for i, layer in enumerate(self.layers):
            output = layer.forward(output, training=False)

            if isinstance(layer, Conv2D):
                if layer_index is None or i == layer_index:
                    feature_maps.append({
                        'layer_index': i,
                        'layer': layer,
                        'feature_map': output.copy()
                    })

        return feature_maps

    def get_filters(self, layer_index=None):
        """
        Get filter weights from convolutional layers.
        
        Args:
            layer_index: Specific conv layer index, or None for all
        
        Returns:
            List of filter weights
        """
        filters = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                if layer_index is None or i == layer_index:
                    filters.append({
                        'layer_index': i,
                        'layer': layer,
                        'weights': layer.params['weight'].copy()
                    })

        return filters

    def summary(self):
        """Print model summary."""
        print("\n" + "=" * 70)
        print("CNN Model Summary")
        print("=" * 70)
        print(f"Input shape: {self.input_shape}")
        print(f"Output classes: {self.num_classes}")
        print("-" * 70)

        total_params = 0

        for i, layer in enumerate(self.layers):
            n_params = 0
            if hasattr(layer, 'params'):
                for param in layer.params.values():
                    n_params += param.size

            total_params += n_params
            print(f"{i:3d}. {str(layer):<45} Params: {n_params:,}")

        print("-" * 70)
        print(f"Total trainable parameters: {total_params:,}")
        print("=" * 70 + "\n")

        return total_params

    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save file (.npz)
        """
        params = {}

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                for name, param in layer.params.items():
                    params[f'layer_{i}_{name}'] = param
    
            # Save batch norm running statistics
            if isinstance(layer, (BatchNorm2D, BatchNorm1D)):
                params[f'layer_{i}_running_mean'] = layer.running_mean
                params[f'layer_{i}_running_var'] = layer.running_var

        # Save metadata
        params['input_shape'] = np.array(self.input_shape)
        params['num_classes'] = np.array(self.num_classes)

        np.savez(filepath, **params)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model (.npz)
        """
        data = np.load(filepath, allow_pickle=True)

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for name in layer.params.keys():
                    key = f'layer_{i}_{name}'
                    if key in data:
                        layer.params[name] = data[key]
    
            # Load batch norm running statistics
            if isinstance(layer, (BatchNorm2D, BatchNorm1D)):
                mean_key = f'layer_{i}_running_mean'
                var_key = f'layer_{i}_running_var'
                if mean_key in data:
                    layer.running_mean = data[mean_key]
                if var_key in data:
                    layer.running_var = data[var_key]

        print(f"Model loaded from {filepath}")

    def __repr__(self):
        return f"CNN(input_shape={self.input_shape}, num_classes={self.num_classes})"


def benchmark_inference(model, X, n_runs=100):
    """
    Benchmark inference time.
    
    Args:
        model: CNN model
        X: Sample input
        n_runs: Number of runs
    
    Returns:
        Dictionary with timing statistics
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X[np.newaxis, ...]

    # Warmup
    for _ in range(5):
        _ = model.predict(X)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'n_samples': X.shape[0],
        'per_sample_ms': np.mean(times) / X.shape[0]
    }
