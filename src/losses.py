"""
Loss Functions for CNNs
=======================

Loss functions measure how wrong the model's predictions are.
The goal of training is to minimize the loss.

Each loss implements:
- forward(predictions, targets): Compute loss value
- backward(predictions, targets): Compute gradient for backpropagation

Mathematical Background:
- Loss guides optimization by providing error signal
- Gradient of loss w.r.t. predictions tells us how to adjust weights
"""

import numpy as np


class Loss:
    """Base class for loss functions."""

    def forward(self, predictions, targets):
        """Compute loss value."""
        raise NotImplementedError

    def backward(self, predictions, targets):
        """Compute gradient of loss w.r.t. predictions."""
        raise NotImplementedError

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss for multi-class classification.

    Formula: L = -sum(y_true * log(y_pred))

    For single correct class k: L = -log(p_k)

    Args:
        epsilon: Small constant to prevent log(0)
    """

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Compute cross-entropy loss.

        Args:
            predictions: Softmax outputs, shape (batch, classes) or (batch,)
            targets: One-hot encoded or integer class labels

        Returns:
            Mean loss across batch
        """
        # Handle integer labels
        if targets.ndim == 1:
            num_classes = predictions.shape[-1]
            targets_onehot = np.zeros_like(predictions)
            targets_onehot[np.arange(len(targets)), targets.astype(int)] = 1
            targets = targets_onehot

        # Clip for numerical stability
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)

        # Cross-entropy: -sum(y * log(p))
        loss = -np.sum(targets * np.log(predictions_clipped), axis=-1)
        return np.mean(loss)

    def backward(self, predictions, targets):
        """
        Compute gradient of cross-entropy loss.

        When predictions come from softmax, gradient is simply:
            dL/dz = predictions - targets

        This is one of the beautiful results that makes neural nets practical.
        """
        # Handle integer labels
        if targets.ndim == 1:
            num_classes = predictions.shape[-1]
            targets_onehot = np.zeros_like(predictions)
            targets_onehot[np.arange(len(targets)), targets.astype(int)] = 1
            targets = targets_onehot

        batch_size = predictions.shape[0]

        # Combined softmax + cross-entropy gradient
        grad = (predictions - targets) / batch_size
        return grad


class MSELoss(Loss):
    """
    Mean Squared Error Loss for regression.

    Formula: L = (1/n) * sum((y_pred - y_true)^2)

    Gradient: dL/dy_pred = (2/n) * (y_pred - y_true)
    """

    def forward(self, predictions, targets):
        """Compute mean squared error."""
        return np.mean((predictions - targets) ** 2)

    def backward(self, predictions, targets):
        """Compute gradient of MSE."""
        batch_size = predictions.shape[0]
        n_elements = predictions.size / batch_size
        grad = 2 * (predictions - targets) / (batch_size * n_elements)
        return grad


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy for binary classification.

    Formula: L = -[y*log(p) + (1-y)*log(1-p)]

    Use when output is single probability from sigmoid.
    """

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        predictions = predictions.flatten()
        targets = targets.flatten()

        p = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        loss = -(targets * np.log(p) + (1 - targets) * np.log(1 - p))
        return np.mean(loss)

    def backward(self, predictions, targets):
        predictions = predictions.flatten()
        targets = targets.flatten()

        p = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        grad = (-targets / p + (1 - targets) / (1 - p)) / len(targets)
        return grad.reshape(predictions.shape)


# ============================================================================
# Loss Registry
# ============================================================================

LOSSES = {
    'cross_entropy': CrossEntropyLoss,
    'crossentropy': CrossEntropyLoss,
    'ce': CrossEntropyLoss,
    'categorical_crossentropy': CrossEntropyLoss,
    'mse': MSELoss,
    'mean_squared_error': MSELoss,
    'bce': BinaryCrossEntropyLoss,
    'binary_crossentropy': BinaryCrossEntropyLoss,
}


def get_loss(name):
    """
    Get loss function by name.

    Args:
        name: String name or Loss instance

    Returns:
        Loss instance
    """
    if isinstance(name, Loss):
        return name

    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    if name_lower not in LOSSES:
        available = ', '.join(sorted(set(LOSSES.keys())))
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")

    return LOSSES[name_lower]()
