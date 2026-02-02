"""
Activation Functions for CNNs
=============================

Non-linear activation functions that enable neural networks to learn complex patterns.
Each activation implements forward pass and backward pass for backpropagation.

Mathematical Background:
- Without non-linearities, stacking layers = single linear transformation
- Activations introduce non-linearity, enabling universal function approximation

Key activations for CNNs:
- ReLU: Most common, fast, avoids vanishing gradients for positive values
- LeakyReLU: Fixes "dying ReLU" problem
- Softmax: Output layer for multi-class classification
"""

import numpy as np


class Activation:
    """Base class for all activation functions."""

    def forward(self, x):
        """Apply activation function."""
        raise NotImplementedError

    def backward(self, x):
        """Compute derivative of activation w.r.t. input."""
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class ReLU(Activation):
    """
    Rectified Linear Unit: f(x) = max(0, x)

    The most popular activation in CNNs. Properties:
    - Computationally efficient (just a threshold)
    - Non-saturating for positive values (no vanishing gradient)
    - Sparse activation (many zeros)

    Derivative:
        f'(x) = 1 if x > 0 else 0

    Issue: "Dying ReLU" - neurons can get stuck outputting 0
    """

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(np.float64)


class LeakyReLU(Activation):
    """
    Leaky ReLU: f(x) = x if x > 0 else alpha * x

    Fixes dying ReLU by allowing small negative gradients.

    Args:
        alpha: Slope for negative values (default: 0.01)

    Derivative:
        f'(x) = 1 if x > 0 else alpha
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x):
        return np.where(x > 0, 1.0, self.alpha)


class Sigmoid(Activation):
    """
    Sigmoid: f(x) = 1 / (1 + exp(-x))

    Squashes output to (0, 1). Used for:
    - Binary classification output
    - Gates in LSTMs/GRUs

    Derivative:
        f'(x) = f(x) * (1 - f(x))

    Issues in hidden layers:
    - Vanishing gradients for large |x|
    - Output not zero-centered
    """

    def forward(self, x):
        # Clip for numerical stability
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Tanh(Activation):
    """
    Hyperbolic Tangent: f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    Output range: (-1, 1)
    Zero-centered output is better than sigmoid.

    Derivative:
        f'(x) = 1 - tanh(x)^2
    """

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        t = np.tanh(x)
        return 1 - t ** 2


class Softmax(Activation):
    """
    Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))

    Converts logits to probability distribution (sums to 1).
    Used in output layer for multi-class classification.

    Numerical Stability:
        We subtract max(x) before exp to prevent overflow.
        This doesn't change the result: exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x))

    Note: When combined with cross-entropy loss, the gradient simplifies to:
        dL/dx = softmax(x) - y_true
    """

    def forward(self, x):
        # Handle both 1D and batched inputs
        if x.ndim == 1:
            x_shifted = x - np.max(x)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x)
        else:
            # Batched: shape (batch_size, num_classes)
            x_shifted = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x):
        """
        Full Jacobian of softmax.
        For single sample: J[i,j] = s[i] * (delta[i,j] - s[j])

        In practice, we don't use this directly because softmax + cross-entropy
        has a simpler combined gradient.
        """
        s = self.forward(x)
        if x.ndim == 1:
            return np.diag(s) - np.outer(s, s)
        else:
            # Return per-sample Jacobians
            batch_size = x.shape[0]
            n_classes = x.shape[1]
            jacobians = np.zeros((batch_size, n_classes, n_classes))
            for i in range(batch_size):
                jacobians[i] = np.diag(s[i]) - np.outer(s[i], s[i])
            return jacobians


class Linear(Activation):
    """
    Linear (Identity) activation: f(x) = x

    Used for:
    - Regression output layers
    - When you want raw scores
    """

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


# ====================================
# Activation Registry
# ====================================

ACTIVATIONS = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'leakyrelu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'linear': Linear,
    'none': Linear,
}


def get_activation(name):
    """
    Get activation function by name.

    Args:
        name: String name ('relu', 'sigmoid', etc.) or Activation instance

    Returns:
        Activation instance

    Example:
        >>> act = get_activation('relu')
        >>> act(np.array([-1, 0, 1]))
        array([0, 0, 1])
    """
    if isinstance(name, Activation):
        return name

    if name is None:
        return Linear()

    name_lower = name.lower().replace('-', '_')
    if name_lower not in ACTIVATIONS:
        available = ', '.join(ACTIVATIONS.keys())
        raise ValueError(f"Unknown activation '{name}'. Available: {available}")

    return ACTIVATIONS[name_lower]()


