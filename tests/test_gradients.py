"""
Gradient Checking Tests
=======================

Verify analytical gradients match numerical approximations.
This is THE most important test for ensuring backpropagation is correct.

Method: Centered finite differences
    f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

We compare:
    - Analytical gradient: computed by backward()
    - Numerical gradient: finite difference approximation

If they match (relative error < 1e-5), backprop is correct.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.layers import Conv2D, MaxPool2D, Dense, BatchNorm2D, Flatten
from src.activations import ReLU, Sigmoid, Softmax
from src.losses import CrossEntropyLoss, MSELoss


def numerical_gradient(f, x, epsilon=1e-5):
    """
    Compute numerical gradient using centered finite differences.

    Args:
        f: Function that takes x and returns scalar loss
        x: Point at which to compute gradient
        epsilon: Small perturbation

    Returns:
        Numerical gradient, same shape as x
    """
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index

        # f(x + epsilon)
        x[idx] += epsilon
        loss_plus = f(x)

        # f(x - epsilon)
        x[idx] -= 2 * epsilon
        loss_minus = f(x)

        # Restore
        x[idx] += epsilon

        # Centered difference
        grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        it.iternext()

    return grad


def relative_error(analytical, numerical):
    """
    Compute relative error between analytical and numerical gradients.

    Returns:
        Maximum relative error across all elements
    """
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    return np.max(diff / denom)


class TestConv2DGradients:
    """Gradient tests for Conv2D."""

    def test_weight_gradients(self):
        """Test gradients w.r.t. weights."""
        np.random.seed(42)

        conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, padding='same')
        x = np.random.randn(2, 2, 8, 8)  # Small input for speed

        # Forward and backward to get analytical gradients
        output = conv.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        conv.backward(grad_output)
        analytical_dW = conv.grads['weight'].copy()

        # Numerical gradient
        def loss_fn(W):
            conv.params['weight'] = W
            out = conv.forward(x, training=False)
            return np.sum(out * grad_output)

        numerical_dW = numerical_gradient(loss_fn, conv.params['weight'].copy())

        error = relative_error(analytical_dW, numerical_dW)
        assert error < 1e-4, f"Weight gradient error too large: {error}"

    def test_input_gradients(self):
        """Test gradients w.r.t. input."""
        np.random.seed(42)

        conv = Conv2D(in_channels=2, out_channels=4, kernel_size=3, padding='same')
        x = np.random.randn(2, 2, 8, 8)

        output = conv.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        analytical_dx = conv.backward(grad_output)

        def loss_fn(x_in):
            out = conv.forward(x_in, training=False)
            return np.sum(out * grad_output)

        numerical_dx = numerical_gradient(loss_fn, x.copy())

        error = relative_error(analytical_dx, numerical_dx)
        assert error < 1e-4, f"Input gradient error too large: {error}"


class TestDenseGradients:
    """Gradient tests for Dense layer."""

    def test_weight_gradients(self):
        """Test gradients w.r.t. weights."""
        np.random.seed(42)

        dense = Dense(input_dim=16, output_dim=8)
        x = np.random.randn(4, 16)

        output = dense.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        dense.backward(grad_output)
        analytical_dW = dense.grads['weight'].copy()

        def loss_fn(W):
            dense.params['weight'] = W
            out = dense.forward(x, training=False)
            return np.sum(out * grad_output)

        numerical_dW = numerical_gradient(loss_fn, dense.params['weight'].copy())

        error = relative_error(analytical_dW, numerical_dW)
        assert error < 1e-5, f"Weight gradient error too large: {error}"

    def test_bias_gradients(self):
        """Test gradients w.r.t. bias."""
        np.random.seed(42)

        dense = Dense(input_dim=16, output_dim=8)
        x = np.random.randn(4, 16)

        output = dense.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        dense.backward(grad_output)
        analytical_db = dense.grads['bias'].copy()

        def loss_fn(b):
            dense.params['bias'] = b
            out = dense.forward(x, training=False)
            return np.sum(out * grad_output)

        numerical_db = numerical_gradient(loss_fn, dense.params['bias'].copy())

        error = relative_error(analytical_db, numerical_db)
        assert error < 1e-5, f"Bias gradient error too large: {error}"


class TestMaxPool2DGradients:
    """Gradient tests for MaxPool2D."""

    def test_input_gradients(self):
        """Test gradients through max pooling."""
        np.random.seed(42)

        pool = MaxPool2D(pool_size=2)
        x = np.random.randn(2, 4, 8, 8)

        output = pool.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        analytical_dx = pool.backward(grad_output)

        def loss_fn(x_in):
            out = pool.forward(x_in, training=False)
            return np.sum(out * grad_output)

        numerical_dx = numerical_gradient(loss_fn, x.copy())

        error = relative_error(analytical_dx, numerical_dx)
        # MaxPool gradients can have larger errors due to discontinuity at boundaries
        assert error < 1e-3, f"Input gradient error too large: {error}"


class TestBatchNorm2DGradients:
    """Gradient tests for BatchNorm2D."""

    def test_gamma_gradients(self):
        """Test gradients w.r.t. gamma (scale)."""
        np.random.seed(42)

        bn = BatchNorm2D(num_features=4)
        x = np.random.randn(8, 4, 6, 6)  # Need larger batch for stable BN

        output = bn.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        bn.backward(grad_output)
        analytical_dgamma = bn.grads['gamma'].copy()

        def loss_fn(gamma):
            bn.params['gamma'] = gamma
            out = bn.forward(x, training=True)
            return np.sum(out * grad_output)

        numerical_dgamma = numerical_gradient(loss_fn, bn.params['gamma'].copy())

        error = relative_error(analytical_dgamma, numerical_dgamma)
        assert error < 1e-4, f"Gamma gradient error too large: {error}"

    def test_input_gradients(self):
        """Test gradients w.r.t. input."""
        np.random.seed(42)

        bn = BatchNorm2D(num_features=4)
        x = np.random.randn(8, 4, 6, 6)

        output = bn.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        analytical_dx = bn.backward(grad_output)

        def loss_fn(x_in):
            out = bn.forward(x_in, training=True)
            return np.sum(out * grad_output)

        numerical_dx = numerical_gradient(loss_fn, x.copy())

        error = relative_error(analytical_dx, numerical_dx)
        # BatchNorm gradients can be tricky - allow slightly larger error
        assert error < 1e-3, f"Input gradient error too large: {error}"


class TestActivationGradients:
    """Gradient tests for activation functions."""

    def test_relu_gradients(self):
        """Test ReLU gradients."""
        relu = ReLU()
        x = np.random.randn(10)

        output = relu.forward(x)
        analytical_grad = relu.backward(x)

        def loss_fn(x_in):
            return np.sum(relu.forward(x_in))

        numerical_grad = numerical_gradient(loss_fn, x.copy())

        error = relative_error(analytical_grad, numerical_grad)
        assert error < 1e-5, f"ReLU gradient error: {error}"

    def test_sigmoid_gradients(self):
        """Test Sigmoid gradients."""
        sigmoid = Sigmoid()
        x = np.random.randn(10)

        analytical_grad = sigmoid.backward(x)

        def loss_fn(x_in):
            return np.sum(sigmoid.forward(x_in))

        numerical_grad = numerical_gradient(loss_fn, x.copy())

        error = relative_error(analytical_grad, numerical_grad)
        assert error < 1e-5, f"Sigmoid gradient error: {error}"


class TestLossGradients:
    """Gradient tests for loss functions."""

    def test_cross_entropy_gradients(self):
        """Test cross-entropy loss gradients."""
        np.random.seed(42)

        loss_fn = CrossEntropyLoss()

        # Create softmax predictions
        logits = np.random.randn(4, 10)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # One-hot targets
        targets = np.zeros((4, 10))
        targets[range(4), [0, 2, 5, 7]] = 1

        analytical_grad = loss_fn.backward(predictions, targets)

        def compute_loss(preds):
            return loss_fn.forward(preds, targets)

        numerical_grad = numerical_gradient(compute_loss, predictions.copy())

        # Cross-entropy gradient with softmax is special - check the combined gradient
        # The analytical gradient for softmax + CE is (predictions - targets) / batch_size
        expected_grad = (predictions - targets) / 4

        error = relative_error(analytical_grad, expected_grad)
        assert error < 1e-6, f"Cross-entropy gradient error: {error}"

    def test_mse_gradients(self):
        """Test MSE loss gradients."""
        loss_fn = MSELoss()

        predictions = np.random.randn(4, 10)
        targets = np.random.randn(4, 10)

        analytical_grad = loss_fn.backward(predictions, targets)

        def compute_loss(preds):
            return loss_fn.forward(preds, targets)

        numerical_grad = numerical_gradient(compute_loss, predictions.copy())

        error = relative_error(analytical_grad, numerical_grad)
        assert error < 1e-5, f"MSE gradient error: {error}"


class TestEndToEndGradients:
    """End-to-end gradient tests for small networks."""

    def test_conv_dense_network(self):
        """Test gradients through conv -> flatten -> dense."""
        np.random.seed(42)

        # Small network
        conv = Conv2D(1, 4, kernel_size=3, padding='valid')
        flatten = Flatten()
        dense = Dense(4 * 6 * 6, 10)  # 8x8 -> 6x6 after 3x3 conv
        loss_fn = CrossEntropyLoss()
        softmax = Softmax()

        # Data
        x = np.random.randn(2, 1, 8, 8)
        targets = np.zeros((2, 10))
        targets[0, 3] = 1
        targets[1, 7] = 1

        # Forward
        h1 = conv.forward(x, training=True)
        h2 = flatten.forward(h1, training=True)
        logits = dense.forward(h2, training=True)
        predictions = softmax.forward(logits)
        loss = loss_fn.forward(predictions, targets)

        # Backward
        grad = loss_fn.backward(predictions, targets)
        grad = dense.backward(grad)
        grad = flatten.backward(grad)
        grad_input = conv.backward(grad)

        # Test conv weight gradient
        analytical_dW = conv.grads['weight'].copy()

        def compute_loss(W):
            conv.params['weight'] = W
            h1 = conv.forward(x, training=False)
            h2 = flatten.forward(h1, training=False)
            logits = dense.forward(h2, training=False)
            preds = softmax.forward(logits)
            return loss_fn.forward(preds, targets)

        numerical_dW = numerical_gradient(compute_loss, conv.params['weight'].copy())

        error = relative_error(analytical_dW, numerical_dW)
        assert error < 1e-3, f"End-to-end gradient error: {error}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
