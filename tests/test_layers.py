"""
Tests for CNN Layers
====================

Unit tests for convolutional, pooling, normalization, and dense layers.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.layers import (Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, BatchNorm1D,
                        Dense, Flatten, Dropout, Dropout2D, Activation)


class TestConv2D:
    """Tests for Conv2D layer."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding='same')
        x = np.random.randn(4, 1, 28, 28)  # batch=4, channels=1, 28x28
        output = conv.forward(x, training=False)

        assert output.shape == (4, 8, 28, 28), f"Expected (4, 8, 28, 28), got {output.shape}"

    def test_forward_valid_padding(self):
        """Test output shape with valid padding."""
        conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding='valid')
        x = np.random.randn(4, 1, 28, 28)
        output = conv.forward(x, training=False)

        # With valid padding and 3x3 kernel: 28 - 3 + 1 = 26
        assert output.shape == (4, 8, 26, 26), f"Expected (4, 8, 26, 26), got {output.shape}"

    def test_forward_stride(self):
        """Test output shape with stride > 1."""
        conv = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding='same')
        x = np.random.randn(4, 1, 28, 28)
        output = conv.forward(x, training=False)

        # With stride 2: 28 -> 14
        assert output.shape == (4, 8, 14, 14), f"Expected (4, 8, 14, 14), got {output.shape}"

    def test_backward_shapes(self):
        """Test gradient shapes are correct."""
        conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, padding='same')
        x = np.random.randn(4, 3, 16, 16)

        # Forward
        output = conv.forward(x, training=True)

        # Backward
        grad_output = np.random.randn(*output.shape)
        grad_input = conv.backward(grad_output)

        assert grad_input.shape == x.shape, f"Expected {x.shape}, got {grad_input.shape}"
        assert conv.grads['weight'].shape == conv.params['weight'].shape
        assert conv.grads['bias'].shape == conv.params['bias'].shape


class TestMaxPool2D:
    """Tests for MaxPool2D layer."""

    def test_forward_shape(self):
        """Test output shape after pooling."""
        pool = MaxPool2D(pool_size=2)
        x = np.random.randn(4, 8, 28, 28)
        output = pool.forward(x, training=False)

        assert output.shape == (4, 8, 14, 14)

    def test_max_values(self):
        """Test that max values are correctly extracted."""
        pool = MaxPool2D(pool_size=2)

        # Create input with known max values
        x = np.array([[[[1, 2], [3, 4]]]])  # 1x1x2x2
        output = pool.forward(x, training=False)

        assert output[0, 0, 0, 0] == 4, "Max value should be 4"

    def test_backward_gradient_routing(self):
        """Test that gradients are routed to max positions."""
        pool = MaxPool2D(pool_size=2)

        x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)
        _ = pool.forward(x, training=True)

        grad_output = np.array([[[[1.0]]]])
        grad_input = pool.backward(grad_output)

        # Gradient should only appear at position of max (3,4) = position (1,1)
        assert grad_input[0, 0, 1, 1] == 1.0, "Gradient should be at max position"
        assert grad_input[0, 0, 0, 0] == 0.0, "Other positions should be zero"


class TestAvgPool2D:
    """Tests for AvgPool2D layer."""

    def test_forward_shape(self):
        """Test output shape after average pooling."""
        pool = AvgPool2D(pool_size=2)
        x = np.random.randn(4, 8, 28, 28)
        output = pool.forward(x, training=False)

        assert output.shape == (4, 8, 14, 14)

    def test_average_values(self):
        """Test that average values are correctly computed."""
        pool = AvgPool2D(pool_size=2)

        x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float64)
        output = pool.forward(x, training=False)

        expected_avg = (1 + 2 + 3 + 4) / 4
        assert output[0, 0, 0, 0] == expected_avg


class TestBatchNorm2D:
    """Tests for BatchNorm2D layer."""

    def test_forward_shape(self):
        """Test output shape is preserved."""
        bn = BatchNorm2D(num_features=8)
        x = np.random.randn(4, 8, 14, 14)
        output = bn.forward(x, training=True)

        assert output.shape == x.shape

    def test_normalization(self):
        """Test that output is roughly normalized."""
        bn = BatchNorm2D(num_features=8)
        x = np.random.randn(32, 8, 14, 14) * 10 + 5  # Large mean and std
        output = bn.forward(x, training=True)

        # After normalization, each channel should have mean ≈ 0, std ≈ 1
        for c in range(8):
            channel_mean = np.mean(output[:, c, :, :])
            channel_std = np.std(output[:, c, :, :])

            assert abs(channel_mean) < 0.5, f"Mean should be near 0, got {channel_mean}"
            assert abs(channel_std - 1) < 0.5, f"Std should be near 1, got {channel_std}"

    def test_eval_mode(self):
        """Test that eval mode uses running statistics."""
        bn = BatchNorm2D(num_features=8)

        # Train to update running stats
        for _ in range(10):
            x = np.random.randn(32, 8, 14, 14)
            _ = bn.forward(x, training=True)

        # Eval mode should use running stats
        x_test = np.random.randn(4, 8, 14, 14)
        output_eval = bn.forward(x_test, training=False)

        assert output_eval.shape == x_test.shape


class TestDense:
    """Tests for Dense layer."""

    def test_forward_shape(self):
        """Test output shape."""
        dense = Dense(input_dim=784, output_dim=128)
        x = np.random.randn(32, 784)
        output = dense.forward(x, training=False)

        assert output.shape == (32, 128)

    def test_backward_shapes(self):
        """Test gradient shapes."""
        dense = Dense(input_dim=784, output_dim=128)
        x = np.random.randn(32, 784)

        output = dense.forward(x, training=True)
        grad_output = np.random.randn(*output.shape)
        grad_input = dense.backward(grad_output)

        assert grad_input.shape == x.shape
        assert dense.grads['weight'].shape == dense.params['weight'].shape


class TestFlatten:
    """Tests for Flatten layer."""

    def test_forward_shape(self):
        """Test flattening."""
        flatten = Flatten()
        x = np.random.randn(4, 64, 7, 7)
        output = flatten.forward(x)

        assert output.shape == (4, 64 * 7 * 7)

    def test_backward_shape(self):
        """Test unflattening in backward pass."""
        flatten = Flatten()
        x = np.random.randn(4, 64, 7, 7)

        _ = flatten.forward(x)
        grad_output = np.random.randn(4, 64 * 7 * 7)
        grad_input = flatten.backward(grad_output)

        assert grad_input.shape == x.shape


class TestDropout:
    """Tests for Dropout layer."""

    def test_training_mode(self):
        """Test that dropout zeros some values in training."""
        dropout = Dropout(rate=0.5)
        x = np.ones((100, 100))

        output = dropout.forward(x, training=True)

        # Some values should be zero, others scaled up
        assert np.any(output == 0), "Some values should be dropped"
        assert np.any(output > 1), "Remaining values should be scaled"

    def test_eval_mode(self):
        """Test that dropout does nothing in eval mode."""
        dropout = Dropout(rate=0.5)
        x = np.ones((100, 100))

        output = dropout.forward(x, training=False)

        np.testing.assert_array_equal(output, x, "Eval mode should not modify input")


class TestDropout2D:
    """Tests for Dropout2D layer."""

    def test_training_drops_channels(self):
        """Test that entire channels are dropped."""
        dropout = Dropout2D(rate=0.5)
        x = np.ones((4, 16, 8, 8))

        np.random.seed(42)
        output = dropout.forward(x, training=True)

        # Check that entire channels are dropped (all zeros in that channel)
        for b in range(4):
            for c in range(16):
                channel = output[b, c]
                # Channel should be either all zeros or all same value
                unique_vals = np.unique(channel)
                assert len(unique_vals) == 1, "Each channel should be uniform"


class TestActivation:
    """Tests for Activation wrapper layer."""

    def test_relu(self):
        """Test ReLU activation."""
        act = Activation('relu')
        x = np.array([-1, 0, 1])
        output = act.forward(x)

        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(output, expected)

    def test_softmax(self):
        """Test softmax activation."""
        act = Activation('softmax')
        x = np.array([[1, 2, 3]])
        output = act.forward(x)

        # Softmax should sum to 1
        assert abs(np.sum(output) - 1.0) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
