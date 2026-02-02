"""
CNN Layers - From Scratch Implementation
=========================================

This module contains the core building blocks of CNNs implemented using only NumPy.
Each layer implements forward and backward passes for backpropagation.

Layers implemented:
- Conv2D: 2D Convolution
- MaxPool2D: Max pooling for spatial downsampling
- AvgPool2D: Average pooling
- BatchNorm2D: Batch normalization for convolutional layers
- BatchNorm1D: Batch normalization for dense layers
- Dense: Fully connected layer
- Flatten: Reshape 4D to 2D for dense layers
- Dropout: Regularization by randomly zeroing activations
- Dropout2D: Spatial dropout for conv layers
"""

import numpy as np
from .activations import get_activation


class Layer:
    """Base class for all layers."""

    def __init__(self):
        self.params = {}    # Trainable parameters
        self.grads = {}     # Gradients of parameters
        self.training = True

    def forward(self, x, training=True):
        """Forward pass."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Backward pass."""
        raise NotImplementedError

    def __call__(self, x, training=True):
        return self.forward(x, training)

    def set_training(self, mode):
        """Set training mode."""
        self.training = mode


class Conv2D(Layer):
    """
    2D Convolutional Layer.

    Performs spatial convolution over input. This is the core operation of CNNs,
    enabling the network to learn local spatial features.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (number of filters)
        kernel_size: Size of convolutional kernel (int or tuple)
        stride: Stride of convolution (default: 1)
        padding: 'valid' (no padding) or 'same' (pad to keep size) or int
        use_bias: Whether to use bias (default: True)
        weight_init: 'he' or 'xavier' (default: 'he')

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, out_channels, out_height, out_width)

    Where:
        out_height = (height + 2*pad - kernel_size) // stride + 1
        out_width = (width + 2*pad - kernel_size) // stride + 1

    The backward pass computes:
    1. dL/dW: Gradient w.r.t. weights (for weight update)
    2. dL/db: Gradient w.r.t. bias
    3. dL/dX: Gradient w.r.t. input (to pass to previous layer)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='valid', use_bias=True, weight_init='he'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding_mode = padding
        self.use_bias = use_bias

        # Calculate padding
        if padding == 'same':
            # Pad to keep same spatial dimensions (assuming stride=1)
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        elif padding == 'valid':
            self.padding = (0, 0)
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Initialize weights using He initialization (good for ReLU)
        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw

        if weight_init == 'he':
            scale = np.sqrt(2.0 / fan_in)
        else:  # xavier
            scale = np.sqrt(1.0 / fan_in)

        # Weights shape: (out_channels, in_channels, kernel_height, kernel_width)
        self.params['weight'] = np.random.randn(out_channels, in_channels, kh, kw) * scale

        if use_bias:
            self.params['bias'] = np.zeros(out_channels)

        # Cache for backward pass
        self.cache = {}

    def _pad_input(self, x):
        """Apply padding to input."""
        if self.padding == (0, 0):
            return x

        ph, pw = self.padding
        return np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')

    def _im2col(self, x_padded, kh, kw, sh, sw, h_out, w_out):
        """
        Convert image patches to columns for efficient convolution.

        Uses numpy stride tricks to create a view (no memory copy) of all patches
        that would be convolved with the kernel, then reshapes for matrix multiply.

        Returns:
            col: Shape (batch_size * h_out * w_out, in_channels * kh * kw)
        """
        batch_size, C, H, W = x_padded.shape

        # Use stride tricks to create view of all patches
        shape = (batch_size, C, kh, kw, h_out, w_out)
        strides = (
            x_padded.strides[0],      # batch stride
            x_padded.strides[1],      # channel stride
            x_padded.strides[2],      # kernel height stride
            x_padded.strides[3],      # kernel width stride
            x_padded.strides[2] * sh, # output height stride (strided)
            x_padded.strides[3] * sw  # output width stride (strided)
        )

        # Create view of patches without copying
        patches = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

        # Reshape: (batch, C, kh, kw, h_out, w_out) -> (batch * h_out * w_out, C * kh * kw)
        col = patches.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * h_out * w_out, -1)

        return col

    def _col2im(self, col, x_padded_shape, kh, kw, sh, sw, h_out, w_out):
        """
        Convert columns back to image format (inverse of im2col).

        Used in backward pass to compute gradient w.r.t. input.
        """
        batch_size, C, H, W = x_padded_shape

        # Reshape col back: (batch * h_out * w_out, C * kh * kw) -> (batch, h_out, w_out, C, kh, kw)
        col_reshaped = col.reshape(batch_size, h_out, w_out, C, kh, kw)

        # Initialize output
        x_padded = np.zeros(x_padded_shape, dtype=col.dtype)

        # Accumulate values back to original positions
        # This needs to handle overlapping patches correctly
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                x_padded[:, :, h_start:h_start+kh, w_start:w_start+kw] += col_reshaped[:, i, j]

        return x_padded

    def forward(self, x, training=True):
        """
        Forward pass using im2col for efficient matrix multiplication.

        This vectorized implementation is 10-50x faster than loop-based.

        Args:
            x: Input tensor, shape (batch, in_channels, height, width)
            training: Whether in training mode

        Returns:
            Output tensor, shape (batch, out_channels, out_height, out_width)
        """
        self.training = training

        # Pad input
        x_padded = self._pad_input(x)

        batch_size, _, h_in, w_in = x_padded.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        # Calculate output dimensions
        h_out = (h_in - kh) // sh + 1
        w_out = (w_in - kw) // sw + 1

        # im2col: convert patches to columns
        # Shape: (batch * h_out * w_out, in_channels * kh * kw)
        col = self._im2col(x_padded, kh, kw, sh, sw, h_out, w_out)

        # Reshape weights: (out_channels, in_channels, kh, kw) -> (out_channels, in_channels * kh * kw)
        W_col = self.params['weight'].reshape(self.out_channels, -1)

        # Single matrix multiplication! Much faster than nested loops
        # (batch * h_out * w_out, in_channels * kh * kw) @ (in_channels * kh * kw, out_channels)
        # = (batch * h_out * w_out, out_channels)
        output = col @ W_col.T

        # Reshape to (batch, h_out, w_out, out_channels) then transpose to (batch, out_channels, h_out, w_out)
        output = output.reshape(batch_size, h_out, w_out, self.out_channels)
        output = output.transpose(0, 3, 1, 2)

        # Add bias
        if self.use_bias:
            output = output + self.params['bias'].reshape(1, -1, 1, 1)

        # Cache for backward pass
        self.cache['x'] = x
        self.cache['x_padded'] = x_padded
        self.cache['col'] = col
        self.cache['h_out'] = h_out
        self.cache['w_out'] = w_out

        return output

    def backward(self, grad_output):
        """
        Backward pass using im2col for efficient computation.

        Args:
            grad_output: Gradient from next layer, shape (batch, out_channels, h_out, w_out)

        Returns:
            grad_input: Gradient to pass to previous layer
        """
        x = self.cache['x']
        x_padded = self.cache['x_padded']
        col = self.cache['col']
        h_out = self.cache['h_out']
        w_out = self.cache['w_out']

        batch_size = x.shape[0]
        kh, kw = self.kernel_size
        sh, sw = self.stride

        W = self.params['weight']
        W_col = W.reshape(self.out_channels, -1)

        # Reshape grad_output: (batch, out_channels, h_out, w_out) -> (batch * h_out * w_out, out_channels)
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # Gradient w.r.t. weights: col.T @ grad_output
        # (in_channels * kh * kw, batch * h_out * w_out) @ (batch * h_out * w_out, out_channels)
        # = (in_channels * kh * kw, out_channels)
        dW_col = col.T @ grad_output_reshaped
        self.grads['weight'] = dW_col.T.reshape(W.shape)

        # Gradient w.r.t. bias
        if self.use_bias:
            self.grads['bias'] = np.sum(grad_output, axis=(0, 2, 3))

        # Gradient w.r.t. input: grad_output @ W
        # (batch * h_out * w_out, out_channels) @ (out_channels, in_channels * kh * kw)
        # = (batch * h_out * w_out, in_channels * kh * kw)
        dcol = grad_output_reshaped @ W_col

        # col2im: convert columns back to image
        dx_padded = self._col2im(dcol, x_padded.shape, kh, kw, sh, sw, h_out, w_out)

        # Remove padding from gradient
        if self.padding != (0, 0):
            ph, pw = self.padding
            dx = dx_padded[:, :, ph:-ph, pw:-pw]
        else:
            dx = dx_padded

        return dx

    def __repr__(self):
        return (f"Conv2D({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding_mode})")


class MaxPool2D(Layer):
    """
    Max Pooling Layer (Vectorized).

    Downsamples by taking maximum value in each window.
    Reduces spatial dimensions while keeping strongest activations.

    Args:
        pool_size: Size of pooling window (int or tuple)
        stride: Stride (default: same as pool_size)

    Backprop: Gradient flows only to the max element in each window.
    """

    def __init__(self, pool_size=2, stride=None):
        super().__init__()

        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        self.cache = {}

    def forward(self, x, training=True):
        """
        Forward pass: max pooling using vectorized operations.

        Args:
            x: Input, shape (batch, channels, height, width)

        Returns:
            Output, shape (batch, channels, h_out, w_out)
        """
        self.training = training

        batch_size, channels, h_in, w_in = x.shape
        ph, pw = self.pool_size
        sh, sw = self.stride

        h_out = (h_in - ph) // sh + 1
        w_out = (w_in - pw) // sw + 1

        # Use stride tricks to extract all windows at once
        shape = (batch_size, channels, h_out, w_out, ph, pw)
        strides = (
            x.strides[0],           # batch
            x.strides[1],           # channel
            x.strides[2] * sh,      # output height (strided)
            x.strides[3] * sw,      # output width (strided)
            x.strides[2],           # pool height
            x.strides[3]            # pool width
        )

        # Create view of all pooling windows
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # Reshape for easier max computation: (batch, channels, h_out, w_out, ph*pw)
        windows_flat = windows.reshape(batch_size, channels, h_out, w_out, -1)

        # Get max values and indices
        output = np.max(windows_flat, axis=-1)
        max_indices = np.argmax(windows_flat, axis=-1)

        # Cache for backward pass
        self.cache['x_shape'] = x.shape
        self.cache['max_indices'] = max_indices
        self.cache['windows_shape'] = (h_out, w_out, ph, pw)

        return output

    def backward(self, grad_output):
        """
        Backward pass: route gradient to max positions only.

        Uses vectorized operations for speed.
        """
        x_shape = self.cache['x_shape']
        max_indices = self.cache['max_indices']
        h_out, w_out, ph, pw = self.cache['windows_shape']
        sh, sw = self.stride

        batch_size, channels, h_in, w_in = x_shape
        grad_input = np.zeros(x_shape, dtype=grad_output.dtype)

        # Convert flat indices to (h, w) offsets within pool window
        max_h = max_indices // pw
        max_w = max_indices % pw

        # Compute absolute positions in input
        # Create output position grids
        h_grid = np.arange(h_out).reshape(1, 1, h_out, 1) * sh
        w_grid = np.arange(w_out).reshape(1, 1, 1, w_out) * sw

        # Absolute positions: (batch, channels, h_out, w_out)
        abs_h = h_grid + max_h
        abs_w = w_grid + max_w

        # Use advanced indexing to scatter gradients
        # Create batch and channel indices
        b_idx = np.arange(batch_size).reshape(batch_size, 1, 1, 1)
        c_idx = np.arange(channels).reshape(1, channels, 1, 1)

        # Broadcast to match shape
        b_idx = np.broadcast_to(b_idx, (batch_size, channels, h_out, w_out))
        c_idx = np.broadcast_to(c_idx, (batch_size, channels, h_out, w_out))

        # Scatter gradients to max positions
        np.add.at(grad_input, (b_idx, c_idx, abs_h, abs_w), grad_output)

        return grad_input

    def __repr__(self):
        return f"MaxPool2D(pool_size={self.pool_size}, stride={self.stride})"


class AvgPool2D(Layer):
    """
    Average Pooling Layer (Vectorized).

    Downsamples by taking average value in each window.
    Smoother than max pooling, sometimes used in later layers.

    Backprop: Gradient is distributed equally to all elements in window.
    """

    def __init__(self, pool_size=2, stride=None):
        super().__init__()

        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        self.cache = {}

    def forward(self, x, training=True):
        """Average pooling forward pass using vectorized operations."""
        self.training = training

        batch_size, channels, h_in, w_in = x.shape
        ph, pw = self.pool_size
        sh, sw = self.stride

        h_out = (h_in - ph) // sh + 1
        w_out = (w_in - pw) // sw + 1

        # Use stride tricks to extract all windows at once
        shape = (batch_size, channels, h_out, w_out, ph, pw)
        strides = (
            x.strides[0],           # batch
            x.strides[1],           # channel
            x.strides[2] * sh,      # output height (strided)
            x.strides[3] * sw,      # output width (strided)
            x.strides[2],           # pool height
            x.strides[3]            # pool width
        )

        # Create view of all pooling windows
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # Compute mean over last two dimensions (the pool window)
        output = np.mean(windows, axis=(4, 5))

        self.cache['x_shape'] = x.shape

        return output

    def backward(self, grad_output):
        """Distribute gradient equally to all positions in each window."""
        x_shape = self.cache['x_shape']
        batch_size, channels, h_in, w_in = x_shape
        _, _, h_out, w_out = grad_output.shape

        ph, pw = self.pool_size
        sh, sw = self.stride
        pool_size = ph * pw

        grad_input = np.zeros(x_shape, dtype=grad_output.dtype)

        # Distribute gradients - need to handle overlapping windows
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                # Add distributed gradient
                grad_input[:, :, h_start:h_start+ph, w_start:w_start+pw] += \
                    grad_output[:, :, i:i+1, j:j+1] / pool_size

        return grad_input

    def __repr__(self):
        return f"AvgPool2D(pool_size={self.pool_size}, stride={self.stride})"


class BatchNorm2D(Layer):
    """
    Batch Normalization for Convolutional Layers.

    Normalizes activations across the batch for each channel.
    This stabilizes training and acts as a regularizer.

    Args:
        num_features: Number of channels
        momentum: Momentum for running statistics (default: 0.1)
        epsilon: Small constant for numerical stability

    Forward pass:
        x_norm = (x - mean) / sqrt(var + eps)
        output = gamma * x_norm + beta

    During training: use batch statistics
    During inference: use running statistics

    Learnable parameters:
        gamma (scale): initialized to 1
        beta (shift): initialized to 0
    """

    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters
        self.params['gamma'] = np.ones(num_features)
        self.params['beta'] = np.zeros(num_features)

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.cache = {}

    def forward(self, x, training=True):
        """
        Forward pass with batch normalization.

        Args:
            x: Input, shape (batch, channels, height, width)
            training: Whether in training mode

        Returns:
            Normalized output, same shape as input
        """
        self.training = training

        gamma = self.params['gamma']
        beta = self.params['beta']

        if training:
            # Compute batch statistics (mean/var over batch and spatial dims)
            # Shape: (channels,)
            mean = np.mean(x, axis=(0, 2, 3))
            var = np.var(x, axis=(0, 2, 3))

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.epsilon)

        # Scale and shift
        output = gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)

        # Cache for backward pass
        self.cache['x'] = x
        self.cache['x_norm'] = x_norm
        self.cache['mean'] = mean
        self.cache['var'] = var

        return output

    def backward(self, grad_output):
        """
        Backward pass through batch normalization.

        This is tricky because the normalization couples all samples in the batch.
        """
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        gamma = self.params['gamma']

        batch_size, channels, height, width = x.shape
        N = batch_size * height * width  # Total elements per channel

        # Gradients w.r.t. gamma and beta
        # Sum over batch and spatial dimensions
        self.grads['gamma'] = np.sum(grad_output * x_norm, axis=(0, 2, 3))
        self.grads['beta'] = np.sum(grad_output, axis=(0, 2, 3))

        # Gradient w.r.t. x_norm
        dx_norm = grad_output * gamma.reshape(1, -1, 1, 1)

        # Gradient w.r.t. variance
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        x_centered = x - mean.reshape(1, -1, 1, 1)
        dvar = np.sum(dx_norm * x_centered * -0.5 * (std_inv ** 3).reshape(1, -1, 1, 1), axis=(0, 2, 3))

        # Gradient w.r.t. mean
        dmean = np.sum(dx_norm * -std_inv.reshape(1, -1, 1, 1), axis=(0, 2, 3))
        dmean += dvar * np.mean(-2 * x_centered, axis=(0, 2, 3))

        # Gradient w.r.t. input
        grad_input = dx_norm * std_inv.reshape(1, -1, 1, 1)
        grad_input += dvar.reshape(1, -1, 1, 1) * 2 * x_centered / N
        grad_input += dmean.reshape(1, -1, 1, 1) / N

        return grad_input

    def __repr__(self):
        return f"BatchNorm2D({self.num_features})"


class BatchNorm1D(Layer):
    """
    Batch Normalization for Dense Layers.

    Same concept as BatchNorm2D but for 2D inputs (batch, features).
    """

    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.params['gamma'] = np.ones(num_features)
        self.params['beta'] = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.cache = {}

    def forward(self, x, training=True):
        """Forward pass for 1D batch normalization."""
        self.training = training

        gamma = self.params['gamma']
        beta = self.params['beta']

        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        output = gamma * x_norm + beta

        self.cache['x'] = x
        self.cache['x_norm'] = x_norm
        self.cache['mean'] = mean
        self.cache['var'] = var

        return output

    def backward(self, grad_output):
        """Backward pass for 1D batch normalization."""
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        mean = self.cache['mean']
        var = self.cache['var']
        gamma = self.params['gamma']

        batch_size = x.shape[0]

        self.grads['gamma'] = np.sum(grad_output * x_norm, axis=0)
        self.grads['beta'] = np.sum(grad_output, axis=0)

        dx_norm = grad_output * gamma

        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        x_centered = x - mean

        dvar = np.sum(dx_norm * x_centered * -0.5 * (std_inv ** 3), axis=0)
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2 * x_centered, axis=0)

        grad_input = dx_norm * std_inv + dvar * 2 * x_centered / batch_size + dmean / batch_size

        return grad_input

    def __repr__(self):
        return f"BatchNorm1D({self.num_features})"


class Flatten(Layer):
    """
    Flatten layer: reshapes 4D tensor to 2D.

    Input: (batch, channels, height, width)
    Output: (batch, channels * height * width)

    Used to connect convolutional layers to dense layers.
    """

    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, x, training=True):
        """Flatten input while preserving batch dimension."""
        self.cache['input_shape'] = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output):
        """Reshape gradient back to original shape."""
        return grad_output.reshape(self.cache['input_shape'])

    def __repr__(self):
        return "Flatten()"


class Dense(Layer):
    """
    Fully Connected (Dense) Layer.

    Each output is connected to every input.

    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        use_bias: Whether to use bias (default: True)
        weight_init: 'he' or 'xavier'

    Forward: output = input @ W + b
    """

    def __init__(self, input_dim, output_dim, use_bias=True, weight_init='he'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # Weight initialization
        if weight_init == 'he':
            scale = np.sqrt(2.0 / input_dim)
        else:
            scale = np.sqrt(1.0 / input_dim)

        self.params['weight'] = np.random.randn(input_dim, output_dim) * scale

        if use_bias:
            self.params['bias'] = np.zeros(output_dim)

        self.cache = {}

    def forward(self, x, training=True):
        """Forward pass: y = x @ W + b"""
        self.training = training
        self.cache['x'] = x

        output = x @ self.params['weight']
        if self.use_bias:
            output += self.params['bias']

        return output

    def backward(self, grad_output):
        """
        Backward pass.

        dL/dW = x.T @ grad_output
        dL/db = sum(grad_output)
        dL/dx = grad_output @ W.T
        """
        x = self.cache['x']

        self.grads['weight'] = x.T @ grad_output

        if self.use_bias:
            self.grads['bias'] = np.sum(grad_output, axis=0)

        grad_input = grad_output @ self.params['weight'].T

        return grad_input

    def __repr__(self):
        return f"Dense({self.input_dim}, {self.output_dim})"


class Dropout(Layer):
    """
    Dropout Layer for regularization.

    Randomly sets activations to zero during training.
    Uses "inverted dropout": scales remaining activations so we don't
    need to scale at test time.

    Args:
        rate: Fraction of activations to drop (default: 0.5)
    """

    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.cache = {}

    def forward(self, x, training=True):
        """Apply dropout during training."""
        self.training = training

        if training and self.rate > 0:
            # Create mask: 1 = keep, 0 = drop
            self.cache['mask'] = (np.random.rand(*x.shape) > self.rate).astype(np.float64)
            # Inverted dropout: scale by 1/(1-rate)
            return x * self.cache['mask'] / (1 - self.rate)
        else:
            return x

    def backward(self, grad_output):
        """Route gradient through non-dropped positions only."""
        if self.training and self.rate > 0:
            return grad_output * self.cache['mask'] / (1 - self.rate)
        else:
            return grad_output

    def __repr__(self):
        return f"Dropout(rate={self.rate})"


class Dropout2D(Layer):
    """
    Spatial Dropout for Convolutional Layers.

    Drops entire feature maps (channels) instead of individual elements.
    More effective for conv layers because adjacent pixels are correlated.

    Args:
        rate: Fraction of channels to drop (default: 0.25)
    """

    def __init__(self, rate=0.25):
        super().__init__()
        self.rate = rate
        self.cache = {}

    def forward(self, x, training=True):
        """Apply spatial dropout during training."""
        self.training = training

        if training and self.rate > 0:
            batch_size, channels, height, width = x.shape
            # Mask entire channels
            mask_shape = (batch_size, channels, 1, 1)
            self.cache['mask'] = (np.random.rand(*mask_shape) > self.rate).astype(np.float64)
            return x * self.cache['mask'] / (1 - self.rate)
        else:
            return x

    def backward(self, grad_output):
        """Route gradient through non-dropped channels."""
        if self.training and self.rate > 0:
            return grad_output * self.cache['mask'] / (1 - self.rate)
        else:
            return grad_output

    def __repr__(self):
        return f"Dropout2D(rate={self.rate})"


class Activation(Layer):
    """
    Activation layer wrapper.

    Wraps activation functions as layers for use in sequential models.
    """

    def __init__(self, activation='relu'):
        super().__init__()
        self.activation = get_activation(activation)
        self.activation_name = activation
        self.cache = {}

    def forward(self, x, training=True):
        """Apply activation function."""
        self.cache['x'] = x
        return self.activation.forward(x)

    def backward(self, grad_output):
        """Multiply by activation derivative."""
        x = self.cache['x']

        # Special case for softmax
        if self.activation_name == 'softmax':
            # For softmax with cross-entropy, gradient is already computed
            return grad_output

        return grad_output * self.activation.backward(x)

    def __repr__(self):
        return f"Activation({self.activation_name})"
