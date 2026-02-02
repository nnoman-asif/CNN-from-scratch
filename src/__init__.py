"""
CNN Library from Scratch
========================

A complete implementation of Convolutional Neural Networks using only NumPy.
This library demonstrates the mechanics of CNNs including:
- 2D Convolution
- Pooling operations (Max, Average)
- Batch Normalization
- Dropout and Spatial Dropout
- Forward and backward propagation
- Adam and SGD optimizers
"""

from .activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, get_activation
from .layers import Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, BatchNorm1D
from .layers import Dense, Flatten, Dropout, Dropout2D
from .losses import CrossEntropyLoss, MSELoss, get_loss
from .optimizers import Adam, SGD
from .cnn import CNN
from .utils import load_mnist, one_hot_encode, create_batches
from . import visualizations

__version__ = "1.0.0"
__all__ = [
    # Activations
    'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'Softmax', 'get_activation',
    # Layers
    'Conv2D', 'MaxPool2D', 'AvgPool2D', 'BatchNorm2D', 'BatchNorm1D',
    'Dense', 'Flatten', 'Dropout', 'Dropout2D',
    # Losses
    'CrossEntropyLoss', 'MSELoss', 'get_loss',
    # Optimizers
    'Adam', 'SGD',
    # Main class
    'CNN',
    # Utilities
    'load_mnist', 'one_hot_encode', 'create_batches',
]
