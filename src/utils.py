"""
Utility Functions for CNN Library
=================================

Helper functions for:
- Data loading (MNIST)
- Preprocessing
- Batching
- Metrics
- Visualization helpers
"""

import numpy as np
import struct
from pathlib import Path


def load_mnist(data_dir='mnist_dataset', subset_size=None, normalize=True):
    """
    Load MNIST dataset from IDX binary files.

    Args:
        data_dir: Path to folder containing MNIST files
        subset_size: Tuple (train_size, test_size) for subset, None for full
        normalize: Scale pixel values to [0, 1]

    Returns:
        X_train: Training images, shape (N, 1, 28, 28)
        y_train: Training labels, shape (N,)
        X_test: Test images, shape (N, 1, 28, 28)
        y_test: Test labels, shape (N,)

    Note: Images are returned in NCHW format (batch, channels, height, width) which is standard for CNNs.
    """
    data_dir = Path(data_dir)

    # Find files
    train_images_file = _find_file(data_dir, 'train-images')
    train_labels_file = _find_file(data_dir, 'train-labels')
    test_images_file = _find_file(data_dir, 't10k-images')
    test_labels_file = _find_file(data_dir, 't10k-labels')

    # Load data
    X_train = _load_idx_images(train_images_file)
    y_train = _load_idx_labels(train_labels_file)
    X_test = _load_idx_images(test_images_file)
    y_test = _load_idx_labels(test_labels_file)

    # Take subset if requested
    if subset_size is not None:
        train_size, test_size = subset_size

        # Shuffle before taking subset
        train_indices = np.random.permutation(len(X_train))[:train_size]
        test_indices = np.random.permutation(len(X_test))[:test_size]

        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

    # Normalize to [0, 1]
    if normalize:
        X_train = X_train.astype(np.float64) / 255.0
        X_test = X_test.astype(np.float64) / 255.0

    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    print(f"Loaded MNIST: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    print(f"Image shape: {X_train.shape[1:]} (channels, height, width)")

    return X_train, y_train, X_test, y_test


def _find_file(data_dir, prefix):
    """Find IDX file with given prefix."""
    data_dir = Path(data_dir)

    # Try direct file
    for ext in ['.idx3-ubyte', '.idx1-ubyte', '-idx3-ubyte', '-idx1-ubyte', '']:
        candidate = data_dir / f"{prefix}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

    # Try subfolders (Kaggle extraction quirk)
    for item in data_dir.iterdir():
        if item.is_dir() and prefix in item.name:
            for file in item.iterdir():
                if file.is_file():
                    return file

    # Recursive search
    for file in data_dir.rglob('*'):
        if file.is_file() and prefix in file.name:
            return file

    raise FileNotFoundError(f"Could not find MNIST file with prefix '{prefix}' in {data_dir}")


def _load_idx_images(filepath):
    """Load images from IDX file."""
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} (expected 2051)")

        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)

    return images


def _load_idx_labels(filepath):
    """Load labels from IDX file."""
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} (expected 2049)")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def one_hot_encode(labels, num_classes=None):
    """
    Convert integer labels to one-hot encoded vectors.

    Args:
        labels: Integer labels, shape (N,)
        num_classes: Number of classes (inferred if None)

    Returns:
        One-hot matrix, shape (N, num_classes)
    """
    labels = np.asarray(labels).astype(int)

    if num_classes is None:
        num_classes = labels.max() + 1

    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0

    return one_hot


def create_batches(X, y, batch_size, shuffle=True):
    """
    Create mini-batches for training.

    Args:
        X: Features, shape (N, ...)
        y: Labels, shape (N,) or (N, C)
        batch_size: Batch size
        shuffle: Whether to shuffle

    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = len(X)

    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        yield X[start_idx:end_idx], y[start_idx:end_idx]


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: True labels (integers or one-hot)
        y_pred: Predictions (probabilities or one-hot)

    Returns:
        Accuracy as float
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Confusion matrix, shape (num_classes, num_classes)
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def normalize_data(X, method='minmax'):
    """
    Normalize data.

    Args:
        X: Data to normalize
        method: 'minmax' (0-1), 'zscore' (mean=0, std=1), 'l2' (unit norm)

    Returns:
        Normalized data
    """
    X = np.asarray(X, dtype=np.float64)

    if method == 'minmax':
        X_min, X_max = X.min(), X.max()
        if X_max - X_min > 0:
            return (X - X_min) / (X_max - X_min)
        return X
    elif method == 'zscore':
        mean = X.mean()
        std = X.std()
        return (X - mean) / (std + 1e-8)
    elif method == 'l2':
        norms = np.linalg.norm(X.reshape(X.shape[0], -1), axis=1, keepdims=True)
        norms = norms.reshape(X.shape[0], *([1] * (X.ndim - 1)))
        return X / (norms + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split data into train and test sets.

    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set
        shuffle: Whether to shuffle
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    print(f"Random seed set to {seed}")


def get_model_summary(layers):
    """
    Generate model summary.

    Args:
        layers: List of layer objects

    Returns:
        Summary string
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Layer':<30} {'Output Shape':<20} {'Params':<15}")
    lines.append("=" * 70)

    total_params = 0

    for i, layer in enumerate(layers):
        name = str(layer)

        # Count parameters
        n_params = 0
        if hasattr(layer, 'params'):
            for param in layer.params.values():
                n_params += param.size

        total_params += n_params

        lines.append(f"{name:<30} {'':<20} {n_params:,}")

    lines.append("=" * 70)
    lines.append(f"Total trainable parameters: {total_params:,}")
    lines.append("=" * 70)

    return '\n'.join(lines)


def augment_images(images, rotation_range=10, shift_range=0.1,
                   horizontal_flip=False, zoom_range=0.1):
    """
    Simple image augmentation.

    Args:
        images: Input images, shape (N, C, H, W)
        rotation_range: Max rotation in degrees
        shift_range: Max shift as fraction of size
        horizontal_flip: Whether to randomly flip horizontally
        zoom_range: Max zoom factor

    Returns:
        Augmented images
    """
    augmented = images.copy()
    batch_size, channels, height, width = images.shape

    for i in range(batch_size):
        # Random horizontal flip
        if horizontal_flip and np.random.rand() > 0.5:
            augmented[i] = augmented[i, :, :, ::-1]

        # Random shift (simplified - just roll)
        if shift_range > 0:
            h_shift = int(np.random.uniform(-shift_range, shift_range) * height)
            w_shift = int(np.random.uniform(-shift_range, shift_range) * width)
            augmented[i] = np.roll(augmented[i], (h_shift, w_shift), axis=(1, 2))

    return augmented
