"""
Visualization Utilities for CNN Library
========================================

This module provides functions for visualizing:
- Training progress (loss/accuracy curves)
- Feature maps from convolutional layers
- Convolutional filters
- Confusion matrix
- Sample predictions
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(history, figsize=(14, 5), save_path=None):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history['loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if history.get('val_loss'):
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if history.get('val_accuracy'):
        axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()
    return fig


def visualize_feature_maps(feature_maps, max_maps=16, figsize=(12, 12), save_path=None):
    """
    Visualize feature maps from a convolutional layer.

    Args:
        feature_maps: Feature maps, shape (1, C, H, W) or (C, H, W)
        max_maps: Maximum number of feature maps to display
        figsize: Figure size
        save_path: Path to save figure
    """
    if feature_maps.ndim == 4:
        feature_maps = feature_maps[0]  # Remove batch dimension

    n_maps = min(feature_maps.shape[0], max_maps)
    n_cols = int(np.ceil(np.sqrt(n_maps)))
    n_rows = int(np.ceil(n_maps / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n_maps):
        axes[i].imshow(feature_maps[i], cmap='viridis')
        axes[i].set_title(f'Filter {i}', fontsize=8)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Feature Maps', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature maps saved to {save_path}")

    plt.show()
    return fig


def visualize_filters(filters, max_filters=32, figsize=(12, 8), save_path=None):
    """
    Visualize convolutional filter weights.

    Args:
        filters: Filter weights, shape (out_channels, in_channels, H, W)
        max_filters: Maximum number of filters to display
        figsize: Figure size
        save_path: Path to save figure
    """
    n_filters = min(filters.shape[0], max_filters)
    n_channels = filters.shape[1]

    n_cols = int(np.ceil(np.sqrt(n_filters)))
    n_rows = int(np.ceil(n_filters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n_filters):
        # For multi-channel filters, average across input channels
        if n_channels > 1:
            filter_img = np.mean(filters[i], axis=0)
        else:
            filter_img = filters[i, 0]

        # Normalize for visualization
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)

        axes[i].imshow(filter_img, cmap='gray')
        axes[i].set_title(f'Filter {i}', fontsize=8)
        axes[i].axis('off')

    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Convolutional Filters', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Filters visualization saved to {save_path}")

    plt.show()
    return fig


def visualize_confusion_matrix(cm, class_names=None, figsize=(10, 8), save_path=None):
    """
    Visualize confusion matrix.

    Args:
        cm: Confusion matrix, shape (num_classes, num_classes)
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    return fig


def plot_sample_predictions(images, true_labels, pred_labels, class_names=None,
                           n_samples=16, figsize=(12, 12), save_path=None):
    """
    Plot sample predictions with true and predicted labels.

    Args:
        images: Images, shape (N, C, H, W)
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: Class names for labels
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: Path to save figure
    """
    n_samples = min(n_samples, len(images))
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(n_samples):
        # Prepare image for display
        img = images[i]
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = np.squeeze(img)

        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)

        true = class_names[true_labels[i]] if class_names else str(true_labels[i])
        pred = class_names[pred_labels[i]] if class_names else str(pred_labels[i])

        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        axes[i].set_title(f'True: {true}\nPred: {pred}', color=color, fontsize=9)
        axes[i].axis('off')

    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")

    plt.show()
    return fig
