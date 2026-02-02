"""
Optimizers for CNNs
===================

Optimizers update network weights based on computed gradients.
The choice of optimizer significantly affects training speed and convergence.

This module implements:
- Adam (default): Adaptive learning rates, works well out of the box
- SGD with Momentum: Classic, often preferred for final fine-tuning
"""

import numpy as np


class Optimizer:
    """Base class for optimizers."""

    def step(self, layers):
        """Update weights for all layers."""
        raise NotImplementedError

    def get_lr(self):
        """Get current learning rate."""
        raise NotImplementedError


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines the benefits of:
    - Momentum: Uses running average of gradients
    - RMSprop: Uses running average of squared gradients

    Advantages:
    - Works well with default hyperparameters
    - Handles sparse gradients
    - Adapts learning rate per parameter

    Args:
        learning_rate: Step size (default: 0.001)
        beta1: Decay rate for first moment (default: 0.9)
        beta2: Decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 regularization strength (default: 0)
        clip_grad: Max gradient norm for clipping (default: None)
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0, clip_grad=None):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad

        self.t = 0  # Time step for bias correction
        self.lr_scheduler = None

        # Per-layer momentum and velocity caches
        self._initialized = False
        self._cache = {}

    def _init_cache(self, layers):
        """Initialize momentum and velocity caches for each layer."""
        self._cache = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params'):
                layer_cache = {}
                for name, param in layer.params.items():
                    layer_cache[f'm_{name}'] = np.zeros_like(param)
                    layer_cache[f'v_{name}'] = np.zeros_like(param)
                self._cache[i] = layer_cache
        self._initialized = True

    def set_lr_scheduler(self, scheduler):
        """Attach a learning rate scheduler."""
        self.lr_scheduler = scheduler

    def step(self, layers):
        """
        Update weights using Adam algorithm.

        Args:
            layers: List of layer objects with computed gradients
        """
        if not self._initialized:
            self._init_cache(layers)

        self.t += 1

        # Update learning rate if scheduler is set
        if self.lr_scheduler is not None:
            self.learning_rate = self.lr_scheduler(self.t, self.initial_lr)

        for i, layer in enumerate(layers):
            if not hasattr(layer, 'params') or not hasattr(layer, 'grads'):
                continue

            if layer.grads is None:
                continue

            for name, param in layer.params.items():
                if name not in layer.grads or layer.grads[name] is None:
                    continue

                grad = layer.grads[name].copy()

                # Gradient clipping
                if self.clip_grad is not None:
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > self.clip_grad:
                        grad = grad * (self.clip_grad / (grad_norm + 1e-8))

                # Weight decay (L2 regularization)
                if self.weight_decay > 0 and 'bias' not in name:
                    grad = grad + self.weight_decay * param

                # Get cached momentum and velocity
                cache = self._cache.get(i, {})
                m = cache.get(f'm_{name}', np.zeros_like(param))
                v = cache.get(f'v_{name}', np.zeros_like(param))

                # Update biased first moment estimate
                m = self.beta1 * m + (1 - self.beta1) * grad

                # Update biased second raw moment estimate
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

                # Bias-corrected estimates
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                # Update parameters
                layer.params[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                # Store updated cache
                if i not in self._cache:
                    self._cache[i] = {}
                self._cache[i][f'm_{name}'] = m
                self._cache[i][f'v_{name}'] = v

    def get_lr(self):
        return self.learning_rate

    def reset(self):
        """Reset optimizer state."""
        self.t = 0
        self._initialized = False
        self._cache = {}


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.

    Despite newer optimizers, SGD with momentum remains competitive
    and is often preferred for its simplicity and generalization.

    Args:
        learning_rate: Step size (default: 0.01)
        momentum: Momentum factor (default: 0.9)
        weight_decay: L2 regularization (default: 0)
        nesterov: Use Nesterov momentum (default: False)
        clip_grad: Max gradient norm (default: None)
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0,
                 nesterov=False, clip_grad=None):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.clip_grad = clip_grad

        self.t = 0
        self.lr_scheduler = None

        self._initialized = False
        self._velocity = {}

    def _init_velocity(self, layers):
        """Initialize velocity cache for momentum."""
        self._velocity = {}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'params'):
                self._velocity[i] = {
                    name: np.zeros_like(param)
                    for name, param in layer.params.items()
                }
        self._initialized = True

    def set_lr_scheduler(self, scheduler):
        """Attach a learning rate scheduler."""
        self.lr_scheduler = scheduler

    def step(self, layers):
        """
        Update weights using SGD with momentum.

        Args:
            layers: List of layer objects with computed gradients
        """
        if not self._initialized:
            self._init_velocity(layers)

        self.t += 1

        # Update learning rate if scheduler is set
        if self.lr_scheduler is not None:
            self.learning_rate = self.lr_scheduler(self.t, self.initial_lr)

        for i, layer in enumerate(layers):
            if not hasattr(layer, 'params') or not hasattr(layer, 'grads'):
                continue

            if layer.grads is None:
                continue

            for name, param in layer.params.items():
                if name not in layer.grads or layer.grads[name] is None:
                    continue

                grad = layer.grads[name].copy()

                # Gradient clipping
                if self.clip_grad is not None:
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > self.clip_grad:
                        grad = grad * (self.clip_grad / (grad_norm + 1e-8))

                # Weight decay
                if self.weight_decay > 0 and 'bias' not in name:
                    grad = grad + self.weight_decay * param

                # Get velocity
                v = self._velocity.get(i, {}).get(name, np.zeros_like(param))

                # Update velocity
                v = self.momentum * v + grad

                # Update parameters
                if self.nesterov:
                    # Nesterov: look ahead
                    layer.params[name] = param - self.learning_rate * (self.momentum * v + grad)
                else:
                    layer.params[name] = param - self.learning_rate * v

                # Store velocity
                if i not in self._velocity:
                    self._velocity[i] = {}
                self._velocity[i][name] = v

    def get_lr(self):
        return self.learning_rate

    def reset(self):
        """Reset optimizer state."""
        self.t = 0
        self._initialized = False
        self._velocity = {}


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

def step_decay(drop_rate=0.5, drop_every=10):
    """
    Step decay: LR = initial_lr * drop_rate^(step // drop_every)

    Example: drop by 0.5 every 10 epochs
    """
    def scheduler(step, initial_lr):
        return initial_lr * (drop_rate ** (step // drop_every))
    return scheduler


def exponential_decay(decay_rate=0.95):
    """
    Exponential decay: LR = initial_lr * decay_rate^step
    """
    def scheduler(step, initial_lr):
        return initial_lr * (decay_rate ** step)
    return scheduler


def cosine_annealing(total_steps, min_lr=0.0):
    """
    Cosine annealing: Smooth decay following cosine curve.

    Popular in modern training. LR decreases slowly at first,
    faster in middle, then slowly again at end.
    """
    def scheduler(step, initial_lr):
        progress = min(step / total_steps, 1.0)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
    return scheduler


def warmup_cosine(warmup_steps, total_steps, min_lr=0.0):
    """
    Linear warmup followed by cosine decay.

    Helps stabilize training at the start when gradients might be noisy.
    """
    def scheduler(step, initial_lr):
        if step < warmup_steps:
            return initial_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(progress, 1.0)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
    return scheduler


def constant_lr():
    """No decay - constant learning rate."""
    def scheduler(step, initial_lr):
        return initial_lr
    return scheduler


# Optimizer registry
OPTIMIZERS = {
    'adam': Adam,
    'sgd': SGD,
}


def get_optimizer(name, **kwargs):
    """
    Get optimizer by name.

    Args:
        name: 'adam' or 'sgd'
        **kwargs: Arguments to pass to optimizer

    Returns:
        Optimizer instance
    """
    if isinstance(name, Optimizer):
        return name

    name_lower = name.lower()
    if name_lower not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS.keys())}")

    return OPTIMIZERS[name_lower](**kwargs)


# Learning rate scheduler registry
LR_SCHEDULERS = {
    'step': step_decay,
    'exponential': exponential_decay,
    'cosine': cosine_annealing,
    'warmup_cosine': warmup_cosine,
    'constant': constant_lr,
}
