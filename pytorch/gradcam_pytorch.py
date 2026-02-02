"""
Grad-CAM Implementation for PyTorch
===================================

Uses PyTorch hooks for efficient gradient computation.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAMPyTorch:
    """
    Grad-CAM implementation using PyTorch hooks.
    
    Example:
        >>> model = CNNPyTorch()
        >>> model.load_state_dict(torch.load('model.pth'))
        >>> gradcam = GradCAMPyTorch(model, target_layer='conv4')
        >>> heatmap = gradcam.generate(image, target_class=5)
    """
    
    def __init__(self, model, target_layer='conv4'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer for CAM
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get target layer
        target = getattr(self.model, self.target_layer, None)
        
        if target is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")
        
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def generate(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image tensor, shape (1, C, H, W)
            target_class: Class to visualize (default: predicted class)
        
        Returns:
            heatmap: Numpy array, shape (H, W), values in [0, 1]
        """
        self.model.eval()
        
        # Ensure image requires grad
        image = image.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, h, w)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()
        
        return heatmap
    
    def generate_batch(self, images, target_classes=None):
        """
        Generate Grad-CAM for batch of images.
        
        Args:
            images: Batch of images, shape (N, C, H, W)
            target_classes: List of target classes
        
        Returns:
            List of heatmaps
        """
        heatmaps = []
        
        for i in range(images.shape[0]):
            image = images[i:i+1]
            target = target_classes[i] if target_classes else None
            heatmap = self.generate(image, target)
            heatmaps.append(heatmap)
        
        return heatmaps
    
    def overlay(self, image, heatmap, alpha=0.5):
        """Overlay heatmap on image."""
        from scipy.ndimage import zoom
        
        # Convert image to numpy
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        
        # Remove batch dimension if present
        while image.ndim > 3:
            image = image[0]
        
        # Convert CHW to HWC
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        # Squeeze single channel
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]
        
        # Get target size
        h, w = image.shape[:2]
        
        # Resize heatmap to match image
        heatmap = np.asarray(heatmap)
        if heatmap.shape[0] != h or heatmap.shape[1] != w:
            scale_h = h / heatmap.shape[0]
            scale_w = w / heatmap.shape[1]
            heatmap = zoom(heatmap, (scale_h, scale_w), order=1)
        
        # Apply colormap
        heatmap_colored = self._apply_colormap(heatmap)
        
        # Convert grayscale image to RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize image to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Blend
        overlay = (alpha * heatmap_colored.astype(float) + (1 - alpha) * image.astype(float)).astype(np.uint8)
        
        return overlay
    
    def _apply_colormap(self, heatmap):
        """Apply jet colormap to heatmap."""
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                v = float(heatmap[i, j])
                if v < 0.25:
                    r, g, b = 0, v * 4, 1
                elif v < 0.5:
                    r, g, b = 0, 1, 1 - (v - 0.25) * 4
                elif v < 0.75:
                    r, g, b = (v - 0.5) * 4, 1, 0
                else:
                    r, g, b = 1, 1 - (v - 0.75) * 4, 0
                colored[i, j] = [int(r * 255), int(g * 255), int(b * 255)]
        
        return colored
    
    def visualize(self, image, heatmap=None, target_class=None, 
                  figsize=(12, 4), save_path=None):
        """Visualize Grad-CAM result."""
        if heatmap is None:
            heatmap = self.generate(image, target_class)
        
        # Prepare image for display
        if torch.is_tensor(image):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.asarray(image)
        
        # Remove batch dim
        while image_np.ndim > 3:
            image_np = image_np[0]
        
        # CHW to HWC
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Squeeze grayscale
        if image_np.ndim == 3 and image_np.shape[-1] == 1:
            image_np = image_np[:, :, 0]
        
        # Create overlay
        overlay = self.overlay(image, heatmap)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original
        if image_np.ndim == 2:
            axes[0].imshow(image_np, cmap='gray')
        else:
            axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return fig


class GradCAMPlusPlus(GradCAMPyTorch):
    """
    Grad-CAM++ implementation.
    
    Improvement over Grad-CAM that provides better localization
    for multiple instances of the same class.
    """
    
    def generate(self, image, target_class=None):
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        
        image = image.clone().requires_grad_(True)
        
        output = self.model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        # Get score for target class
        score = output[0, target_class]
        
        # Compute first derivative
        score.backward(retain_graph=True)
        gradients = self.gradients.clone()
        
        # Compute second derivative
        grad_2 = gradients ** 2
        
        # Compute third derivative (approximation)
        grad_3 = gradients ** 3
        
        # Compute weights using Grad-CAM++ formula
        activations = self.activations
        
        # Global sum of activations
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        
        # Alpha calculation
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Weights
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


