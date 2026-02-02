"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
=====================================================

Simplified, robust implementation that works reliably with our CNN.
"""

import numpy as np
from .layers import Conv2D


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.
    """
    
    def __init__(self, model, target_layer_index=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained CNN model
            target_layer_index: Index of conv layer to use (default: last conv layer)
        """
        self.model = model
        
        # Find conv layer indices
        self.conv_indices = [i for i, layer in enumerate(model.layers) 
                            if isinstance(layer, Conv2D)]
        
        if not self.conv_indices:
            raise ValueError("Model has no convolutional layers")
        
        # Use last conv layer by default
        if target_layer_index is None:
            self.target_layer_index = self.conv_indices[-1]
        else:
            self.target_layer_index = target_layer_index
    
    def generate(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap using numerical gradients.
        
        This is a robust implementation that computes gradients numerically
        to avoid issues with layer backward passes.
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        
        # Step 1: Forward pass to get target layer activations
        x = image.copy()
        target_activations = None
        
        for i, layer in enumerate(self.model.layers):
            x = layer.forward(x, training=False)
            if i == self.target_layer_index:
                target_activations = x.copy()
        
        predictions = x
        
        # Determine target class
        if target_class is None:
            target_class = np.argmax(predictions[0])
        
        # Step 2: Compute importance weights using gradient approximation
        # For each channel, compute how much it contributes to the target class score
        
        # Get the pre-softmax score for target class
        # We need to find the logits (before softmax)
        def get_target_score(activations_at_target):
            """Forward from target layer to get target class score."""
            x = activations_at_target
            for i in range(self.target_layer_index + 1, len(self.model.layers)):
                layer = self.model.layers[i]
                x = layer.forward(x, training=False)
            # Return pre-softmax logit for target class
            return x[0, target_class]
        
        # Compute gradient of score w.r.t. each spatial position using finite differences
        # This is slower but robust
        epsilon = 1e-4
        
        _, C, H, W = target_activations.shape
        gradients = np.zeros_like(target_activations)
        
        # For efficiency, compute gradient per channel (not per pixel)
        # by using the mean activation change
        base_score = get_target_score(target_activations)
        
        for c in range(C):
            # Perturb this channel slightly
            perturbed = target_activations.copy()
            perturbed[0, c, :, :] += epsilon
            
            new_score = get_target_score(perturbed)
            
            # Gradient is how much the score changes per unit activation
            grad_c = (new_score - base_score) / epsilon
            gradients[0, c, :, :] = grad_c
        
        # Step 3: Global average pooling of gradients to get importance weights
        weights = np.mean(gradients, axis=(2, 3))[0]  # Shape: (C,)
        
        # Step 4: Weighted combination of feature maps
        cam = np.zeros((H, W), dtype=np.float64)
        for c in range(C):
            cam += weights[c] * target_activations[0, c]
        
        # Step 5: ReLU - only positive influences
        cam = np.maximum(cam, 0)
        
        # Step 6: Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_batch(self, images, target_classes=None):
        """Generate Grad-CAM heatmaps for a batch of images."""
        heatmaps = []
        for i, image in enumerate(images):
            target = target_classes[i] if target_classes is not None else None
            heatmap = self.generate(image, target)
            heatmaps.append(heatmap)
        return heatmaps
    
    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """Overlay heatmap on original image."""
        from scipy.ndimage import zoom
        
        # Handle different image formats
        if image.ndim == 3:
            if image.shape[0] in [1, 3]:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))
            if image.shape[2] == 1:  # Grayscale
                image = np.squeeze(image)
        
        # Upsample heatmap to image size
        scale_h = image.shape[0] / heatmap.shape[0]
        scale_w = image.shape[1] / heatmap.shape[1]
        heatmap_upsampled = zoom(heatmap, (scale_h, scale_w), order=1)
        
        # Normalize image if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert heatmap to colormap
        heatmap_colored = self._apply_colormap(heatmap_upsampled)
        
        # Convert grayscale to RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Blend
        overlaid = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
        
        return overlaid
    
    def _apply_colormap(self, heatmap):
        """Apply jet colormap to heatmap."""
        h, w = heatmap.shape
        colored = np.zeros((h, w, 3), dtype=np.float64)
        
        for i in range(h):
            for j in range(w):
                v = heatmap[i, j]
                
                if v < 0.25:
                    colored[i, j] = [0, v * 4, 1]
                elif v < 0.5:
                    colored[i, j] = [0, 1, 1 - (v - 0.25) * 4]
                elif v < 0.75:
                    colored[i, j] = [(v - 0.5) * 4, 1, 0]
                else:
                    colored[i, j] = [1, 1 - (v - 0.75) * 4, 0]
        
        return (colored * 255).astype(np.uint8)
    
    def visualize(self, image, heatmap, figsize=(12, 4), save_path=None):
        """Create visualization with original image, heatmap, and overlay."""
        import matplotlib.pyplot as plt
        
        # Prepare image for display
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image_display = np.transpose(image, (1, 2, 0))
        else:
            image_display = image
        
        if image_display.ndim == 3 and image_display.shape[2] == 1:
            image_display = np.squeeze(image_display)
        
        # Create overlay
        overlay = self.overlay_heatmap(image, heatmap)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        if image_display.ndim == 2:
            axes[0].imshow(image_display, cmap='gray')
        else:
            axes[0].imshow(image_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
