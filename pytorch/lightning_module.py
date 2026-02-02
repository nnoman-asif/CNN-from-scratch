"""
PyTorch Lightning Module for CNN
================================

Production-ready training with PyTorch Lightning.
Provides:
- Automatic logging
- Checkpointing
- Multi-GPU support
- Early stopping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset


class CNNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for CNN training.
    
    Example:
        >>> from pytorch_lightning import Trainer
        >>> 
        >>> model = CNNLightningModule(input_channels=1, num_classes=10)
        >>> trainer = Trainer(max_epochs=10)
        >>> trainer.fit(model, train_dataloader, val_dataloader)
    """
    
    def __init__(self, input_channels=1, num_classes=10, learning_rate=0.001,
                 dropout_rate=0.25, dense_dropout=0.5, use_batchnorm=True):
        """
        Initialize Lightning module.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate for conv blocks
            dense_dropout: Dropout rate for dense layers
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Build network
        self.conv_block1 = self._make_conv_block(input_channels, 32, use_batchnorm)
        self.conv_block2 = self._make_conv_block(32, 32, use_batchnorm)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout2d_1 = nn.Dropout2d(dropout_rate)
        
        self.conv_block3 = self._make_conv_block(32, 64, use_batchnorm)
        self.conv_block4 = self._make_conv_block(64, 64, use_batchnorm)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2d_2 = nn.Dropout2d(dropout_rate)
        
        # FC layers (28x28 -> 14x14 -> 7x7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout(dense_dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # For storing feature maps (Grad-CAM)
        self.feature_maps = None
    
    def _make_conv_block(self, in_channels, out_channels, use_batchnorm):
        """Create a conv-bn-relu block."""
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        # Block 1
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool1(x)
        x = self.dropout2d_1(x)
        
        # Block 2
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Store for Grad-CAM
        self.feature_maps = x
        
        x = self.pool2(x)
        x = self.dropout2d_2(x)
        
        # FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        logits = self(x)
        return F.softmax(logits, dim=1)


def train_with_lightning(X_train, y_train, X_val, y_val, 
                        max_epochs=20, batch_size=32,
                        learning_rate=0.001, checkpoint_dir='checkpoints',
                        early_stopping_patience=5):
    """
    Train CNN using PyTorch Lightning.
    
    Args:
        X_train, y_train: Training data (numpy arrays)
        X_val, y_val: Validation data
        max_epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        early_stopping_patience: Patience for early stopping
    
    Returns:
        Trained model and trainer
    """
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = CNNLightningModule(
        input_channels=X_train.shape[1],
        num_classes=len(set(y_train)),
        learning_rate=learning_rate
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='cnn-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer


def load_lightning_checkpoint(checkpoint_path, **kwargs):
    """
    Load model from Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        **kwargs: Arguments for model
    
    Returns:
        Loaded model
    """
    model = CNNLightningModule.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


