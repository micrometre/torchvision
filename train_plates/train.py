#!/usr/bin/env python3
"""
License Plate Segmentation Training Script

This script trains a DeepLabV3 model for license plate segmentation using
custom dataset with images and binary masks.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import model
from model import create_model


class LicensePlateDataset(Dataset):
    """Custom dataset for license plate segmentation."""
    
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        """
        Args:
            images_dir (str): Directory with training images
            masks_dir (str): Directory with segmentation masks
            transform: Transform to apply to images
            mask_transform: Transform to apply to masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get list of image files
        self.image_files = []
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(file)
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask - try different extensions
        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
            # Also try with .jpg.png pattern
            potential_path = os.path.join(self.masks_dir, img_name + '.png')
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for image {img_name}")
        
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Convert mask to binary (0 or 1) and add channel dimension
        mask = (mask > 0.5).float().unsqueeze(0)
        
        return image, mask


def get_transforms(image_size=512):
    """Get training and validation transforms."""
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Mask transforms (no normalization for masks)
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    return train_transform, val_transform, mask_transform


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combination of Binary Cross-Entropy and Dice loss."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        
        # Apply sigmoid for dice loss calculation
        predictions_sigmoid = torch.sigmoid(predictions)
        dice = self.dice_loss(predictions_sigmoid, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_iou(predictions, targets, threshold=0.5):
    """Calculate Intersection over Union (IoU) metric."""
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            predictions = torch.sigmoid(outputs)
            iou = calculate_iou(predictions, masks)
        
        running_loss += loss.item()
        running_iou += iou
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{iou:.4f}',
            'Avg Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Avg IoU': f'{running_iou/(batch_idx+1):.4f}'
        })
    
    avg_loss = running_loss / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    
    return avg_loss, avg_iou


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            predictions = torch.sigmoid(outputs)
            iou = calculate_iou(predictions, masks)
            
            running_loss += loss.item()
            running_iou += iou
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Avg IoU': f'{running_iou/(batch_idx+1):.4f}'
            })
    
    avg_loss = running_loss / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    
    return avg_loss, avg_iou


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_iou, val_iou, 
                   checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_iou': train_iou,
        'val_iou': val_iou
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def plot_training_history(train_losses, val_losses, train_ious, val_ious, save_path):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot IoU
    ax2.plot(train_ious, label='Train IoU', color='blue')
    ax2.plot(val_ious, label='Val IoU', color='red')
    ax2.set_title('Training and Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train license plate segmentation model')
    parser.add_argument('--train-images', default='dataset/train/images',
                       help='Path to training images directory')
    parser.add_argument('--train-masks', default='dataset/train/masks',
                       help='Path to training masks directory')
    parser.add_argument('--val-images', default='dataset/val/images',
                       help='Path to validation images directory')
    parser.add_argument('--val-masks', default='dataset/val/masks',
                       help='Path to validation masks directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Input image size (default: 512)')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', 
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone weights')
    
    args = parser.parse_args()
    
    # Check if datasets exist
    for path in [args.train_images, args.train_masks]:
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            print("Please make sure your dataset is organized as:")
            print("dataset/train/images/ - training images")
            print("dataset/train/masks/  - training masks")
            print("dataset/val/images/   - validation images") 
            print("dataset/val/masks/    - validation masks")
            sys.exit(1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform, mask_transform = get_transforms(args.image_size)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = LicensePlateDataset(
        args.train_images, args.train_masks, 
        transform=train_transform, mask_transform=mask_transform
    )
    
    # Create validation dataset if validation paths exist
    val_dataset = None
    if os.path.exists(args.val_images) and os.path.exists(args.val_masks):
        val_dataset = LicensePlateDataset(
            args.val_images, args.val_masks,
            transform=val_transform, mask_transform=mask_transform
        )
    else:
        print("Validation dataset not found. Training without validation.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
    
    # Create model
    print("Creating model...")
    model = create_model(outputchannels=1, aux_loss=True, freeze_backbone=args.freeze_backbone)
    model.to(device)
    
    # Create loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.6, dice_weight=0.4)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Resume training if checkpoint provided
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training history
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    best_val_iou = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    print("-" * 50)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        
        # Validate
        val_loss, val_iou = 0.0, 0.0
        if val_loader:
            val_loss, val_iou = validate_epoch(model, val_loader, criterion, device, epoch)
            val_losses.append(val_loss)
            val_ious.append(val_iou)
            scheduler.step(val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        if val_loader:
            print(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_iou > best_val_iou if val_loader else train_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou if val_loader else train_iou
        
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_iou, val_iou, 
                       args.checkpoint_dir, is_best)
        
        # Plot training history
        if (epoch + 1) % 5 == 0:
            plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
            plot_training_history(train_losses, val_losses, train_ious, val_ious, plot_path)
    
    print("\nTraining completed!")
    print(f"Best IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {os.path.join(args.checkpoint_dir, 'model.pth')}")


if __name__ == '__main__':
    main()
