#!/usr/bin/env python3
"""
Training script for fine-tuning a pre-trained Faster R-CNN model on custom data.
"""

import os
import sys
import json
import time
import copy
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

# Define a custom dataset class for COCO-format data
class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Args:
            root (string): Root directory where images are stored
            annFile (string): Path to the annotation file
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.transform = transform
        
        # Load annotations
        with open(annFile, 'r') as f:
            self.coco_data = json.load(f)
        
        # Map category_id to continuous index
        self.cat_mapping = {}
        for i, cat in enumerate(self.coco_data['categories']):
            self.cat_mapping[cat['id']] = i + 1  # 0 is background in RCNN
        
        # Get all image ids that have annotations
        self.image_ids = []
        self.image_info = {}
        for img in self.coco_data['images']:
            self.image_info[img['id']] = img
            self.image_ids.append(img['id'])
        
        # Create annotation mapping
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        target = {}
        if img_id in self.annotations:
            anns = self.annotations[img_id]
            boxes = []
            labels = []
            
            for ann in anns:
                # COCO bbox format: [x, y, width, height]
                # PyTorch expects: [x_min, y_min, x_max, y_max]
                x, y, width, height = ann['bbox']
                boxes.append([x, y, x + width, y + height])
                labels.append(self.cat_mapping[ann['category_id']])
            
            # Convert to tensors
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.tensor([(x2 - x1) * (y2 - y1) for [x1, y1, x2, y2] in boxes])
            target["iscrowd"] = torch.zeros((len(anns),), dtype=torch.int64)
        else:
            # Empty annotations
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        return img, target

def get_transform(train, input_size=320):
    transforms_list = []
    # Resize images to smaller dimensions for faster processing
    transforms_list.append(transforms.Resize((input_size, input_size)))
    # Convert PIL image to tensor and normalize
    transforms_list.append(transforms.ToTensor())
    # Add normalization for better model performance
    transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225]))
    if train:
        # Add training augmentations specific for license plate detection
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        # Add slight rotations to simulate different camera angles
        transforms_list.append(transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    return transforms.Compose(transforms_list)

def get_model_instance_segmentation(num_classes):
    # Option 1: Use smaller backbone (MobileNet instead of ResNet50)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    )
    
    # Replace the classifier with a new one for custom num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def evaluate(model, data_loader, device):
    model.eval()
    
    all_losses = []
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        try:
            with torch.no_grad():
                # Force the model to return losses by passing targets
                outputs = model(images, targets)
                
                # Handle different return types
                if isinstance(outputs, dict):
                    # Standard loss dictionary
                    losses = sum(loss for loss in outputs.values())
                    all_losses.append(losses.item())
                else:
                    print("Warning: Model returned unexpected output type during evaluation")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            continue
    
    if not all_losses:
        print("Warning: No loss values were collected during evaluation.")
        return float('inf')  # Return a high loss to avoid selecting this as best model
        
    return sum(all_losses) / len(all_losses)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    
    all_losses = []
    lr_scheduler = None
    
    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                
                # Print losses
                if i % print_freq == 0:
                    loss_str = ' '.join(f'{k}: {v.item():.4f}' for k, v in loss_dict.items())
                    print(f"Epoch: {epoch}, Batch: {i}/{len(data_loader)}, Losses: {loss_str}")
                
                losses.backward()
                optimizer.step()
                
                all_losses.append(losses.item())
            else:
                print(f"Warning: Unexpected output type in batch {i}. Skipping.")
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    if not all_losses:
        print("Warning: No loss values were collected during training.")
        return float('inf')
        
    return sum(all_losses) / len(all_losses)

def collate_fn(batch):
    return tuple(zip(*batch))

def save_model(model, optimizer, epoch, loss, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    
    # Also save an optimized version for inference
    if filename == "best_model.pth":
        optimize_for_inference(model, output_dir)

def optimize_for_inference(model, output_dir):
    """Create an optimized version of the model for faster inference on CPU"""
    # Make a copy of the model for quantization
    model_quantized = copy.deepcopy(model)
    model_quantized.eval()
    
    # Apply static quantization
    try:
        # Fuse Conv, BN and ReLU layers for better performance
        model_quantized.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_quantized_prepared = torch.quantization.prepare(model_quantized)
        model_quantized = torch.quantization.convert(model_quantized_prepared)
        
        # Save quantized model
        torch.save(model_quantized.state_dict(), os.path.join(output_dir, "model_quantized.pth"))
        print("Saved quantized model for faster CPU inference")
    except Exception as e:
        print(f"Quantization failed: {e}")
        # Save a JIT traced model as fallback
        try:
            # Create a script module from the model
            dummy_input = torch.randn(1, 3, 320, 320)
            traced_model = torch.jit.trace(model, [dummy_input])
            traced_model.save(os.path.join(output_dir, "model_traced.pt"))
            print("Saved traced model for faster CPU inference")
        except Exception as e:
            print(f"Model tracing failed: {e}")

def plot_losses(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

def visualize_predictions(model, data_loader, device, output_dir, num_images=5):
    """Visualize predictions on validation set"""
    model.eval()
    images_seen = 0
    
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # Mapping of label IDs to class names
    label_map = {1: 'license_plate'}  # Adjust based on your dataset
    
    for images, targets in data_loader:
        try:
            images = [img.to(device) for img in images]
            
            with torch.no_grad():
                # When calling the model without targets, it returns predictions
                predictions = model(images)
            
            for i, (img_tensor, prediction, target) in enumerate(zip(images, predictions, targets)):
                if images_seen >= num_images:
                    break
                    
                # Convert tensor to PIL Image for drawing
                img = F.to_pil_image(img_tensor.cpu())
                draw = ImageDraw.Draw(img)
                
                # Draw ground truth bounding boxes in green
                for box in target['boxes'].cpu().numpy():
                    draw.rectangle(
                        [(box[0], box[1]), (box[2], box[3])],
                        outline='green',
                        width=3
                    )
                
                # Draw predicted bounding boxes in red
                if 'boxes' in prediction and len(prediction['boxes']) > 0:
                    for box, score, label in zip(
                        prediction['boxes'].cpu().numpy(),
                        prediction['scores'].cpu().numpy(),
                        prediction['labels'].cpu().numpy()
                    ):
                        if score > 0.5:  # Only show predictions with confidence > 50%
                            draw.rectangle(
                                [(box[0], box[1]), (box[2], box[3])],
                                outline='red',
                                width=3
                            )
                            
                            # Add score and label text
                            label_text = f"{label_map.get(label, f'class_{label}')} {score:.2f}"
                            draw.text((box[0], box[1] - 10), label_text, fill='red')
                
                # Save the image with predictions
                img.save(os.path.join(output_dir, 'predictions', f'pred_{images_seen}.jpg'))
                images_seen += 1
            
            if images_seen >= num_images:
                break
        except Exception as e:
            print(f"Error during visualization: {e}")
            continue
    
    print(f"Saved {images_seen} prediction visualizations to {os.path.join(output_dir, 'predictions')}")

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set thread and optimization parameters for CPU
    if device.type == 'cpu':
        # Set number of threads for better CPU performance
        torch.set_num_threads(args.num_workers)
        print(f"Set PyTorch to use {args.num_workers} threads")
        
        # Use deterministic algorithms for better CPU performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load categories from annotation file
    with open(args.train_ann_file, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    num_classes = len(categories) + 1  # +1 for background class
    
    print(f"Training with {len(categories)} classes plus background")
    
    # Initialize datasets and dataloaders
    dataset_train = COCODataset(args.train_img_dir, args.train_ann_file, transform=get_transform(train=True, input_size=args.img_size))
    dataset_val = COCODataset(args.val_img_dir, args.val_ann_file, transform=get_transform(train=False, input_size=args.img_size))
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )
    
    # Initialize model
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    
    # Training loop
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Resume training if checkpoint exists
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    print("Starting training")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch)
        train_losses.append(train_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Validate
        val_loss = evaluate(model, data_loader_val, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model
        save_model(model, optimizer, epoch, val_loss, args.output_dir, f"model_epoch_{epoch}.pth")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, args.output_dir, "best_model.pth")
            print(f"Saved new best model with val_loss: {val_loss:.4f}")
        
        # Plot losses
        plot_losses(train_losses, val_losses, args.output_dir)
    
    # After training, visualize some predictions
    print("Visualizing predictions on validation data...")
    visualize_predictions(model, data_loader_val, device, args.output_dir, num_images=10)
    
    print(f"Training complete. Model saved to {os.path.join(args.output_dir, 'best_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN model")
    parser.add_argument('--train-img-dir', type=str, default='dataset/train/images',
                        help='path to training images directory')
    parser.add_argument('--val-img-dir', type=str, default='dataset/valid/images',
                        help='path to validation images directory')
    parser.add_argument('--train-ann-file', type=str, default='dataset/train/annotations.json',
                        help='path to training annotations file')
    parser.add_argument('--val-ann-file', type=str, default='dataset/valid/annotations.json',
                        help='path to validation annotations file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='directory to save outputs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='weight decay for SGD optimizer')
    parser.add_argument('--lr-step-size', type=int, default=3,
                        help='step size for learning rate scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='gamma for learning rate scheduler')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading and CPU threads')
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint to resume from')
    parser.add_argument('--img-size', type=int, default=320,
                        help='image size for training and inference (smaller is faster)')
    parser.add_argument('--optimize-cpu', action='store_true',
                        help='apply additional CPU optimizations')
    
    args = parser.parse_args()
    main(args)
