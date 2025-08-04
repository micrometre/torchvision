#!/usr/bin/env python3
"""
Dataset Preparation Script for License Plate Segmentation

This script helps organize and prepare your dataset for training.
It can split data into train/validation sets and verify dataset integrity.
"""

import os
import shutil
import random
import numpy as np
from PIL import Image
import argparse


def verify_dataset(images_dir, masks_dir):
    """Verify that images and masks are properly paired."""
    print(f"Verifying dataset in {images_dir} and {masks_dir}")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory does not exist: {images_dir}")
        return False
    
    if not os.path.exists(masks_dir):
        print(f"Error: Masks directory does not exist: {masks_dir}")
        return False
    
    # Get image files
    image_files = []
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(file)
    
    # Check for corresponding masks
    missing_masks = []
    valid_pairs = 0
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_found = False
        
        # Check different mask naming patterns
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(masks_dir, base_name + ext)
            if os.path.exists(mask_path):
                mask_found = True
                break
            
            # Also check pattern: image.jpg.png
            mask_path = os.path.join(masks_dir, img_file + '.png')
            if os.path.exists(mask_path):
                mask_found = True
                break
        
        if mask_found:
            valid_pairs += 1
        else:
            missing_masks.append(img_file)
    
    print(f"Found {len(image_files)} images")
    print(f"Found {valid_pairs} valid image-mask pairs")
    
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images are missing corresponding masks:")
        for img in missing_masks[:5]:  # Show first 5
            print(f"  - {img}")
        if len(missing_masks) > 5:
            print(f"  ... and {len(missing_masks) - 5} more")
    
    return len(missing_masks) == 0


def split_dataset(source_images, source_masks, output_dir, train_ratio=0.8, random_seed=42):
    """Split dataset into train and validation sets."""
    print(f"Splitting dataset with {train_ratio:.1%} for training")
    
    # Create output directories
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_mask_dir = os.path.join(output_dir, 'train', 'masks')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_mask_dir = os.path.join(output_dir, 'val', 'masks')
    
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get image files
    image_files = []
    for file in os.listdir(source_images):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(file)
    
    # Shuffle with fixed seed for reproducibility
    random.seed(random_seed)
    random.shuffle(image_files)
    
    # Split
    train_size = int(len(image_files) * train_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    
    # Copy files
    def copy_files(file_list, dest_img_dir, dest_mask_dir):
        copied = 0
        for img_file in file_list:
            # Copy image
            src_img = os.path.join(source_images, img_file)
            dst_img = os.path.join(dest_img_dir, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Find and copy corresponding mask
            base_name = os.path.splitext(img_file)[0]
            mask_copied = False
            
            for ext in ['.png', '.jpg', '.jpeg']:
                src_mask = os.path.join(source_masks, base_name + ext)
                if os.path.exists(src_mask):
                    dst_mask = os.path.join(dest_mask_dir, base_name + ext)
                    shutil.copy2(src_mask, dst_mask)
                    mask_copied = True
                    break
                
                # Also try pattern: image.jpg.png
                src_mask = os.path.join(source_masks, img_file + '.png')
                if os.path.exists(src_mask):
                    dst_mask = os.path.join(dest_mask_dir, img_file + '.png')
                    shutil.copy2(src_mask, dst_mask)
                    mask_copied = True
                    break
            
            if mask_copied:
                copied += 1
            else:
                print(f"Warning: No mask found for {img_file}")
        
        return copied
    
    train_copied = copy_files(train_files, train_img_dir, train_mask_dir)
    val_copied = copy_files(val_files, val_img_dir, val_mask_dir)
    
    print(f"Dataset split completed:")
    print(f"  Training: {train_copied} pairs")
    print(f"  Validation: {val_copied} pairs")
    print(f"  Total: {train_copied + val_copied} pairs")


def inspect_masks(masks_dir, num_samples=5):
    """Inspect mask files to understand their format."""
    print(f"Inspecting masks in {masks_dir}")
    
    mask_files = []
    for file in os.listdir(masks_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            mask_files.append(file)
    
    if not mask_files:
        print("No mask files found!")
        return
    
    print(f"Found {len(mask_files)} mask files")
    
    # Sample a few masks
    sample_files = mask_files[:num_samples]
    
    for mask_file in sample_files:
        mask_path = os.path.join(masks_dir, mask_file)
        try:
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            
            print(f"\nMask: {mask_file}")
            print(f"  Size: {mask.size}")
            print(f"  Mode: {mask.mode}")
            print(f"  Shape: {mask_array.shape}")
            print(f"  Data type: {mask_array.dtype}")
            print(f"  Value range: {mask_array.min()} - {mask_array.max()}")
            print(f"  Unique values: {np.unique(mask_array)}")
            
        except Exception as e:
            print(f"Error reading {mask_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for license plate segmentation training')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset integrity')
    parser.add_argument('--split', action='store_true',
                       help='Split dataset into train/validation sets')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect mask files')
    parser.add_argument('--source-images', default='raw_data/images',
                       help='Source images directory (for splitting)')
    parser.add_argument('--source-masks', default='raw_data/masks',
                       help='Source masks directory (for splitting)')
    parser.add_argument('--output', default='dataset',
                       help='Output dataset directory')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--images-dir', default='dataset/train/images',
                       help='Images directory to verify/inspect')
    parser.add_argument('--masks-dir', default='dataset/train/masks',
                       help='Masks directory to verify/inspect')
    
    args = parser.parse_args()
    
    if args.split:
        split_dataset(args.source_images, args.source_masks, args.output, args.train_ratio)
    
    if args.verify:
        verify_dataset(args.images_dir, args.masks_dir)
    
    if args.inspect:
        inspect_masks(args.masks_dir)
    
    if not any([args.split, args.verify, args.inspect]):
        print("No action specified. Use --help to see available options.")
        print("\nQuick start:")
        print("1. Verify your dataset: python prepare_dataset.py --verify")
        print("2. Inspect your masks: python prepare_dataset.py --inspect")
        print("3. Split your data: python prepare_dataset.py --split")


if __name__ == '__main__':
    main()
