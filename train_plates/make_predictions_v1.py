#!/usr/bin/env python3
"""
License Plate Segmentation Prediction Script

This script loads a trained model and makes predictions on individual images,
showing both the original image and the segmentation output.
"""

import argparse
import os
import sys
import torch

# Handle matplotlib backend for headless environments
import matplotlib
try:
    import tkinter
    matplotlib.use('TkAgg')  # Try TkAgg first
except (ImportError, Exception):
    try:
        matplotlib.use('Agg')  # Fallback to non-interactive backend
        print("Warning: Using non-interactive matplotlib backend. Plots will be saved but not displayed.")
    except Exception:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_seg.model import create_model


def pred(image, model, device):
    """
    Predict license plate segmentation for a single image.
    
    Args:
        image: PIL Image
        model: PyTorch model
        device: torch device (cuda/cpu)
    
    Returns:
        torch.Tensor: Segmentation output
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    if device.type == 'cuda':
        input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
        return output


def plot_prediction(img, pred, threshold=0.5, output_path=None, show_plot=True):
    """
    Plot the original image and segmentation prediction side by side.
    
    Args:
        img: PIL Image - original image
        pred: torch.Tensor - prediction output
        threshold: float - threshold for binary segmentation
        output_path: str - path to save the plot (optional)
        show_plot: bool - whether to display the plot
    """
    # Check if we can display plots
    backend = matplotlib.get_backend()
    can_display = show_plot and backend != 'Agg'
    
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original Image', fontsize=16)
    plt.axis('off')

    # Segmentation output
    plt.subplot(122)
    segmentation = pred.cpu().numpy()[0] > threshold
    plt.imshow(segmentation, cmap='gray')
    plt.title(f'License Plate Segmentation (threshold={threshold})', fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to: {output_path}")
    
    if can_display:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot - {e}")
            if not output_path:
                print("Consider using --output-dir to save plots instead of displaying them")
    else:
        if not output_path:
            print("Warning: Cannot display plots in current environment. Use --output-dir to save plots.")
        plt.close()


def create_overlay(image, pred, threshold=0.5, alpha=0.7):
    """
    Create an overlay of the segmentation on the original image.
    
    Args:
        image: PIL Image - original image
        pred: torch.Tensor - prediction output
        threshold: float - threshold for binary segmentation
        alpha: float - transparency of the overlay
    
    Returns:
        PIL Image: Image with segmentation overlay
    """
    # Convert prediction to binary mask
    mask = pred.cpu().numpy()[0] > threshold
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create colored overlay (red for license plates)
    overlay = img_array.copy()
    overlay[mask] = [255, 0, 0]  # Red color for detected regions
    
    # Blend original image with overlay
    result = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
    
    return Image.fromarray(result)


def process_image(model_path, image_path, output_dir=None, threshold=0.1, show_plot=True, save_overlay=False):
    """
    Process a single image and generate predictions.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the input image
        output_dir: Directory to save outputs (optional)
        threshold: Confidence threshold for segmentation
        show_plot: Whether to display the plot
        save_overlay: Whether to save overlay image
    """
    # Load model
    print("Loading model...")
    model = create_model(aux_loss=True)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Load and process image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Make prediction
    outputs = pred(image=image, model=model, device=device)
    
    # Generate output paths if output directory is specified
    plot_path = None
    overlay_path = None
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        plot_path = os.path.join(output_dir, f"{base_name}_prediction.png")
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    
    # Plot prediction
    plot_prediction(image, outputs, threshold=threshold, output_path=plot_path, show_plot=show_plot)
    
    # Create and save overlay if requested
    if save_overlay:
        overlay_image = create_overlay(image, outputs, threshold=threshold)
        if overlay_path:
            overlay_image.save(overlay_path)
            print(f"Overlay saved to: {overlay_path}")
        elif show_plot:
            backend = matplotlib.get_backend()
            can_display = backend != 'Agg'
            
            if can_display:
                try:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(overlay_image)
                    plt.title(f'License Plate Detection Overlay (threshold={threshold})', fontsize=16)
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print(f"Warning: Could not display overlay - {e}")
                    print("Consider using --output-dir to save overlay instead")
            else:
                print("Warning: Cannot display overlay in current environment. Use --output-dir to save overlay.")
    
    # Print statistics
    mask = outputs.cpu().numpy()[0] > threshold
    detected_pixels = np.sum(mask)
    total_pixels = mask.size
    coverage = (detected_pixels / total_pixels) * 100
    
    print(f"Detection statistics:")
    print(f"  - Detected pixels: {detected_pixels:,}")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Coverage: {coverage:.2f}%")


def main():
    """Main function to parse arguments and process image."""
    parser = argparse.ArgumentParser(description='Make license plate segmentation predictions on images')
    parser.add_argument('--model', '-m', 
                       default='./examples/model.pth',
                       help='Path to the trained model file (default: ./examples/model.pth)')
    parser.add_argument('--image', '-i',
                       default='./examples/picture.jpg', 
                       help='Path to input image file (default: ./examples/picture.jpg)')
    parser.add_argument('--output-dir', '-o',
                       help='Directory to save output images (optional)')
    parser.add_argument('--threshold', '-t',
                       type=float,
                       default=0.1,
                       help='Confidence threshold for segmentation (default: 0.1)')
    parser.add_argument('--no-display',
                       action='store_true',
                       help='Do not display plots (useful for headless environments)')
    parser.add_argument('--save-overlay',
                       action='store_true',
                       help='Save overlay image with detected regions highlighted')
    parser.add_argument('--save-plots',
                       action='store_true',
                       help='Force saving plots to files (auto-enables if display not available)')
    
    args = parser.parse_args()
    
    # Auto-enable output directory if plots can't be displayed and save-plots is requested
    backend = matplotlib.get_backend()
    if (args.save_plots or backend == 'Agg') and not args.output_dir:
        args.output_dir = './outputs'
        print(f"Auto-creating output directory: {args.output_dir}")
    
    # Validate input files
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Process image
    try:
        process_image(
            model_path=args.model,
            image_path=args.image,
            output_dir=args.output_dir,
            threshold=args.threshold,
            show_plot=not args.no_display,
            save_overlay=args.save_overlay
        )
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
