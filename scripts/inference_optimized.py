#!/usr/bin/env python3
"""
Optimized inference script for license plate detection using a fine-tuned Faster R-CNN model.
Designed for improved CPU performance.
"""

import os
import time
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

def load_and_preprocess_image(image_path, input_size=320):
    """Load and preprocess an image for inference with resizing for faster processing"""
    # Load the image
    img = Image.open(image_path).convert("RGB")
    original_img = img.copy()
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations and add a batch dimension
    img_tensor = transform(img).unsqueeze(0)
    return original_img, img_tensor

def load_custom_model(model_path, num_classes=2, use_jit=False, input_size=320):
    """Load a custom model with options for JIT or regular loading"""
    if use_jit and os.path.exists(model_path.replace('.pth', '_traced.pt')):
        # Load JIT traced model if available
        model = torch.jit.load(model_path.replace('.pth', '_traced.pt'))
        device = torch.device('cpu')
        return model, device
    
    # Check if quantized model exists
    quantized_path = model_path.replace('.pth', '_quantized.pth')
    if os.path.exists(quantized_path):
        try:
            # Load the model architecture with custom number of classes
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
            
            # Replace the classifier with a new one for custom num_classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # Load the quantized state dict
            model.load_state_dict(torch.load(quantized_path, map_location=torch.device('cpu')))
            print("Loaded quantized model for faster inference")
            
            device = torch.device('cpu')
            model.eval()
            return model, device
        except Exception as e:
            print(f"Failed to load quantized model: {e}, falling back to regular model")
    
    # Fall back to regular model loading
    # Load the model architecture with custom number of classes
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    
    # Replace the classifier with a new one for custom num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load the state dict from the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cpu')
    model.eval()
    
    # Apply torch.no_grad for better performance
    for param in model.parameters():
        param.requires_grad = False
        
    return model, device

def detect_objects(model, image_tensor, device):
    """Run object detection with timing"""
    # Move tensor to appropriate device (CPU in this case)
    image_tensor = image_tensor.to(device)
    
    # Perform inference with timing
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor)
    inference_time = time.time() - start_time
    
    return predictions, inference_time

def visualize_detection(image, predictions, confidence_threshold=0.5, show_inference_time=None):
    """Visualize detection results with bounding boxes"""
    # Get the first (and only) image's predictions
    pred = predictions[0]
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Filter predictions by confidence
    high_conf_indices = pred['scores'] > confidence_threshold
    boxes = pred['boxes'][high_conf_indices]
    labels = pred['labels'][high_conf_indices]
    scores = pred['scores'][high_conf_indices]
    
    # Custom class labels for license plate detection
    custom_labels = ['__background__', 'license_plate']
    
    # Draw bounding boxes and labels
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create a rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        # Add label text
        label_text = f"{custom_labels[label]}: {score:.2f}"
        ax.text(
            x1, y1 - 10, label_text,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
        )
    
    # Add inference time if provided
    if show_inference_time:
        ax.text(
            10, 30, f"Inference time: {show_inference_time:.3f} seconds",
            color='white', fontsize=14, bbox=dict(facecolor='blue', alpha=0.5)
        )
    
    plt.axis('off')
    plt.tight_layout()
    return fig

def save_or_show_results(fig, output_path=None):
    """Save figure to file or display it"""
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Results saved to {output_path}")
    else:
        plt.show()

def main(args):
    print(f"Processing image: {args.image_path}")
    
    # Load and preprocess image
    image, image_tensor = load_and_preprocess_image(args.image_path, input_size=args.img_size)
    
    # Load custom model
    model, device = load_custom_model(
        args.model_path, 
        use_jit=args.use_jit,
        input_size=args.img_size
    )
    
    # Run detection
    predictions, inference_time = detect_objects(model, image_tensor, device)
    print(f"Inference completed in {inference_time:.3f} seconds")
    
    # Visualize results
    fig = visualize_detection(
        image, 
        predictions, 
        confidence_threshold=args.threshold,
        show_inference_time=inference_time
    )
    
    # Save or show results
    save_or_show_results(fig, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimized inference for license plate detection")
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--model-path', type=str, default='../outputs/license_plates/model_epoch_0.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save the output image (if not specified, image will be displayed)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for displaying detections')
    parser.add_argument('--img-size', type=int, default=320,
                        help='Image size for inference (smaller is faster)')
    parser.add_argument('--use-jit', action='store_true',
                        help='Use JIT traced model if available')
    
    args = parser.parse_args()
    main(args)
