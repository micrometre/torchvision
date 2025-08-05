#!/usr/bin/env python3
"""
License Plate Segmentation Video Prediction Script

This script loads a trained model and makes predictions on video input,
detecting and highlighting license plates in each frame.
"""

import argparse
import os
import sys
import time
import torch

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves files only

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_seg.model import create_model


def pred(image, model, device):
    """
    Predict license plate segmentation for a single frame.
    
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


def create_overlay(frame, pred, threshold=0.5, border_thickness=3, label=True):
    """
    Create an overlay with green border around detected license plates.
    
    Args:
        frame: numpy array - original video frame
        pred: torch.Tensor - prediction output
        threshold: float - threshold for binary segmentation
        border_thickness: int - thickness of the green border
        label: bool - whether to add "License Plate" label
    
    Returns:
        tuple: (result_frame, license_plate_crops)
            - result_frame: numpy array - Frame with green border around detected regions
            - license_plate_crops: list of (cropped_image, (x, y, w, h)) tuples
    """
    # Convert prediction to binary mask
    mask = (pred.cpu().numpy()[0] > threshold).astype(np.uint8)
    
    # Convert frame to numpy array if it's not already
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    
    # Make a copy to avoid modifying the original
    result_frame = frame.copy()
    
    # Find contours of detected regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store cropped license plates
    license_plate_crops = []
    
    # Draw green borders around detected regions
    for contour in contours:
        # Get bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add some padding around the license plate (10% on each side)
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        # Make sure we don't go outside the image boundaries
        x_padded = max(0, x - padding_x)
        y_padded = max(0, y - padding_y)
        w_padded = min(frame.shape[1] - x_padded, w + 2 * padding_x)
        h_padded = min(frame.shape[0] - y_padded, h + 2 * padding_y)
        
        # Crop the license plate region with padding
        crop = frame[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded].copy()
        
        # Add to the list of crops
        license_plate_crops.append((crop, (x, y, w, h)))
        
        # Draw green rectangle border
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), border_thickness)
        
        # Add "License Plate" label above the rectangle
        if label:
            cv2.putText(result_frame, "License Plate", (x, max(0, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_frame, license_plate_crops


def process_video(model_path, video_path, output_dir, threshold=0.1, 
                 border_thickness=3, skip_frames=0, max_frames=None, 
                 display_progress=True, resize_factor=1.0, save_frequency=1,
                 image_format='jpg', quality=95):
    """
    Process a video file and generate predictions on each frame, saving as individual images.
    
    Args:
        model_path: Path to the trained model
        video_path: Path to the input video
        output_dir: Directory to save the output images
        threshold: Confidence threshold for segmentation
        border_thickness: Thickness of the green border around detected plates
        skip_frames: Number of frames to skip between processing (0 = process every frame)
        max_frames: Maximum number of frames to process (None = process all frames)
        display_progress: Whether to show a progress indicator
        resize_factor: Factor by which to resize frames (1.0 = original size)
        save_frequency: Save every nth processed frame (1 = save all processed frames)
        image_format: Format to save images ('jpg' or 'png')
        quality: Image quality for JPEG (0-100)
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

    # Open video
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is None:
        max_frames = total_frames
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for cropped license plates
    cropped_plates_dir = os.path.join(output_dir, "wrapped_images")
    os.makedirs(cropped_plates_dir, exist_ok=True)
    
    # Create directory for original frames (before detection)
    original_frames_dir = os.path.join(output_dir, "original_frames")
    os.makedirs(original_frames_dir, exist_ok=True)
    
    # Generate a base filename from the video filename
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"Video info:")
    print(f"  - Dimensions: {frame_width}x{frame_height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Processing: {min(max_frames, total_frames)} frames")
    print(f"  - Frame skip: {skip_frames} (processing every {skip_frames + 1}th frame)")
    print(f"  - Saving processed images to: {output_dir}")
    print(f"  - Saving original frames to: {original_frames_dir}")
    print(f"  - Saving cropped license plates to: {cropped_plates_dir}")
    
    # Process video
    frame_count = 0
    processed_count = 0
    saved_count = 0
    start_time = time.time()
    
    try:
        if skip_frames > 0:
            # More efficient frame skipping for larger skip values
            frame_indices = range(0, min(max_frames, total_frames), skip_frames + 1)
            target_frames = len(frame_indices)
            print(f"  - With frame skip: will process approximately {target_frames} frames")
            
            for frame_idx in frame_indices:
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                frame_count = frame_idx + 1  # +1 because frame indices are 0-based
                
                # Resize if requested
                if resize_factor != 1.0:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                    
                # Convert to PIL Image (required for model prediction)
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Make prediction
                outputs = pred(image=pil_image, model=model, device=device)
                
                # Create overlay and get license plate crops
                result_frame, license_plate_crops = create_overlay(
                    frame, outputs, threshold=threshold, border_thickness=border_thickness
                )
                
                # Save image if it's the right frequency
                if processed_count % save_frequency == 0:
                    # Save original frame (before detection)
                    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    original_pil = Image.fromarray(original_rgb)
                    
                    # Save original image
                    original_filename = f"{video_basename}_frame_{frame_count:06d}.{image_format}"
                    original_path = os.path.join(original_frames_dir, original_filename)
                    
                    if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                        original_pil.save(original_path, quality=quality, optimize=True)
                    else:
                        original_pil.save(original_path)
                    
                    # Convert OpenCV BGR image back to RGB for PIL
                    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_rgb)
                    
                    # Save processed image
                    image_filename = f"{video_basename}_frame_{frame_count:06d}.{image_format}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                        result_pil.save(image_path, quality=quality, optimize=True)
                    else:
                        result_pil.save(image_path)
                    
                    # Save cropped license plates if any were detected
                    if license_plate_crops:
                        for i, (crop, (x, y, w, h)) in enumerate(license_plate_crops):
                            # Convert from BGR to RGB for PIL
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            crop_pil = Image.fromarray(crop_rgb)
                            
                            # Save the crop
                            crop_filename = f"{video_basename}_frame_{frame_count:06d}_plate_{i+1}.{image_format}"
                            crop_path = os.path.join(cropped_plates_dir, crop_filename)
                            
                            if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                                crop_pil.save(crop_path, quality=quality, optimize=True)
                            else:
                                crop_pil.save(crop_path)
                    else:
                        # Create a placeholder small black image with "No License Plate" text
                        placeholder = np.zeros((100, 300, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "No License Plate", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Convert to PIL and save
                        placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
                        placeholder_pil = Image.fromarray(placeholder_rgb)
                        
                        crop_filename = f"{video_basename}_frame_{frame_count:06d}_plate_1.{image_format}"
                        crop_path = os.path.join(cropped_plates_dir, crop_filename)
                        
                        if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                            placeholder_pil.save(crop_path, quality=quality, optimize=True)
                        else:
                            placeholder_pil.save(crop_path)
                    
                    saved_count += 1
                
                processed_count += 1
                
                # Display progress
                if display_progress and processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (target_frames - processed_count) / fps_processing if fps_processing > 0 else 0
                    
                    print(f"Processed {processed_count}/{target_frames} frames "
                          f"({processed_count/target_frames*100:.1f}%) - "
                          f"{fps_processing:.2f} FPS - "
                          f"ETA: {int(remaining/60):02d}:{int(remaining%60):02d}")
        else:
            # Process every frame sequentially
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                frame_count += 1
                
                # Resize if requested
                if resize_factor != 1.0:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                    
                # Convert to PIL Image (required for model prediction)
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Make prediction
                outputs = pred(image=pil_image, model=model, device=device)
                
                # Create overlay and get license plate crops
                result_frame, license_plate_crops = create_overlay(
                    frame, outputs, threshold=threshold, border_thickness=border_thickness
                )
                
                # Save image if it's the right frequency
                if processed_count % save_frequency == 0:
                    # Save original frame (before detection)
                    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    original_pil = Image.fromarray(original_rgb)
                    
                    # Save original image
                    original_filename = f"{video_basename}_frame_{frame_count:06d}.{image_format}"
                    original_path = os.path.join(original_frames_dir, original_filename)
                    
                    if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                        original_pil.save(original_path, quality=quality, optimize=True)
                    else:
                        original_pil.save(original_path)
                    
                    # Convert OpenCV BGR image back to RGB for PIL
                    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_rgb)
                    
                    # Save processed image
                    image_filename = f"{video_basename}_frame_{frame_count:06d}.{image_format}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                        result_pil.save(image_path, quality=quality, optimize=True)
                    else:
                        result_pil.save(image_path)
                    
                    # Save cropped license plates if any were detected
                    if license_plate_crops:
                        for i, (crop, (x, y, w, h)) in enumerate(license_plate_crops):
                            # Convert from BGR to RGB for PIL
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            crop_pil = Image.fromarray(crop_rgb)
                            
                            # Save the crop
                            crop_filename = f"{video_basename}_frame_{frame_count:06d}_plate_{i+1}.{image_format}"
                            crop_path = os.path.join(cropped_plates_dir, crop_filename)
                            
                            if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                                crop_pil.save(crop_path, quality=quality, optimize=True)
                            else:
                                crop_pil.save(crop_path)
                    else:
                        # Create a placeholder small black image with "No License Plate" text
                        placeholder = np.zeros((100, 300, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "No License Plate", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Convert to PIL and save
                        placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
                        placeholder_pil = Image.fromarray(placeholder_rgb)
                        
                        crop_filename = f"{video_basename}_frame_{frame_count:06d}_plate_1.{image_format}"
                        crop_path = os.path.join(cropped_plates_dir, crop_filename)
                        
                        if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                            placeholder_pil.save(crop_path, quality=quality, optimize=True)
                        else:
                            placeholder_pil.save(crop_path)
                    
                    saved_count += 1
                
                processed_count += 1
                
                # Display progress
                if display_progress and processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (min(max_frames, total_frames) - processed_count) / fps_processing if fps_processing > 0 else 0
                    
                    print(f"Processed {processed_count}/{min(max_frames, total_frames)} frames "
                          f"({processed_count/min(max_frames, total_frames)*100:.1f}%) - "
                          f"{fps_processing:.2f} FPS - "
                          f"ETA: {int(remaining/60):02d}:{int(remaining%60):02d}")
    
    finally:
        # Release resources
        cap.release()
        
        if processed_count > 0:
            elapsed = time.time() - start_time
            print(f"Processing complete!")
            print(f"  - Processed {processed_count} frames in {elapsed:.2f} seconds")
            print(f"  - Saved {saved_count} original frames to {original_frames_dir}")
            print(f"  - Saved {saved_count} processed frames to {output_dir}")
            print(f"  - Saved {len(os.listdir(cropped_plates_dir))} license plate crops to {cropped_plates_dir}")
            print(f"  - Average speed: {processed_count/elapsed:.2f} FPS")
        else:
            print("No frames were processed.")


def main():
    """Main function to parse arguments and process video."""
    parser = argparse.ArgumentParser(description='Make license plate segmentation predictions on video')
    parser.add_argument('--model', '-m', 
                       default='./examples/model.pth',
                       help='Path to the trained model file (default: ./examples/model.pth)')
    parser.add_argument('--video', '-v', required=True,
                       help='Path to input video file')
    parser.add_argument('--output-dir', '-o',
                       default='./outputs/frames',
                       help='Directory to save output images (default: ./outputs/frames)')
    parser.add_argument('--threshold', '-t',
                       type=float,
                       default=0.1,
                       help='Confidence threshold for segmentation (default: 0.1)')
    parser.add_argument('--border-thickness', '-b',
                       type=int,
                       default=3,
                       help='Thickness of green border around detected license plates (default: 3)')
    parser.add_argument('--skip-frames', '-s',
                       type=int,
                       default=0,
                       help='Number of frames to skip between processing (default: 0 = process every frame, 1 = process every other frame, 2 = process every third frame, etc.)')
    parser.add_argument('--save-frequency', '-sf',
                       type=int,
                       default=1,
                       help='Save every nth processed frame (default: 1, save all processed frames)')
    parser.add_argument('--max-frames', '-mf',
                       type=int,
                       default=None,
                       help='Maximum number of frames to process (default: None, process all frames)')
    parser.add_argument('--resize-factor', '-rf',
                       type=float,
                       default=1.0,
                       help='Factor by which to resize frames (default: 1.0, original size)')
    parser.add_argument('--image-format', '-if',
                       choices=['jpg', 'png'],
                       default='jpg',
                       help='Format to save images (default: jpg)')
    parser.add_argument('--quality', '-q',
                       type=int,
                       default=95,
                       help='Image quality for JPEG (0-100, default: 95)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Process video
    try:
        process_video(
            model_path=args.model,
            video_path=args.video,
            output_dir=args.output_dir,
            threshold=args.threshold,
            border_thickness=args.border_thickness,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            resize_factor=args.resize_factor,
            save_frequency=args.save_frequency,
            image_format=args.image_format,
            quality=args.quality
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
