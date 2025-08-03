import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image 

def load_onnx_model(onnx_path):
    # Initialize ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    return session

def preprocess_image(image):
    # Convert PIL Image to numpy array and preprocess
    img = np.array(image)
    img = cv2.resize(img, (640, 640))  # Adjust size to match your ONNX model
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0  # Add batch dimension and normalize
    return img

def detect_objects(session, image_tensor):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_tensor})
    
    # For output shape [1, 5, 8400] (YOLO format)
    detections = outputs[0][0]  # Remove batch dimension -> [5, 8400]
    
    # Convert to expected format
    scores = detections[4, :]  # Confidence scores are typically at index 4
    keep = scores > 0.1  # Initial score threshold
    
    # Get boxes (xywh format common in YOLO)
    boxes = detections[:4, keep].T  # [x_center, y_center, width, height]
    scores = detections[4, keep]
    
    # Convert from xywh to xyxy format
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - width/2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - height/2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = x_center + width/2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = y_center + height/2
    
    # For YOLO models, class IDs are often not in this output
    # You'll need to either:
    # 1. Get them from a separate output (if your model has multiple outputs)
    # 2. Use dummy class IDs (if you only have one class)
    labels = np.zeros(len(scores), dtype=np.int32)  # Dummy class 0
    
    return {
        'boxes': boxes_xyxy,
        'scores': scores,
        'labels': labels
    }

def get_coco_labels():
    """Return complete COCO labels list with 91 classes"""
    return [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

def process_video_to_frames(video_path, session, output_dir="output", threshold=0.5, frame_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    coco_labels = get_coco_labels()
    
    actual_frames = (total_frames + frame_interval - 1) // frame_interval
    saved_count = 0
    
    with tqdm(total=actual_frames, desc="Processing video frames") as pbar:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_num % frame_interval == 0:
                # Preprocess frame
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tensor = preprocess_image(img)
                
                # Detect objects
                predictions = detect_objects(session, img_tensor)
                
                # Draw detections - FIX: Don't access [0] since detect_objects already returns the arrays directly
                boxes = predictions['boxes']  # Already the correct shape
                scores = predictions['scores']  # Already the correct shape
                labels = predictions['labels']  # Already the correct shape
                
                # Scale boxes back to original frame size
                original_height, original_width = frame.shape[:2]
                scale_x = original_width / 640
                scale_y = original_height / 640
                
                for box, label, score in zip(boxes, labels, scores):
                    if score > threshold:
                        # Scale coordinates back to original frame size
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, min(x1, original_width - 1))
                        y1 = max(0, min(y1, original_height - 1))
                        x2 = max(0, min(x2, original_width - 1))
                        y2 = max(0, min(y2, original_height - 1))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label_text = f"class_{label}: {score:.2f}"
                        if 0 <= label < len(coco_labels):
                            label_text = f"{coco_labels[label]}: {score:.2f}"
                        
                        cv2.putText(frame, label_text, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                output_path = os.path.join(output_dir, f"frame_{frame_num:05d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                pbar.update(1)
            
            frame_num += 1
            if frame_num >= total_frames:
                break
    
    cap.release()
    print(f"\nProcessing complete. Saved {saved_count} frames to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with ONNX Model")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--input", type=str, required=True, help="Input image or video file path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output frames")
    parser.add_argument("--frame_interval", type=int, default=29, 
                       help="Process every N frames")
    args = parser.parse_args()
    
    # Load ONNX model
    session = load_onnx_model(args.model)
    
    if args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Image processing
        img = Image.open(args.input).convert("RGB")
        img_tensor = preprocess_image(img)
        predictions = detect_objects(session, img_tensor)
        
        # Visualization (you'll need to adapt visualize_results() similarly)
        print("Visualization for ONNX model needs custom implementation")
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        # Video processing
        process_video_to_frames(
            args.input, 
            session, 
            output_dir=args.output_dir, 
            threshold=args.threshold,
            frame_interval=args.frame_interval
        )
    else:
        raise ValueError("Unsupported file format")