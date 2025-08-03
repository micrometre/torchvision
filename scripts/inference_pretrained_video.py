
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import argparse
import os
from tqdm import tqdm

def load_model():
    # Load with new weights parameter
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    # Option 1: Use smaller backbone (MobileNet instead of ResNet50)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn( weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1)
    

    model.eval()
    return model

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return img, transform(img).unsqueeze(0)

def detect_objects(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)
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

def process_video_to_frames(video_path, model, output_dir="output", threshold=0.5, frame_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = transform(pil_img).unsqueeze(0)
                
                with torch.no_grad():
                    predictions = model(img_tensor)
                
                pred = predictions[0]
                boxes = pred['boxes'][pred['scores'] > threshold].cpu().numpy()
                labels = pred['labels'][pred['scores'] > threshold].cpu().numpy()
                scores = pred['scores'][pred['scores'] > threshold].cpu().numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Safe label lookup with fallback
                    label_text = f"class_{label}: {score:.2f}"  # Default if label out of range
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
    print(f"Original frame count: {total_frames}, Processed every {frame_interval} frame(s)")

def visualize_results(image, predictions, threshold=0.5):
    pred = predictions[0]
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    boxes = pred['boxes'][pred['scores'] > threshold].cpu().numpy()
    labels = pred['labels'][pred['scores'] > threshold].cpu().numpy()
    scores = pred['scores'][pred['scores'] > threshold].cpu().numpy()
    
    coco_labels = get_coco_labels()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                               edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Safe label display
        label_text = f"class_{label}: {score:.2f}"
        if 0 <= label < len(coco_labels):
            label_text = f"{coco_labels[label]}: {score:.2f}"
            
        ax.text(x1, y1-10, label_text,
               color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with Faster R-CNN")
    parser.add_argument("--input", type=str, required=True, help="Input image or video file path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output frames")
    parser.add_argument("--frame_interval", type=int, default=29, 
                       help="Process every N frames (e.g., 2=every other frame, 24=process at 1fps for 24fps video)")
    args = parser.parse_args()
    
    model = load_model()
    
    if args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
        image, image_tensor = load_image(args.input)
        predictions = detect_objects(model, image_tensor)
        visualize_results(image, predictions, args.threshold)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video_to_frames(
            args.input, 
            model, 
            output_dir=args.output_dir, 
            threshold=args.threshold,
            frame_interval=args.frame_interval
        )
    else:
        raise ValueError("Unsupported file format. Please provide an image (.jpg, .png) or video (.mp4, .avi)")