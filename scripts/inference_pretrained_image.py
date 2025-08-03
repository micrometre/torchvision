import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_and_preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Apply transformations and add a batch dimension
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

def load_pretrained_model():
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)
    model.eval()
    return model

def detect_objects(model, image_tensor):
    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

def visualize_detection(image, predictions, confidence_threshold=0.5):
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
    
    # COCO class labels (81 classes including background)
    coco_labels = [
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
        label_text = f"{coco_labels[label]}: {score:.2f}"
        ax.text(
            x1, y1 - 10, label_text,
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
        ) 
    plt.axis('off')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Replace with your image path
    image_path = "1.jpg"  # or use a URL with urllib.request.urlretrieve
    
    # Load and preprocess image
    image, image_tensor = load_and_preprocess_image(image_path)
    
    # Load model
    model = load_pretrained_model()
    
    # Run detection
    predictions = detect_objects(model, image_tensor)
    
    # Visualize results
    visualize_detection(image, predictions)