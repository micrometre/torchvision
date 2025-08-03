    
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model():
    # Load with new weights parameter
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    model.eval()
    return model

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return img, transform(img).unsqueeze(0)

def detect_objects(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)

def visualize_results(image, predictions, threshold=0.5):
    pred = predictions[0]
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    boxes = pred['boxes'][pred['scores'] > threshold].cpu().numpy()
    labels = pred['labels'][pred['scores'] > threshold].cpu().numpy()
    scores = pred['scores'][pred['scores'] > threshold].cpu().numpy()
    
    coco_labels = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                               edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-10, f"{coco_labels[label]}: {score:.2f}",
               color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model = load_model()
    image, image_tensor = load_image("1.jpg")
    predictions = detect_objects(model, image_tensor)
    visualize_results(image, predictions)

