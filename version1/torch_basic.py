import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# Load a pretrained Faster R-CNN model
#model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
#model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn( weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

model.eval()

# Load your image
image_path = "1.jpg"  # Change to your image path
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(image_tensor)

# Draw boxes for detected objects with score > 0.5
draw = ImageDraw.Draw(image)
for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
    if score > 0.5:
        box = box.tolist()
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label.item()}:{score:.2f}", fill="red")

# Save the result
image.save("result.jpg")
print("Detection result saved as result.jpg")