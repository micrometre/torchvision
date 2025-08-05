import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Load an example image (replace with your own image path)
image_path = "test_images/picture1.jpg"
image = read_image(image_path)

# Preprocess the image
image_float = image.float() / 255.0  # Convert to float and scale to [0,1]

# Perform inference
with torch.no_grad():
    predictions = model([image_float])

# Get the predictions for the first (and only) image
pred = predictions[0]

# Filter predictions with confidence > 0.5
confidence_threshold = 0.5
high_conf_idx = pred['scores'] > confidence_threshold
boxes = pred['boxes'][high_conf_idx]
labels = pred['labels'][high_conf_idx]
scores = pred['scores'][high_conf_idx]

# Convert box coordinates to integers
boxes = boxes.int()

# Draw bounding boxes on the image
# Note: The model uses COCO labels (80 classes)
result_image = draw_bounding_boxes(image, boxes, width=3)

# Convert to PIL image and display
plt.imshow(to_pil_image(result_image))
plt.axis('off')
plt.show()

# Print detection information
print(f"Detected {len(boxes)} objects:")
for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
    print(f"Object {i+1}:")
    print(f"  Class: {label.item()} (score: {score.item():.2f})")
    print(f"  Box coordinates: {box.tolist()}")