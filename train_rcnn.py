# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image

#from inference import load_model  # assuming it has model loading logic


# In inference.py
def load_model():
    model = models.resnet18(pretrained=False)
    model.load_state_dict(torch.load("./weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
    model.eval()
    return model

# 1. Define Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append((os.path.join(cls_dir, img_name), label))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load Datasets
train_dataset = CustomImageDataset(root_dir="dataset/train", transform=transform)
val_dataset = CustomImageDataset(root_dir="dataset/valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. Load Pre-trained Model (from inference.py or define here)
# Option A: Use model from inference.py
# Make sure inference.py has a function like `get_model()` or `load_model()`

# Example: if inference.py has `model = load_model()`

# WARNING: Make sure load_model() returns the actual model object
model = load_model()

# If it's a classifier like ResNet, modify the final layer
num_classes = len(train_dataset.classes)
if hasattr(model, 'fc'):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif hasattr(model, 'classifier'):
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
else:
    # Replace last layer manually
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 10

# 6. Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"Validation Acc: {val_acc:.2f}%")

    scheduler.step()

# 7. Save Fine-tuned Model
torch.save(model.state_dict(), "fine_tuned_model.pth")
print("Model saved as fine_tuned_model.pth")