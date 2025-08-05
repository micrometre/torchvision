# License Plate Segmentation Training

This project implements license plate detection using **semantic segmentation** with a DeepLabV3 model. Instead of bounding boxes, it predicts pixel-level masks for license plates.

## Overview

- **Model**: DeepLabV3 with ResNet101 backbone
- **Task**: Semantic segmentation (pixel-level classification)
- **Output**: Binary masks where white pixels indicate license plate regions
- **Framework**: PyTorch + TorchVision

## Project Structure

```
├── train.py                 # Main training script
├── model.py                 # Model definition (DeepLabV3)
├── prepare_dataset.py       # Dataset preparation utilities
├── make_predictions_v1.py   # Inference script
├── basic_test.py           # Object detection example (different approach)
├── requirements.txt         # Python dependencies
├── Makefile                # Common commands
├── dataset/                # Training data
│   ├── train/
│   │   ├── images/         # Training images
│   │   └── masks/          # Training masks (binary, white=plate)
│   └── val/
│       ├── images/         # Validation images
│       └── masks/          # Validation masks
├── checkpoints/            # Model checkpoints (created during training)
└── outputs/               # Prediction outputs
```

## Quick Start

### 1. Install Dependencies

```bash
make install
# or
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Your dataset should have:
- **Images**: Photos containing license plates
- **Masks**: Binary masks where white pixels (255) mark license plate regions

Expected structure:
```
raw_data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png      # or image1.jpg.png
    ├── image2.png
    └── ...
```

### 3. Verify and Split Dataset

```bash
# Inspect your masks
python prepare_dataset.py --inspect --masks-dir raw_data/masks

# Verify dataset integrity
python prepare_dataset.py --verify --images-dir raw_data/images --masks-dir raw_data/masks

# Split into train/validation (80/20 split)
python prepare_dataset.py --split --source-images raw_data/images --source-masks raw_data/masks
```

### 4. Train the Model

```bash
# Basic training
make train

# Or with custom parameters
python train.py --epochs 100 --batch-size 8 --learning-rate 0.0001

# Resume from checkpoint
make resume
```

### 5. Make Predictions

```bash
# Single image
python make_predictions_v1.py --model checkpoints/model.pth --image test_image.jpg

# Batch processing
python make_predictions_v1.py --model checkpoints/model.pth --folder test_images/
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 4 | Batch size (reduce if GPU memory issues) |
| `--learning-rate` | 0.0001 | Learning rate |
| `--image-size` | 512 | Input image size (512x512) |
| `--freeze-backbone` | False | Freeze ResNet101 backbone weights |

## Understanding the Training Process

### Data Flow

1. **Input**: RGB images (512x512)
2. **Preprocessing**: Normalization with ImageNet stats
3. **Model**: DeepLabV3 with ResNet101 backbone
4. **Output**: Single-channel probability map
5. **Loss**: Combined BCE + Dice loss
6. **Metrics**: IoU (Intersection over Union)

### Loss Function

The model uses a **Combined Loss** that balances:
- **Binary Cross-Entropy (60%)**: Pixel-wise classification loss
- **Dice Loss (40%)**: Handles class imbalance, focuses on overlap

### Monitoring Training

Training outputs:
- **Loss**: Lower is better
- **IoU**: Higher is better (0-1 scale, 1 = perfect overlap)
- **Learning curves**: Saved as `checkpoints/training_history.png`

### Checkpoints

- `checkpoints/model.pth`: Best model (highest validation IoU)
- `checkpoints/checkpoint_epoch_X.pth`: Regular checkpoints

## Data Requirements

### Image Format
- **Format**: JPG, PNG
- **Size**: Any size (automatically resized to 512x512)
- **Content**: Clear license plate visibility

### Mask Format
- **Format**: PNG (grayscale)
- **Values**: Binary (0 = background, 255 = license plate)
- **Size**: Should match corresponding image
- **Naming**: 
  - `image1.jpg` → `image1.png`
  - Or `image1.jpg` → `image1.jpg.png`

### Creating Masks

You can create masks using:
1. **Manual annotation**: Tools like LabelMe, CVAT, or Photoshop
2. **Semi-automatic**: Use a pre-trained model for initial masks, then refine manually
3. **Synthetic data**: Generate artificial license plates with known masks

## Common Issues & Solutions

### Memory Issues
```bash
# Reduce batch size
python train.py --batch-size 2

# Reduce image size
python train.py --image-size 256
```

### Poor Training Performance
- **Check masks**: Use `--inspect` to verify mask format
- **Data augmentation**: Already included (rotation, flip, color jitter)
- **Learning rate**: Try 0.0001 to 0.001
- **Freeze backbone**: Use `--freeze-backbone` for small datasets

### Validation Loss Not Decreasing
- **Overfitting**: Reduce learning rate, add more data
- **Data quality**: Verify image-mask alignment
- **Class imbalance**: Adjust loss weights in `train.py`

## Model Architecture

```
Input (3, 512, 512)
    ↓
ResNet101 Backbone (frozen/trainable)
    ↓
DeepLabV3 Head
    ↓
Output (1, 512, 512) - probability map
    ↓
Sigmoid → Binary mask
```

## Performance Tips

1. **Start small**: Begin with 10-20 epochs to verify training works
2. **Monitor closely**: Watch for overfitting (train IoU >> val IoU)
3. **Data quality**: Good masks are crucial for segmentation
4. **Augmentation**: Helps with small datasets
5. **Transfer learning**: Uses ImageNet pre-trained weights

## Extending the Project

### Adding New Loss Functions
Edit `train.py` and modify the `CombinedLoss` class.

### Custom Data Augmentation
Modify transforms in `get_transforms()` function.

### Multi-class Segmentation
Change `outputchannels=1` to number of classes in `model.py`.

### Different Backbone
Replace ResNet101 with other backbones in `model.py`.

## Comparison with Object Detection

This project uses **segmentation** instead of **object detection**:

| Approach | Output | Use Case |
|----------|--------|----------|
| Segmentation (this) | Pixel masks | Precise shape, multiple plates |
| Object Detection (`basic_test.py`) | Bounding boxes | General objects, faster |

## Files Overview

- **`train.py`**: Complete training pipeline with data loading, training loop, validation, and checkpointing
- **`model.py`**: DeepLabV3 model definition
- **`prepare_dataset.py`**: Dataset utilities for verification and splitting
- **`make_predictions_v1.py`**: Inference script for trained models
- **`basic_test.py`**: Example of object detection approach (different method)

Start with the training script and adjust parameters based on your dataset size and available compute resources!



🚀 Performance Expectations

For license plate segmentation with DeepLabV3:

Good IoU: 0.7+ (70%+ pixel overlap)

Excellent IoU: 0.8+ (80%+ pixel overlap)

Ycurrent: 0.137 (13.7%) - needs more data/training


🔄 Alternative Architectures

experiment with other models:

- U-Net: Lighter, faster training
- PSPNet: Different pooling strategy
- Faster R-CNN: Object detection (bounding boxes)
- YOLO: Real-time detection