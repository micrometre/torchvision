.PHONY: install train prediction clean

create_dataset:
	python create__dataset.py  --folder anpr_images
inspect_masks:
	python prepare_dataset.py --inspect --masks-dir raw_data/masks
verify_dataset:
	python prepare_dataset.py --verify --images-dir raw_data/images --masks-dir raw_data/masks
split: 
	python prepare_dataset.py --split --source-images raw_data/images 
# Installation and setup
install:
	pip install -r requirements.txt

# Training
train:
	python train.py --epochs 5 --batch-size 2 --learning-rate 0.0001

# Small dataset training (handles very few samples)
train-small:
	python train.py --epochs 10 --batch-size 1 --learning-rate 0.001 --image-size 256

# Resume training from checkpoint
resume:
	python train.py --resume checkpoints/model.pth

# Make predictions
prediction:
	python make_predictions_v1.py --model checkpoints/model.pth --image your_image.jpg

# Clean generated files
clean:
	rm -rf outputs dataset checkpoints raw_data