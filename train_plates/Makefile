.PHONY: install train prediction clean

# Installation and setup
install:
	pip install -r requirements.txt

# Training
train:
	python train.py --epochs 50 --batch-size 4 --learning-rate 0.0001

# Resume training from checkpoint
resume:
	python train.py --resume checkpoints/model.pth

# Make predictions
prediction:
	python make_predictions_v1.py

# Clean generated files
clean:
	rm -rf outputs dataset checkpoints