.PHONY: run train install clean create-annotations list-images setup-dataset

install:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip3 install -r requirements.txt
	pip3 install pillow exifread

train:
	python3 train_rcnn.py \
		--train-img-dir dataset/train/images \
		--val-img-dir dataset/valid/images \
		--train-ann-file dataset/train/annotations.json \
		--val-ann-file dataset/valid/annotations.json \
		--output-dir outputs \
		--batch-size 2 \
		--epochs 1 \
		--lr 0.005

# Special training configuration optimized for license plate detection
train-license-plates:
	python3 train_rcnn.py \
		--train-img-dir dataset/train/images \
		--val-img-dir dataset/valid/images \
		--train-ann-file dataset/train/annotations.json \
		--val-ann-file dataset/valid/annotations.json \
		--output-dir outputs/license_plates \
		--batch-size 1 \
		--img-size 320 \
		--epochs 10 \
		--optimize-cpu 

# Create annotation files with enhanced license plate features
create-annotations-train:
	python3 anotation.py dataset/train/images -o dataset/train/annotations.json -c license_plate vehicle car truck motorcycle -r

create-annotations-valid:
	python3 anotation.py dataset/valid/images -o dataset/valid/annotations.json -c license_plate vehicle car truck motorcycle -r

# Create all annotation files
create-annotations: create-annotations-train create-annotations-valid

# Create annotation files with dummy license plate annotations (useful as starting point)
create-dummy-annotations-train:
	python3 anotation.py dataset/train/images -o dataset/train/annotations.json -c license_plate vehicle --generate-dummy -r

create-dummy-annotations-valid:
	python3 anotation.py dataset/valid/images -o dataset/valid/annotations.json -c license_plate vehicle --generate-dummy -r

create-dummy-annotations: create-dummy-annotations-train create-dummy-annotations-valid

# List images in directories without creating annotations
list-images-train:
	python3 anotation.py dataset/train/images --list-images -r

list-images-valid:
	python3 anotation.py dataset/valid/images --list-images -r

list-images: list-images-train list-images-valid

# Setup dataset directory structure
setup-dataset:
	mkdir -p dataset/train/images
	mkdir -p dataset/valid/images
	@echo "Dataset directory structure created at dataset/"
	@echo "Place training images in dataset/train/images/"
	@echo "Place validation images in dataset/valid/images/"

# Run inference on test images
inference:
	python3 detect.py \
		--model-path outputs/license_plates/model_final.pth \
		--img-dir test_images \
		--output-dir results \
		--threshold 0.5

# Evaluate model performance
evaluate:
	python3 evaluate.py \
		--model-path outputs/license_plates/model_final.pth \
		--test-img-dir dataset/test/images \
		--test-ann-file dataset/test/annotations.json

clean:
	rm -rf output images outputs output results