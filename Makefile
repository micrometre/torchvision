.PHONY: run train install clean create-annotations list-images

install:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip3 install -r requirements.txt

train:
	python3 train_rcnn.py \
		--train-img-dir dataset/train/images \
		--val-img-dir dataset/valid/images \
		--train-ann-file dataset/train/annotations.json \
		--val-ann-file dataset/valid/annotations.json \
		--output-dir outputs \
		--batch-size 2 \
		--epochs 10 \
		--lr 0.005

# Create annotation files for training and validation datasets
create-annotations-train:
	python3 anotation.py dataset/train/images -o dataset/train/annotations.json -c person car bicycle

create-annotations-valid:
	python3 anotation.py dataset/valid/images -o dataset/valid/annotations.json -c person car bicycle

# Create all annotation files
create-annotations: create-annotations-train create-annotations-valid

# List images in directories without creating annotations
list-images-train:
	python3 anotation.py dataset/train/images --list-images

list-images-valid:
	python3 anotation.py dataset/valid/images --list-images

clean:
	rm -rf output images outputs