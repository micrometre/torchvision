.PHONY: run 
# Installation and setup
install:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	pip3 install -r requirements.txt
	pip3 install pillow exifread

prediction:
	python make_predictions.py  
clean:
	rm -rf outputs