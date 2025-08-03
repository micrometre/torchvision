.PHONY: run
install:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
clean:
	rm -rf output images