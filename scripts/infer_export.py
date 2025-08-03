import onnxruntime as ort
import cv2
import numpy as np

# Load ONNX model
session = ort.InferenceSession("./weights/best4.onnx")

# Preprocess image
img = cv2.imread("1.jpg")
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)  # HWC to CHW
img = np.expand_dims(img, 0).astype(np.float32) / 255.0

# Run inference
outputs = session.run(None, {"images": img})