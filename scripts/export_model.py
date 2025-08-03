from ultralytics import YOLO

# Load your trained model
model = YOLO("./weights/best4.pt")

# Export to ONNX (recommended for most cases)
model.export(format="onnx")  # Creates 'best.onnx'