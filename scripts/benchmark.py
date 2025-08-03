from ultralytics import YOLO

model = YOLO("./weights/best4.onnx")  # or 'best.engine', 'best.tflite'
model.predict("1.jpg", mode="benchmark" )  # Runs speed test