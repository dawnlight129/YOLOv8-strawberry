from ultralytics import YOLO
#加载模型
model = YOLO("yolov8n.pt")
# model = YOLO("weights/yolov8s.py")

# results = model("ultralytics/assets/bus.jpg")
# success = model.export(format="onnx")
model.predict('ultralytics/assets/bus.jpg', save=True, imgsz=320, conf=0.5)