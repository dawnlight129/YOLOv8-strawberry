from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model.predict('data/dataset/images/val',show=True,save=True)  # predict on an image