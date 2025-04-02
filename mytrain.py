from ultralytics import YOLO
import torch

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" #总是报显存不足的问题，是因为碎片没完全释放
if hasattr(torch.cuda, 'empty_cache'):
   torch.cuda.empty_cache()


if __name__ == '__main__':
    # 代码

    # Load a model
    model = YOLO('ultralytics/cfg/models/v8/myyolov8.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='ultralytics/cfg/datasets/mycoco128.yaml', epochs=40)
    results = model.val()