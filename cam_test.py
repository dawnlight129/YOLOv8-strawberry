import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
import os
import time

# 加载 YOLOv8 模型
model = YOLO(r"E:\YOLOV8\ultralytics\runs\detect\train2\weights\best.pt")

# 获取摄像头内容，参数 0 表示使用默认的摄像头
cap = cv2.VideoCapture(0)
index = 0

# 确保保存图像的目录存在
save_dir = "imgs/imgpath"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

while cap.isOpened():

    loop_start = getTickCount()
    success, frame = cap.read()  # 读取摄像头的一帧图像

    if success:
        results = model.predict(source=frame)  # 对当前帧进行目标检测并显示结果
        annotated_frame = results[0].plot()
        index += 1  # 图片数+1

        # 中间放自己的显示程序
        loop_time = getTickCount() - loop_start
        total_time = loop_time / getTickFrequency()
        FPS = int(1 / total_time)

        # 在图像左上角添加FPS文本
        fps_text = f"FPS: {FPS:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 255)  # 红色
        text_position = (10, 30)  # 左上角位置

        cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
        cv2.imshow('img', annotated_frame)

        # 使用时间戳和索引作为文件名
        timestamp = int(time.time())  # 获取当前时间戳
        file_name = f"{timestamp}_{index}.png"
        file_path = os.path.join(save_dir, file_name)

        # 保存图像
        cv2.imwrite(file_path, annotated_frame)

    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口
