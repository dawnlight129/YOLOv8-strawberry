from ultralytics import YOLO
from collections import defaultdict
import cv2
from cv2 import getTickCount, getTickFrequency

import os
# model = YOLO("weights/yolov8s.pt")
model = YOLO('runs/detect/train/weights/best.pt')
video_path = "data/videos/grow_fruits.mp4"

# 打开视频文件
cap = cv2.VideoCapture(video_path)

frame_rate_divider = 5  # 设置帧率除数
frame_count = 0  # 初始化帧计数器

counts = defaultdict(int)
object_str = ""
index = 0

# 判断文件夹是否存在
if os.path.exists("imgs"):
    print("文件夹存在")
    if len(os.listdir("imgs")) == 0:
        os.rmdir("imgs")
        print("删除成功")
        os.mkdir("imgs")

while cap.isOpened():  # 检查视频文件是否成功打开
    loop_start = getTickCount()
    ret, frame = cap.read()  # 读取视频文件中的下一帧,ret 是一个布尔值，如果读取帧成功
    if not ret:
        break

    # 每隔 frame_rate_divider 帧进行一次预测
    if frame_count % frame_rate_divider == 0:
        results = model.predict(source=frame)
        annotated_frame = results[0].plot()

        key = f"({index}): "
        index = index + 1
        for result in results:
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                counts[class_id] += 1

        object_str = object_str + ". " + key
        for class_id, count in counts.items():
            object_str = object_str + f"{count} {class_id},"
            counts = defaultdict(int)

    # 中间放自己的显示程序
    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    # 在图像左上角添加FPS文本
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # 红色
    text_position = (10, 30)  # 左上角位置

    cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)

    cv2.namedWindow("enhanced", 0)   # CV_WINDOW_NORMAL就是0
    cv2.resizeWindow("enhanced", 640, 480)    #设置长宽大小为640 * 480

    cv2.imshow('img', annotated_frame)

    path1 = "imgs/"
    path2 = ".png"
    # if(index %5 == 0):          #每5张保存一次
    #     index1 = index/5
    url = path1+ str(index)+path2
    print(url)
    is_true = cv2.imwrite(url, annotated_frame)
    print('is_true',is_true)
    # 通过按下 'q' 键退出循环

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    frame_count += 1  # 更新帧计数器

object_str = object_str.strip(',').strip('.')
print("reuslt:", object_str)

cap.release()
cv2.destroyAllWindows()