
import ultralytics
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

classes = ['plastic', 'mask', 'glass_bottle', 'cardboard', 'metal', 'paper']


# 1. Tải mô hình đã huấn luyện
model = YOLO(r'D:\SourceCode\Project_Tỉnh\Trash recognition\best.pt')
  # Đường dẫn đến mô hình đã huấn luyện của bạn

# 2. Định nghĩa hàm nhận diện đối tượng từ khung hình
def detect_objects_from_frame(frame, model, conf_threshold=0.8):
    # Chạy mô hình để nhận diện đối tượng
    results = model(frame)

    # Vẽ bounding box và nhãn lên khung hình
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding box
        labels = result.boxes.cls.numpy()  # Class labels
        scores = result.boxes.conf.numpy() # Confidence scores

        for box, label, score in zip(boxes, labels, scores):
            if score >= conf_threshold:
                x1, y1, x2, y2 = box
                class_name = model.names[int(label)]
                color = (0, 255, 0)  # Màu xanh lá cho bounding box

                # Vẽ bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Vẽ nhãn
                cv2.putText(frame, f'{class_name} {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# 3. Mở webcam và thực hiện nhận diện đối tượng trên từng khung hình
def run_webcam_detection(model, conf_threshold=0.8):
    cap = cv2.VideoCapture(1)  # Mở webcam, có thể thay đổi số nếu dùng nhiều webcam

    while True:
        ret, frame = cap.read()  # Đọc khung hình từ webcam
        if not ret:
            print("Error: Unable to capture image from webcam.")
            break

        frame = detect_objects_from_frame(frame, model, conf_threshold)  # Nhận diện đối tượng từ khung hình

        cv2.imshow('Webcam Object Detection', frame)  # Hiển thị khung hình

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Giải phóng webcam
    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ

# 4. Chạy hàm nhận diện đối tượng từ webcam
run_webcam_detection(model)