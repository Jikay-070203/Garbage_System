import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import os

# Hàm xử lý sự kiện chuột
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Tải tên các lớp từ COCO
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

# Tải model YOLO
model = YOLO(r"D:\SourceCode\ProGabage\promain\yolov11\detect_track_ID\best.pt")

# Mở video (sử dụng webcam hoặc file video khác)
cap = cv2.VideoCapture(0)

# Khai báo các biến cần thiết
count = 0
object_count = []  # Khởi tạo danh sách để theo dõi các ID đối tượng đã đếm
roi_area = [(222, 118), (194, 337), (799, 300), (728, 112)]  # Vùng quan tâm

while True:
    # Đọc một khung hình từ video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Chạy YOLO để theo dõi đối tượng
    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
        confidences = results[0].boxes.conf.cpu().tolist()
        masks = results[0].masks.xy if results[0].masks is not None else None

        overlay = frame.copy()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Vẽ bounding box, ID theo dõi và lớp
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

            # Kiểm tra nếu đối tượng thuộc vùng quan tâm (ROI)
            result = cv2.pointPolygonTest(np.array(roi_area, np.int32), (cx, cy), False)
            if result >= 0 and track_id not in object_count:
                object_count.append(track_id)

            # Xử lý mặt nạ nếu có
            if masks is not None and len(masks) > 0:
                mask = np.array(masks[class_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [mask], color=(0, 0, 255))

        # Thêm lớp overlay để hiển thị mặt nạ
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Hiển thị số lượng đối tượng đếm được
    o_counter = len(object_count)
    cvzone.putTextRect(frame, f'Object Count: {o_counter}', (50, 60), 2, 2)

    # Vẽ vùng quan tâm (ROI)
    cv2.polylines(frame, [np.array(roi_area, np.int32)], True, (255, 0, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
