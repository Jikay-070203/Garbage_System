import os
import sys
import argparse
import glob
import time
import torch
from tkinter import Tk, filedialog  # Thêm thư viện tkinter để tạo GUI

import cv2
import numpy as np
from ultralytics import YOLO

# Hàm để chọn file hoặc thư mục
def select_file(title, filetypes):
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của tkinter
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def select_folder(title):
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của tkinter
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

# Hàm để chọn model và input đầu vào
def select_model_and_source():
    # Chọn file model
    model_path = select_file("Chọn file model (.pt)", [("Model files", "*.pt")])
    if not model_path:
        print("Không có file model được chọn. Thoát chương trình.")
        sys.exit(0)

    # Chọn input đầu vào (ảnh, video, thư mục ảnh, hoặc camera)
    input_type = input("Chọn loại input (1: Ảnh, 2: Video, 3: Thư mục ảnh, 4: Camera): ")
    if input_type == '1':
        source = select_file("Chọn file ảnh", [("Image files", "*.jpg *.jpeg *.png *.bmp")])
    elif input_type == '2':
        source = select_file("Chọn file video", [("Video files", "*.mp4 *.avi *.mov *.mkv")])
    elif input_type == '3':
        source = select_folder("Chọn thư mục ảnh")
    elif input_type == '4':
        source = 'camera'  # Sử dụng camera làm đầu vào
    else:
        print("Lựa chọn không hợp lệ. Thoát chương trình.")
        sys.exit(0)

    if not source and input_type != '4':  # Camera không cần chọn file
        print("Không có input được chọn. Thoát chương trình.")
        sys.exit(0)

    return model_path, source

# Chọn model và input đầu vào
model_path, img_source = select_model_and_source()

# Các tham số cố định (có thể thay đổi nếu cần)
min_thresh = 0.5
user_res = None
record = False

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # Chuyển model sang GPU hoặc CPU
print(f'Using {device.upper()} for inference.')

# Parse input to determine if image source is a file, folder, video, or camera
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if img_source == 'camera':
    source_type = 'camera'
elif os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video', 'camera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)

    # Set up recording
    record_name = 'chai.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
elif source_type == 'camera':
    cap = cv2.VideoCapture(1)  # Sử dụng camera mặc định (index 0)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Create a resizable window
cv2.namedWindow('YOLO detection results', cv2.WINDOW_NORMAL)

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':  # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1

    elif source_type == 'video' or source_type == 'camera':  # If source is a video or camera, load next frame
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file or camera is disconnected. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()  # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int)  # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > 0.5:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Draw label text

            # Basic example: count the number of objects in the image
            object_count = object_count + 1

    # Calculate and draw framerate (if using video or camera source)
    if source_type == 'video' or source_type == 'camera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)  # Draw framerate

    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)  # Draw total number of detected objects
    cv2.putText(frame, f'Device: {device.upper()}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)  # Hiển thị thiết bị đang sử dụng
    cv2.imshow('YOLO detection results', frame)  # Display image
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'camera':
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png', frame)
    elif key == ord('w') or key == ord('W'):  # Press 'w' to toggle between GPU and CPU
        if device == 'cuda':
            device = 'cpu'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)  # Chuyển model sang thiết bị mới
        print(f'Switched to {device.upper()} for inference.')

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'camera':
    cap.release()
if record: recorder.release()
cv2.destroyAllWindows()