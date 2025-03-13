import os
import sys
import glob
import time
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime
import torch
import onnxruntime as ort

# Create GUI for parameter selection
def get_user_inputs():
    root = tk.Tk()
    root.title("YOLO Detection Settings")
    
    model_path = tk.StringVar()
    source = tk.StringVar()
    thresh = tk.DoubleVar(value=0.5)
    resolution = tk.StringVar(value="Auto")
    student_id = tk.StringVar()
    labels_path = tk.StringVar()
    
    tk.Label(root, text="Select Model:").grid(row=0, column=0, padx=5, pady=5)
    tk.Button(root, text="Browse", 
             command=lambda: model_path.set(filedialog.askopenfilename(filetypes=[("Model files", "*.pt *.onnx")]))).grid(row=0, column=1)
    tk.Entry(root, textvariable=model_path).grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(root, text="Select Labels (.txt):").grid(row=1, column=0, padx=5, pady=5)
    tk.Button(root, text="Browse", 
             command=lambda: labels_path.set(filedialog.askopenfilename(filetypes=[("Text files", "*.txt")]))).grid(row=1, column=1)
    tk.Entry(root, textvariable=labels_path).grid(row=1, column=2, padx=5, pady=5)
    
    tk.Label(root, text="Select Source:").grid(row=2, column=0, padx=5, pady=5)
    source_type = ttk.Combobox(root, textvariable=source, 
                              values=["Image File", "Image Folder", "Video File", "USB Camera 0", "Picamera 0"])
    source_type.grid(row=2, column=1, padx=5, pady=5)
    source_type.set("USB Camera 0")
    
    def update_source_entry(*args):
        if "Camera" in source.get():
            source_entry.config(state='disabled')
        else:
            source_entry.config(state='normal')
            if source.get() == "Image File":
                path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
            elif source.get() == "Image Folder":
                path = filedialog.askdirectory()
            elif source.get() == "Video File":
                path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mov *.mp4 *.mkv *.wmv")])
            source.set(path)
    
    source_type.bind('<<ComboboxSelected>>', update_source_entry)
    source_entry = tk.Entry(root, textvariable=source)
    source_entry.grid(row=2, column=2, padx=5, pady=5)
    
    tk.Label(root, text="Confidence Threshold (0-1):").grid(row=3, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=thresh).grid(row=3, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Resolution (WxH or Auto):").grid(row=4, column=0, padx=5, pady=5)
    resolution_combo = ttk.Combobox(root, textvariable=resolution, 
                                  values=["Auto", "640x480", "1280x720", "1920x1080"])
    resolution_combo.grid(row=4, column=1, padx=5, pady=5)
    resolution_entry = tk.Entry(root, textvariable=resolution)
    resolution_entry.grid(row=4, column=2, padx=5, pady=5)
    
    def update_resolution_entry(*args):
        if resolution.get() == "Auto":
            resolution_entry.config(state='disabled')
        else:
            resolution_entry.config(state='normal')
    
    resolution_combo.bind('<<ComboboxSelected>>', update_resolution_entry)
    
    tk.Label(root, text="Student ID:").grid(row=5, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=student_id).grid(row=5, column=1, padx=5, pady=5)
    
    def submit():
        root.quit()
    
    tk.Button(root, text="Start Detection", command=submit).grid(row=6, column=1, pady=10)
    
    root.mainloop()
    root.destroy()
    
    return {
        'model': model_path.get(),
        'source': source.get(),
        'thresh': thresh.get(),
        'resolution': resolution.get() if resolution.get() != "Auto" else None,
        'student_id': student_id.get(),
        'labels_path': labels_path.get()
    }

# Function to load model based on file extension
def load_model(model_path, device):
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == '.pt':  # PyTorch model
        model = YOLO(model_path, task='detect')
        model.to(device)
        return model, 'yolo'
    
    elif ext == '.onnx':  # ONNX model
        if device.type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        return session, 'onnx'
    
    else:
        raise ValueError(f"Only .pt and .onnx formats are supported")

# Function to load labels from .txt file
def load_labels(labels_path):
    if not labels_path or not os.path.exists(labels_path):
        raise ValueError("A valid .txt file with class labels must be provided")
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    if not labels:
        raise ValueError("The .txt file is empty or contains no valid labels")
    return {i: label for i, label in enumerate(labels)}

# Function to preprocess frame for ONNX
def preprocess_onnx(frame, input_size=(640, 640)):
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

# Function to postprocess ONNX output
def postprocess_onnx(outputs, orig_shape, input_size=(640, 640), conf_thresh=0.5):
    # Handle common ONNX output formats (e.g., YOLOv8: [1, 84, num_boxes])
    output = outputs[0]  # Assume first output tensor
    if len(output.shape) == 3:  # [batch, num_features, num_boxes]
        output = output[0].T  # Transpose to [num_boxes, num_features]

    boxes = []
    scores = []
    class_ids = []
    
    for pred in output:
        # Extract confidence (assuming it's the 5th element or part of a different structure)
        if len(pred) > 5:  # Standard YOLO format: [x, y, w, h, conf, class_scores...]
            conf = pred[4]  # Confidence score
            if conf > conf_thresh:
                cx, cy, w, h = pred[:4]
                class_scores = pred[5:]
                class_id = np.argmax(class_scores)
                score = conf * class_scores[class_id]

                if score > conf_thresh:
                    # Convert to absolute coordinates
                    xmin = (cx - w/2) * orig_shape[1] / input_size[0]
                    ymin = (cy - h/2) * orig_shape[0] / input_size[1]
                    xmax = (cx + w/2) * orig_shape[1] / input_size[0]
                    ymax = (cy + h/2) * orig_shape[0] / input_size[1]
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    scores.append(score)
                    class_ids.append(class_id)
    
    return np.array(boxes), np.array(scores), np.array(class_ids)

# Get user inputs from GUI
args = get_user_inputs()

# Parse user inputs
model_path = args['model']
img_source = args['source']
min_thresh = args['thresh']
user_res = args['resolution']
student_id = args['student_id']
labels_path = args['labels_path']

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device.type.upper()}")

# Load model
model, model_type = load_model(model_path, device)

# Load labels from .txt file (mandatory)
try:
    labels = load_labels(labels_path)
except ValueError as e:
    print(f"ERROR: {str(e)}")
    sys.exit(0)

# Parse input to determine source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if "Image Folder" in img_source or os.path.isdir(img_source):
    source_type = 'folder'
elif "Image File" in img_source or (os.path.isfile(img_source) and os.path.splitext(img_source)[1] in img_ext_list):
    source_type = 'image'
elif "Video File" in img_source or (os.path.isfile(img_source) and os.path.splitext(img_source)[1] in vid_ext_list):
    source_type = 'video'
elif 'USB' in img_source:
    source_type = 'usb'
    usb_idx = 0
elif 'Picamera' in img_source:
    source_type = 'picamera'
    picam_idx = 0
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
    if not user_res:
        frame = cv2.imread(img_source)
        resH, resW = frame.shape[:2]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
    if not user_res and imgs_list:
        frame = cv2.imread(imgs_list[0])
        resH, resW = frame.shape[:2]
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    
    if not user_res:
        resW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    elif user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    if not user_res:
        config = cap.create_video_configuration(main={"format": 'RGB888'})
        resW, resH = config['main']['size']
    else:
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
detection_results = {}

# Inference loop
while True:
    t_start = time.perf_counter()

    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file.')
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read frames from the Picamera.')
            break

    orig_shape = frame.shape[:2]
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Inference
    if model_type == 'yolo':
        results = model(frame, verbose=False, device=device)
        detections = results[0].boxes
        boxes = detections.xyxy.cpu().numpy()
        scores = detections.conf.cpu().numpy()
        class_ids = detections.cls.cpu().numpy().astype(int)
    elif model_type == 'onnx':
        input_tensor = preprocess_onnx(frame)
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_tensor})
        boxes, scores, class_ids = postprocess_onnx(outputs, orig_shape, conf_thresh=min_thresh)

    object_count = 0
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i].astype(int)
        conf = scores[i]
        class_id = class_ids[i]
        if class_id not in labels:
            continue  # Skip if class_id exceeds loaded labels
        classname = labels[class_id]

        if conf > min_thresh:
            color = bbox_colors[class_id % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1
            detection_results[classname] = detection_results.get(classname, 0) + 1

    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('Detection results', frame)

    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)
    
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)
    
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Save results to CSV
output_dir = os.path.abspath(os.path.join(os.getcwd(),"system", "output"))
os.makedirs(output_dir, exist_ok=True)
csv_filename = f'detection_results_{datetime.now().strftime("%d%m%Y_%H%M%S")}.csv'
csv_path = os.path.join(output_dir, csv_filename)
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Class_ID', 'Class_Name', 'Count', 'Detection_Date', 'Student_ID']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    detection_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    for class_id, class_name in labels.items():
        count = detection_results.get(class_name, 0)
        if count > 0:
            writer.writerow({
                'Class_ID': class_id,
                'Class_Name': class_name,
                'Count': count,
                'Detection_Date': detection_date,
                'Student_ID': student_id
            })

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
print(f'Results saved to {csv_path}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
cv2.destroyAllWindows()