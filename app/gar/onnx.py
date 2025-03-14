import os
import sys
import glob
import time
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import onnxruntime as ort
import csv
from datetime import datetime

# GUI function (unchanged, omitted for brevity)
def get_user_inputs():
    root = tk.Tk()
    root.title("ONNX Detection Settings")
    
    model_path = tk.StringVar()
    source = tk.StringVar()
    thresh = tk.DoubleVar(value=0.5)
    resolution = tk.StringVar(value="Auto")
    student_id = tk.StringVar()
    
    tk.Label(root, text="Select ONNX Model:").grid(row=0, column=0, padx=5, pady=5)
    tk.Button(root, text="Browse", 
             command=lambda: model_path.set(filedialog.askopenfilename(filetypes=[("ONNX files", "*.onnx")]))).grid(row=0, column=1)
    model_entry = tk.Entry(root, textvariable=model_path)
    model_entry.grid(row=0, column=2, padx=5, pady=5)
    
    tk.Label(root, text="Select Source:").grid(row=1, column=0, padx=5, pady=5)
    source_type = ttk.Combobox(root, textvariable=source, 
                              values=["Image File", "Image Folder", "Video File", "USB Camera 0", "Picamera 0"])
    source_type.grid(row=1, column=1, padx=5, pady=5)
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
    source_entry.grid(row=1, column=2, padx=5, pady=5)
    
    tk.Label(root, text="Confidence Threshold (0-1):").grid(row=2, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=thresh).grid(row=2, column=1, padx=5, pady=5)
    
    tk.Label(root, text="Resolution (WxH or Auto):").grid(row=3, column=0, padx=5, pady=5)
    resolution_combo = ttk.Combobox(root, textvariable=resolution, 
                                  values=["Auto", "640x480", "1280x720", "1920x1080"])
    resolution_combo.grid(row=3, column=1, padx=5, pady=5)
    resolution_entry = tk.Entry(root, textvariable=resolution)
    resolution_entry.grid(row=3, column=2, padx=5, pady=5)
    
    def update_resolution_entry(*args):
        if resolution.get() == "Auto":
            resolution_entry.config(state='disabled')
        else:
            resolution_entry.config(state='normal')
    
    resolution_combo.bind('<<ComboboxSelected>>', update_resolution_entry)
    
    tk.Label(root, text="Student ID:").grid(row=4, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=student_id).grid(row=4, column=1, padx=5, pady=5)
    
    def submit():
        root.quit()
    
    tk.Button(root, text="Start Detection", command=submit).grid(row=5, column=1, pady=10)
    
    root.mainloop()
    root.destroy()
    
    return {
        'model': model_path.get(),
        'source': source.get(),
        'thresh': thresh.get(),
        'resolution': resolution.get() if resolution.get() != "Auto" else None,
        'student_id': student_id.get()
    }

# Get user inputs from GUI
args = get_user_inputs()

# Parse user inputs
model_path = args['model']
img_source = args['source']
min_thresh = args['thresh']
user_res = args['resolution']
student_id = args['student_id']

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load class names from classes.txt
with open('D:\SourceCode\ProGabage\system\data\classes.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f"Number of classes in labels: {len(labels)}")

# Load ONNX model
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [0, 3, 0, 0]

# Define default resolution
DEFAULT_MODEL_W, DEFAULT_MODEL_H = 640, 640

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
else:
    resW, resH = DEFAULT_MODEL_W, DEFAULT_MODEL_H

# Parse input source (unchanged)
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

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

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
detection_results = {}

# Pre-processing function
def preprocess(frame, target_w=DEFAULT_MODEL_W, target_h=DEFAULT_MODEL_H):
    img = cv2.resize(frame, (target_w, target_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    return img

# Post-processing function
def postprocess(outputs, conf_thres=0.5, iou_thres=0.5, img_shape=(DEFAULT_MODEL_H, DEFAULT_MODEL_W)):
    output = outputs[0][0]  # [num_attributes, num_detections] -> [14, 8400]
    print(f"Output shape: {output.shape}")
    
    # Transpose to [num_detections, num_attributes] -> [8400, 14]
    output = output.transpose(1, 0)
    print(f"Transposed output shape: {output.shape}")
    
    boxes = output[:, :4]  # [x_center, y_center, width, height]
    scores = output[:, 4]  # Confidence scores
    class_scores = output[:, 5:]  # Should be [8400, 9] for 9 classes (but we need to confirm)
    print(f"Class scores shape: {class_scores.shape}")
    print(f"Class scores sample: {class_scores[0]}")  # Debug print
    
    classes = np.argmax(class_scores, axis=1)
    max_scores = scores * class_scores.max(axis=1)
    print(f"Classes: {classes}")
    print(f"Max scores: {max_scores[:10]}")  # Debug top 10 scores

    # Convert from center-width-height to top-left-bottom-right
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_min
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_min
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x_max
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y_max

    # Scale boxes back to original image size
    h, w = img_shape
    boxes[:, [0, 2]] *= resW / DEFAULT_MODEL_W  # Scale x
    boxes[:, [1, 3]] *= resH / DEFAULT_MODEL_H  # Scale y

    mask = max_scores > conf_thres
    boxes, scores, classes = boxes[mask], max_scores[mask], classes[mask]
    print(f"Filtered classes: {classes}")

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
        return [(boxes[i], scores[i], classes[i]) for i in indices]
    return []

# Inference loop
while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images have been processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print('End of video or camera feed.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Preprocess frame
    input_tensor = preprocess(frame, DEFAULT_MODEL_W, DEFAULT_MODEL_H)

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    detections = postprocess(outputs, min_thresh, img_shape=frame.shape[:2])

    object_count = 0
    for box, conf, cls_idx in detections:
        if int(cls_idx) >= len(labels):
            print(f"Warning: Class index {cls_idx} exceeds number of labels ({len(labels)}). Skipping.")
            continue
        xmin, ymin, xmax, ymax = box.astype(int)
        classname = labels[int(cls_idx)]
        color = bbox_colors[int(cls_idx) % 10]
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
    cv2.imshow('ONNX Detection Results', frame)

    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'picamera'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Save results to CSV (unchanged)
output_dir = os.path.abspath(os.path.join(os.getcwd(), "system", "output"))
os.makedirs(output_dir, exist_ok=True)
csv_filename = f'detection_results_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.csv'
csv_path = os.path.join(output_dir, csv_filename)
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Class_ID', 'Class_Name', 'Count', 'Detection_Date', 'Student_ID']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    detection_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    for class_id, class_name in enumerate(labels):
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