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

# Create GUI for parameter selection
def get_user_inputs():
    root = tk.Tk()
    root.title("YOLO Detection Settings")
    
    # Variables to store selections
    model_path = tk.StringVar()
    source = tk.StringVar()
    thresh = tk.DoubleVar(value=0.5)
    resolution = tk.StringVar(value="Auto")
    student_id = tk.StringVar()
    
    # Model selection
    tk.Label(root, text="Select YOLO Model:").grid(row=0, column=0, padx=5, pady=5)
    tk.Button(root, text="Browse", 
             command=lambda: model_path.set(filedialog.askopenfilename(filetypes=[("Model files", "*.pt")]))).grid(row=0, column=1)
    model_entry = tk.Entry(root, textvariable=model_path)
    model_entry.grid(row=0, column=2, padx=5, pady=5)
    
    # Source selection
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
    
    # Confidence threshold
    tk.Label(root, text="Confidence Threshold (0-1):").grid(row=2, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=thresh).grid(row=2, column=1, padx=5, pady=5)
    
    # Resolution
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
    
    # Student ID
    tk.Label(root, text="Student ID:").grid(row=4, column=0, padx=5, pady=5)
    tk.Entry(root, textvariable=student_id).grid(row=4, column=1, padx=5, pady=5)
    
    # Submit button
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

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Determine device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device.type.upper()}")

# Load the model into memory with specified device
model = YOLO(model_path, task='detect')
model.to(device)  # Move model to the selected device
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
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

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Load or initialize image source and get source resolution if using Auto
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

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
detection_results = {}  # Dictionary to store detection counts per class

# Begin inference loop
while True:
    t_start = time.perf_counter()

    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
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

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference on the selected device
    results = model(frame, verbose=False, device=device)
    detections = results[0].boxes

    object_count = 0

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()  # Move to CPU for drawing
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count = object_count + 1
            detection_results[classname] = detection_results.get(classname, 0) + 1

    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results',frame)

    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)
    
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png',frame)
    
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Create output directory one level up if it doesn't exist
output_dir = os.path.abspath(os.path.join(os.getcwd(),"system", "output"))
os.makedirs(output_dir, exist_ok=True)

# Save results to CSV in the output directory
csv_filename = f'detection_results_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.csv'
csv_path = os.path.join(output_dir, csv_filename)
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Class_ID', 'Class_Name', 'Count', 'Detection_Date', 'Student_ID']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    detection_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    for class_id, class_name in labels.items():
        count = detection_results.get(class_name, 0)
        if count > 0:  # Only write rows for detected classes
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