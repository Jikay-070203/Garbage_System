import os
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import torch
from datetime import datetime
import io

app = FastAPI()

# Load YOLO model
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = r"D:\SourceCode\ProGabage\system\models\pt\v11L\model\best.pt"  
    model = YOLO(model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

@app.post("/detect/")
async def detect(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    resolution: str = Form("Auto"),
):
    try:
        # Read the uploaded file
        file_content = await file.read()
        file_bytes = np.frombuffer(file_content, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize frame if resolution is specified
        if resolution != "Auto":
            resW, resH = map(int, resolution.split('x'))
            frame = cv2.resize(frame, (resW, resH))

        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Draw bounding boxes and labels
        for detection in detections:
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            class_id = int(detection.cls.item())
            class_name = model.names[class_id]
            conf = detection.conf.item()

            if conf > threshold:
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{class_name}: {conf:.2f}'
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode the frame to JPEG format
        _, encoded_frame = cv2.imencode(".jpg", frame)
        frame_bytes = encoded_frame.tobytes()

        # Return the image with detections
        return StreamingResponse(io.BytesIO(frame_bytes), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/status/")
async def status():
    return {"status": "running", "device": str(model.device)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)