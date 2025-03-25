🗑️ Waste Classification Triton API  
FastAPI-based inference server for waste classification using YOLO models converted to ONNX and deployed via NVIDIA Triton Inference Server. Supports both CPU and GPU runtime. Designed for scalable, production-ready deployment (Docker).

🚀 **Features**  
🔍 Real-time waste classification using YOLO.onnx  
⚡ Fast inference with ONNX models + Triton Server  
📡 API-ready with FastAPI + Triton gRPC/HTTP  
☁️ Cloud-ready: Docker, Compose, Kubernetes, Helm  
📤 API Endpoint: `POST /predict`

🏗️ **Project Structure**
```
.
├── app/                    # FastAPI app source code
│   ├── app.py              # API Endpoint
│   └── triton_clients/     # Triton gRPC clients for YOLO ONNX model
├── models/                 # ONNX model repository for Triton
├── Dockerfile              # Docker build config
├── docker-compose.yml      # Multi-container deployment (CPU/GPU)
├── requirements.txt        # Python dependencies
│   ├── triton_clients/     # Triton gRPC or HTTP clients for model inference
│   │   ├── yolo_client.py          # YOLO model inference client
│   │   ├── vae_encoder_client.py   # VAE Encoder client
│   │   ├── vae_decoder_client.py   # VAE Decoder client
│   │   ├── unet_client.py          # UNet client
├── charts/                 # Helm chart for Kubernetes deployment
│   └── waste-classifier/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
├── k8s/                    # Kubernetes manifests
│   ├── waste-classifier-deploy.yaml
│   └── waste-classifier-service.yaml
└── README.md               
```

🐳 **Setup with Docker**  
```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v /path/to/model:/models nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

🚀 **Run FastAPI Server**  
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

🔨 **Build & Run with Docker**  
```bash
docker build -t waste-classifier-triton .
docker run --gpus all -p 8000:8000 waste-classifier-triton
```

