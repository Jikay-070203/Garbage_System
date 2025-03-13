from ultralytics import YOLO
import os
import shutil

# path save
output_dir = r"D:\SourceCode\ProGabage\system\model\onnx"
os.makedirs(output_dir, exist_ok=True)  

# Load mô hình 
model_path = r"D:\SourceCode\ProGabage\system\model\pt\v11S\model\best.pt"
model = YOLO(model_path)

# export to ONNX
model.export(format="onnx", dynamic=True, opset=12, simplify=True)


onnx_file = os.path.join(os.path.dirname(model_path), "best.onnx")

# Kiểm tra 
if os.path.exists(onnx_file):
    output_path = os.path.join(output_dir, "V11S.onnx")
    shutil.move(onnx_file, output_path)
    print(f" Model đã được lưu vào: {output_path}")
else:
    print(f"Không tìm thấy file {onnx_file} sau khi export.")

print("Có thể test model convert to onnx tại website :https://netron.app/")