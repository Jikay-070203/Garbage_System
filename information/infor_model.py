import onnx
import os

# Load model ONNX
model_path = r"D:\SourceCode\ProGabage\system\models\onnx\V11S\1\V11S.onnx"  
model = onnx.load(model_path)

# infor input
input_all = model.graph.input
input_shapes = {inp.name: [dim.dim_value for dim in inp.type.tensor_type.shape.dim] for inp in input_all}

# infor output
output_all = model.graph.output
output_shapes = {out.name: [dim.dim_value for dim in out.type.tensor_type.shape.dim] for out in output_all}

# show infor
print(" Input Information:")
for name, shape in input_shapes.items():
    print(f"- Name: {name}, Shape: {shape}")

print("\n Output Information:")
for name, shape in output_shapes.items():
    print(f"- Name: {name}, Shape: {shape}")
