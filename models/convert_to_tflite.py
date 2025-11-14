# convert_to_onnx.py
import torch
import torchvision
import os

# 1. 加载剪枝模型
model = torchvision.models.mobilenet_v2(num_classes=10)
model.load_state_dict(torch.load("pruned_mobilenetv2.pth", map_location="cpu"))
model.eval()

# 2. 导出为 ONNX（动态 batch，32x32 输入）
dummy_input = torch.randn(1, 3, 32, 32)
onnx_path = "pruned_mobilenetv2.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    export_params=True,
    do_constant_folding=True
)

print(f"✅ ONNX 模型已保存: {onnx_path}")
print(f"  大小: {os.path.getsize(onnx_path) / (1024**2):.2f} MB")