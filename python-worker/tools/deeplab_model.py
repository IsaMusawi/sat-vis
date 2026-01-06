import torch
import os

# 1. Load DeepLabV3 (Fully Pre-trained on COCO)
print("[AI] Downloading DeepLabV3 ResNet50 (Pre-trained)...")
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# 2. Define Dummy Input
dummy_input = torch.randn(1, 3, 256, 256)

# 3. Define Paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_script_dir, "..", "models")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cloud_mask.onnx")

# 4. Export
print(f"[AI] Exporting to {output_path}...")
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input_image"],
    output_names=["output_mask"], # DeepLab output is a dict, we handle this in model.py
    dynamic_axes={
        "input_image": {0: "batch_size"},
        "output_mask": {0: "batch_size"}
    },
    opset_version=18
)

print("✅ Fully Trained DeepLabV3 exported!")