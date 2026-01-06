import torch
import torchvision
import os
from torch import nn

# 1. Wrapper to clean FCN output
class CleanFCN(nn.Module):
    def __init__(self):
        super(CleanFCN, self).__init__()
        # Load FCN ResNet50 (Fully Pre-trained on COCO dataset)
        print("[AI] Loading FCN-ResNet50 weights (Fully Trained)...")
        weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        self.model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
            
    def forward(self, x):
        # Return only the mask, ignore aux output
        return self.model(x)['out']

# 2. Initialize
model = CleanFCN()
model.eval()

# 3. Dummy Input
dummy_input = torch.randn(1, 3, 256, 256)

# 4. Paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_script_dir, "..", "models")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cloud_mask.onnx")

# 5. Export with DYNAMIC AXES (Crucial for padding!)
print(f"[AI] Exporting to {output_path}...")
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input_image"],
    output_names=["output_mask"],
    dynamic_axes={
        "input_image": {0: "batch_size", 2: "height", 3: "width"}, # Allow 256, 288, etc.
        "output_mask": {0: "batch_size", 2: "height", 3: "width"}
    },
    opset_version=18
)

print("✅ FCN-ResNet50 (Fully Trained) exported successfully!")