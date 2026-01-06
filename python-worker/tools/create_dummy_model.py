import torch
import torch.nn as nn
import os

class SimpleCloudModel(nn.Module):
    def __init__(self):
        super(SimpleCloudModel, self).__init__()
        # A simple convulation layer
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


model = SimpleCloudModel()
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
cur_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(cur_dir, "..", "models")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cloud_mask.onnx")


torch.onnx.export(
    model,
    dummy_input, 
    output_path,
    input_names=["input_image"],
    output_names=["output_mask"],
    opset_version=17,
    dynamic_axes={
        "input_image": {0: "batch_size"},
        "output_mask": {0: "batch_size"}
    })

print(f"✅ Dummy model created successfully at: {output_path}")
