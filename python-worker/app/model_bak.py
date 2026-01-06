import onnxruntime as ort
import numpy as np
import io
from PIL import Image
import os

class CloudMaskModel:
    def __init__(self, model_path):
        print(f"[MODEL] Loading ONNX model from {model_path}...")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("[MODEL] ✅ Model loaded successfully!")

    def predict(self, image_bytes):
        # 1. Open Image & Force RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((256, 256))
        
        # Shape is (Height=256, Width=256, Channels=3)
        img_np = np.array(image).astype(np.float32)

        # --- SMART CONTRAST ---
        valid_pixels = img_np.sum(axis=2) > 0
        if valid_pixels.any():
            valid_data = img_np[valid_pixels]
            p2, p98 = np.percentile(valid_data, (2, 98))
            if p98 - p2 > 10: 
                img_np = (img_np - p2) / (p98 - p2) * 255.0
                img_np = np.clip(img_np, 0, 255)
        
        # Force Float32
        img_np = img_np.astype(np.float32)

        # --- PADDING ---
        # Padding Height/Width. Shape becomes (288, 288, 3)
        pad_size = 16
        img_np = np.pad(img_np, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='symmetric')

        # --- DEBUG: CHECK SHAPE ---
        # This will show up in your logs. It MUST be (288, 288, 3)
        print(f"[DEBUG] Image Shape before Save: {img_np.shape}") 
        
        # Save Debug Image
        debug_img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        debug_img.save("last_seen_input.png")
        
        # --- PREPARE FOR AI ---
        img_np /= 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # --- TRANSPOSE ---
        # FLIP HERE: (H,W,C) -> (C,H,W)
        img_np = img_np.transpose(2, 0, 1)
        
        input_tensor = np.expand_dims(img_np, axis=0)

        # Inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: input_tensor})
        
        return self._postprocess(result[0], pad_size)

    def _postprocess(self, output_tensor, pad_size):
        # Softmax for probabilities
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return e_x / e_x.sum(axis=0, keepdims=True)

        probs = softmax(output_tensor[0]) 
        
        # Probability of "Not Background"
        foreground_prob = 1.0 - probs[0] 
        
        # Crop padding
        h, w = foreground_prob.shape
        mask_2d = foreground_prob[pad_size:h-pad_size, pad_size:w-pad_size]

        # Save Heatmap
        heatmap_img = Image.fromarray((mask_2d * 255).astype(np.uint8), mode='L')
        heatmap_img.save("last_seen_heatmap.png")

        # Threshold (10%)
        binary_mask = (mask_2d > 0.1).astype(np.uint8) * 255
        
        result_img = Image.fromarray(binary_mask, mode='L')
        bio = io.BytesIO()
        result_img.save(bio, format='PNG')
        return bio.getvalue()