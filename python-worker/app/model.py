import onnxruntime as ort
import numpy as np
import io
from PIL import Image
import os
import cv2

class CloudMaskModel:
    """
    A model wrapper for cloud masking using an ONNX runtime session.

    This class handles loading the ONNX model, preprocessing input images,
    running inference, and post-processing the output to generate a cloud mask.
    It employs a hybrid approach combining OpenCV brightness thresholding and
    an AI-based texture validator to distinguish clouds from other bright objects.

    Attributes:
        session (ort.InferenceSession): The ONNX Runtime inference session.
    """
    def __init__(self, model_path):
        """
        Initializes the CloudMaskModel with the specified ONNX model.

        Args:
            model_path (str): Path to the .onnx model file.

        Raises:
            Exception: If the model cannot be loaded.
        """
        print(f"[MODEL] Loading ONNX from {model_path}...")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("[MODEL] ✅ Model loaded successfully!")

    def predict(self, image_bytes):
        """
        Predicts the cloud mask for the given image bytes.

        This method performs the following steps:
        1. Decodes the image bytes.
        2. Applies OpenCV thresholding to identify bright areas (potential clouds).
        3. Preprocesses the image for the AI model.
        4. Runs inference using the ONNX model to validate texture.
        5. Combines the OpenCV mask and AI mask using a bitwise AND operation.
        6. Returns the final mask as a PNG image in bytes.

        Args:
            image_bytes (bytes): The raw image data in bytes.

        Returns:
            bytes: The resulting mask image in PNG format as bytes.
                   Returns an empty byte string if an error occurs.
        """
        try:
            # 1. Load Image & Create Valid Mask
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np_rgb = np.array(pil_image)
            valid_data_mask = (img_np_rgb.sum(axis=2) > 10).astype(np.uint8) * 255

            # --- PART 1: OPENCV (Bright Area Candidates) ---
            gray = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY)
            
            # Brightness Threshold (Slightly relaxed to capture candidates)
            # Down to 180. This will capture clouds AND bright beaches.
            _, cv_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # Cleanup (Morphological Open)
            kernel = np.ones((5,5), np.uint8)
            cv_mask = cv2.morphologyEx(cv_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # -----------------------------------------------------------

            # --- PART 2: AI VALIDATOR (Texture Checker) ---
            # (AI Preprocessing same as before...)
            img_ai = img_np_rgb.astype(np.float32)
            pad_size = 16
            img_ai = np.pad(img_ai, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
            img_ai /= 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_ai = (img_ai - mean) / std
            img_ai = img_ai.transpose(2, 0, 1)
            input_tensor = np.expand_dims(img_ai, axis=0)

            # Run Inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            result = self.session.run([output_name], {input_name: input_tensor})
            
            # Post-process
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
                return e_x / e_x.sum(axis=0, keepdims=True)
            
            probs = softmax(result[0][0])
            ai_prob = 1.0 - probs[0] # Probability of "Not Flat Background"
            
            # Crop padding
            h, w = ai_prob.shape
            ai_prob_cropped = ai_prob[pad_size:h-pad_size, pad_size:w-pad_size]
            
            # AI THRESHOLD (Validator)
            # We set it to 30%. AI must be at least 30% suspicious that this has texture.
            ai_mask = (ai_prob_cropped > 0.30).astype(np.uint8) * 255
            # ----------------------------------------------------

            # --- PART 3: HYBRID MERGE ("AND" LOGIC) ---
            # MAIN CHANGE HERE: Use BITWISE_AND
            # Object must be Bright (OpenCV) AND Have Texture (AI).
            # This will filter out flat bright areas (like bright beaches).
            final_mask = cv2.bitwise_and(cv_mask, ai_mask)
            
            # Final Data Validation
            final_mask = cv2.bitwise_and(final_mask, final_mask, mask=valid_data_mask)

            # Save Debug
            Image.fromarray(cv_mask).save("debug_mask_opencv_bright.png")
            Image.fromarray(ai_mask).save("debug_mask_ai_texture.png")
            
            # Return Result
            result_img = Image.fromarray(final_mask, mode='L')
            bio = io.BytesIO()
            result_img.save(bio, format='PNG')
            return bio.getvalue()

        except Exception as e:
            print(f"[ERROR] Hybrid Processing Failed: {e}")
            return b""

    def _postprocess(self, output_tensor, pad_size, valid_mask):
        """
        Post-processes the model output tensor to generate the final binary mask.

        Args:
            output_tensor (np.ndarray): The raw output tensor from the model.
            pad_size (int): The amount of padding that was added during preprocessing.
            valid_mask (np.ndarray): A mask indicating valid data pixels.

        Returns:
            bytes: The final binary mask image in PNG format as bytes.
        """
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return e_x / e_x.sum(axis=0, keepdims=True)

        probs = softmax(output_tensor[0]) 
        foreground_prob = 1.0 - probs[0] 
        
        # Crop the Padding Back
        h, w = foreground_prob.shape
        mask_2d = foreground_prob[pad_size:h-pad_size, pad_size:w-pad_size]

        # Apply Hard Mask (Keep Black Edges Black)
        mask_2d = mask_2d * valid_mask

        # Save Heatmap
        heatmap_img = Image.fromarray((mask_2d * 255).astype(np.uint8), mode='L')
        heatmap_img.save("debug_3_heatmap.png")

        # Threshold (10%)
        binary_mask = (mask_2d > 0.1).astype(np.uint8) * 255
        
        result_img = Image.fromarray(binary_mask, mode='L')
        bio = io.BytesIO()
        result_img.save(bio, format='PNG')
        return bio.getvalue()