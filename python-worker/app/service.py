from time import time
import sys
import os
import grpc
import time
import numpy as np

#this allowed to import the generated python code
cur_dir = os.path.dirname(os.path.abspath(__file__))
prt_dir = os.path.dirname(cur_dir)
generated_dir = os.path.join(prt_dir, "generated")
sys.path.append(generated_dir)

import geo_service_pb2
import geo_service_pb2_grpc

from app.model import CloudMaskModel

MODEL_PATH = "./models/cloud_mask.onnx"

if os.path.exists(MODEL_PATH):
    print(f"[INIT] Found model at {MODEL_PATH}")
    try:
        ai_model = CloudMaskModel(MODEL_PATH)
        print("[INIT] ✅ Model loaded successfully!")
    except Exception as e:
        print(f"[INIT] ❌ Failed to load model: {e}")
else:
    print(f"[INIT] ⚠️ WARNING: Model not found at {MODEL_PATH}")




class ImageAnalyzerService(geo_service_pb2_grpc.ImageAnalyzerServicer):
    """
    gRPC Service implementation for Image Analysis.
    
    This service receives image tiles via gRPC, processes them using the
    loaded CloudMaskModel, and returns the generated cloud mask.
    """

    def AnalyzeTile(self, request, context):
        """
        Handles the AnalyzeTile gRPC request.

        Receives a TileRequest containing image data, passes it to the AI model,
        and returns a TileResponse with the mask data.

        Args:
            request (geo_service_pb2.TileRequest): The gRPC request object containing
                request_id, model_type, dimensions, and image_data.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            geo_service_pb2.TileResponse: The response object containing
                request_id, success status, result_data (mask), and error_message.
        """

        print(f"\n[PYTHON] 📨 Received Request ID: {request.request_id}")
        print(f"[PYTHON]    Model Type: {request.model_type}")
        print(f"[PYTHON]    Image Size: {request.width}x{request.height} pixels")
        print(f"[PYTHON]    Data Payload: {len(request.image_data)} bytes")

        if ai_model is None:
            return geo_service_pb2.TileResponse(
                request_id=request.request_id,
                success=False,
                error_message="Server Configuration Error: Model not loaded."
            )

        try:
            start_time = time.time()
            
            # --- THE REAL WORK ---
            # 1. Pass raw bytes to the AI Model
            mask_bytes = ai_model.predict(request.image_data)
            # ---------------------
            
            duration = time.time() - start_time
            print(f"[PYTHON] ✅ Inference done in {duration:.4f}s")

            return geo_service_pb2.TileResponse(
                request_id=request.request_id,
                success=True,
                result_data=mask_bytes,
                error_message=""
            )

        except Exception as e:
            print(f"[PYTHON] ❌ Processing Error: {str(e)}")
            return geo_service_pb2.TileResponse(
                request_id=request.request_id,
                success=False,
                result_data=b"",
                error_message=str(e)
            )