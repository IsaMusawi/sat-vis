import sys
import os
import grpc
from concurrent import futures
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))
prt_dir = os.path.dirname(cur_dir)
generated_dir = os.path.join(prt_dir, "generated")
sys.path.append(generated_dir)

import geo_service_pb2_grpc
from app.service import ImageAnalyzerService


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    geo_service_pb2_grpc.add_ImageAnalyzerServicer_to_server(ImageAnalyzerService(), server)

    port = '50051'
    server.add_insecure_port(f'[::]:{port}')

    print(f"==========================================")
    print(f"🚀 PYTHON WORKER RUNNING ON PORT {port}")
    print(f"   Waiting for Go to send data...")
    print(f"==========================================")

    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[PYTHON] Shutting down...")

if __name__ == '__main__':
    serve()
