# 🛰️ sat-vis: Satellite Cloud Detection Pipeline

![Go](https://img.shields.io/badge/Backend-Go-00ADD8?style=flat&logo=go)
![Python](https://img.shields.io/badge/AI-Python-3776AB?style=flat&logo=python)
![MinIO](https://img.shields.io/badge/Storage-MinIO-C72C48?style=flat&logo=minio)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?style=flat&logo=docker)
![Status](https://img.shields.io/badge/Status-MVP-orange)

**sat-vis** is a **resource-efficient**, hybrid processing pipeline designed to handle cloud detection in high-resolution satellite imagery.

It demonstrates a transition from traditional Geodesy data processing to modern Software Engineering, bridging **Go** (for optimized I/O) and **Python** (for AI inference), while using **MinIO** as a centralized object storage to simulate a cloud-native environment.

---

## 🎯 The Problem

Processing large-scale satellite imagery presents unique engineering challenges that simple scripts cannot handle effectively:

1.  **False Positives in Detection:** Traditional brightness thresholding often confuses **white sand beaches** or bright urban structures with clouds.
2.  **Data Alignment (The "Stride" Issue):** Slicing raw satellite rasters (Tiff) often results in pixel misalignment ("Zig-Zag" distortion) due to memory padding in binary buffers.
3.  **Scalability:** Storing processed tiles on a local file system prevents the system from scaling horizontally in a distributed environment.

**sat-vis** solves these by combining low-level byte manipulation in Go with a "Hybrid Intelligence" logic in Python.

---

## 🏗️ Architecture & Workflow

The system adopts a microservice-like architecture where storage is decoupled from processing logic.

```mermaid
graph LR
    A[Raw Satellite Tiff] -->|Input| B(Go Tiler Service)
    B -->|Slice & Upload| C[(MinIO Object Storage)]
    C -.->|Async Fetch| D(Python AI Worker)
    D -->|Step 1: OpenCV| E{Is Bright?}
    E -- No --> F[Ignore]
    E -- Yes --> G[Step 2: ONNX ResNet]
    G --> H{Is Cloud Texture?}
    H -- Yes --> I[Generate Cloud Mask]
    H -- No --> F

Note: The model.onnx file is not included in this repository due to file size limits (>100MB). To run this locally, you would need to place a standard ResNet-UNet ONNX model in the /models directory.