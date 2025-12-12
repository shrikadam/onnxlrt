# ONNXLRT 🚀

**ONNXLRT** (ONNX Latency & Runtime Tracker) is a comprehensive benchmarking and deployment framework designed to measure the "Tax of Abstraction" vs. the "Gain of Optimization" in modern AI inference.

This project benchmarks State-of-the-Art models (YOLOv8, SAMURAI, Pi0) across diverse backends (PyTorch, ONNX Runtime, TensorRT) and hardware profiles, ranging from high-density servers (RTX A6000) to power-constrained edge devices (Jetson Orin).

> 📋 **Project Roadmap & Architecture**
>
> For a detailed deep-dive into our hardware strategy, model configurations, and deployment roadmap, please consult the **[Project Master Plan](./plan.md)**.

---

## 🎯 Objectives

1.  **Quantify Performance:** Measure precise P99 Latency and Throughput across:
    * **CPU:** Native vs. ORT vs. OpenVINO.
    * **GPU:** Native CUDA vs. ORT-CUDA vs. TensorRT.
2.  **Verify Efficiency:** Investigate the trade-offs of **INT8 Quantization** (Accuracy loss vs. Speed gain).
3.  **Scale to Production:** Deploy optimized models on **Triton Inference Server** with dynamic batching.
4.  **Profile the Edge:** Measure **FPS/Watt** on Jetson Orin to determine real-world robotics viability.

## 🛠️ Hardware Stack

| Environment | Device | Role |
| :--- | :--- | :--- |
| **Server** | **NVIDIA RTX A6000** (48GB) | High-throughput training, calibration, and Triton serving. |
| **Edge** | **NVIDIA Jetson AGX Orin** | Low-latency inference and power efficiency profiling. |

## 📂 Repository Structure

```text
ONNXLRT/
├── assets/              # Calibration datasets (COCO subset)
├── models/              # Local model artifacts (.onnx, .plan, .pt)
├── src/
│   ├── export/          # Scripts for ONNX export & INT8 Quantization
│   ├── benchmark/       # Pure Python benchmarking (No Triton)
│   └── jetson/          # Power monitoring tools (jtop wrappers)
├── triton_repo/         # Production Model Repository for Triton Server
├── docker/              # Dockerfiles for x86 and aarch64
├── client/              # Triton gRPC clients & perf_analyzer configs
├── plan.md              # Detailed Engineering Plan
└── setup_repo.sh        # Project initialization script
```
## 🏗️ Setup & Installation

Follow these steps to set up the environment on either your Server (x86) or Edge (Jetson/ARM) machine.

### 1. Prerequisites
* **Git**
* **Docker** with **NVIDIA Container Toolkit** installed.
    * *Verify:* Run `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
* **Python 3.8+** (If running locally without Docker)

### 2. Clone the Repository
```bash
git clone [https://github.com/shrikadam/ONNXLRT.git](https://github.com/shrikadam/onnxlrt.git)
cd onnxlrt
```
### 3. Build the environment
#### Option A: For GPU Server (x86_64)
```bash
# Build the container
docker build -t onnxlrt:x86 -f docker/Dockerfile.x86 .

# Run the container (Mounts current directory to /workspace)
docker run -it --rm \
    --gpus all \
    --shm-size=8g \
    -v $(pwd):/workspace \
    onnxlrt:x86
```
#### Option B: For Jetson (ARM64)
```bash
# Build the container (Based on l4t-ml)
docker build -t onnxlrt:jetson -f docker/Dockerfile.jetson .

# Run the container (Requires access to jtop socket for power monitoring)
docker run -it --rm \
    --runtime nvidia \
    -v $(pwd):/workspace \
    -v /run/jtop.sock:/run/jtop.sock \
    onnxlrt:jetson
```
### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```
## 🚀 Usage Guide
### 1. Model Preparation (Export & Quantize)
Before benchmarking, convert the PyTorch model to ONNX and TensorRT engines.
```bash
# Export standard YOLOv8n to ONNX
python src/export/export_onnx.py --model yolov8n --format onnx

# (Optional) Quantize to INT8
python src/export/quantize_ort.py --model models/yolov8/yolov8n.onnx --output models/yolov8/yolov8n_int8.onnx
```
### 2. Run Local Benchmarks
Test pure inference speed (no network overhead).
```bash
# Benchmark ONNX Runtime on CUDA
python src/benchmark/benchmark_local.py --model yolov8n --backend ort --device cuda --precision fp16

# Benchmark PyTorch Native
python src/benchmark/benchmark_local.py --model yolov8n --backend torch --device cuda --precision fp32
```
### 3. Start Triton Server
Start the server pointing to our model repository
```bash
tritonserver --model-repository=/workspace/triton_repo
```
## 📊 Benchmark Results Matrix
Use this table to track results. Lower Latency and Higher FPS are better.
### 1. Server Benchmark Matrix (RTX A6000)
Focuses on raw throughput, latency under load, and memory usage.
| Model | Backend | Precision | Batch Size | Latency P50 (ms) | Latency P99 (ms) | Throughput (FPS) | VRAM Usage (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | PyTorch (Eager) | FP32 | 1 | - | - | - | - |
| **YOLOv8n** | ORT (CUDA) | FP16 | 1 | - | - | - | - |
| **YOLOv8n** | ORT (TensorRT) | FP16 | 1 | - | - | - | - |
| **YOLOv8n** | ORT (TensorRT) | INT8 | 1 | - | - | - | - |
| **YOLOv8n** | ORT (TensorRT) | INT8 | 32 | - | - | - | - |

### 2. Edge Benchmark Matrix (Jetson AGX Orin)
Focuses on power efficiency (FPS/Watt) and hardware offloading (DLA vs GPU).
### **Hardware: NVIDIA Jetson AGX Orin (Edge)**

| Model | Backend | Precision | Accelerator | Latency P99 (ms) | FPS | Power (Avg Watts) | Efficiency (FPS/W) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | PyTorch | FP32 | GPU | - | - | - | - |
| **YOLOv8n** | ORT (CUDA) | FP16 | GPU | - | - | - | - |
| **YOLOv8n** | ORT (TensorRT) | FP16 | GPU | - | - | - | - |
| **YOLOv8n** | ORT (TensorRT) | INT8 | **DLA Core 0** | - | - | - | - |
| **YOLOv8n** | ORT (TensorRT) | INT8 | **DLA Core 1** | - | - | - | - |
| **Pi0** | PyTorch | BF16 | GPU | - | - | - | - |