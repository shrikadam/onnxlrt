## 1. Project Overview
**Goal:** Benchmark and optimize State-of-the-Art (SOTA) models across different hardware architectures (Server vs. Edge) and inference backends.
**Core Objective:** Measure the "Tax" of abstraction vs. the "Gain" of optimization (INT8, TensorRT) on Accuracy, Latency, and Power Efficiency.

## 2. Hardware Specifications

### **Server (The Beast)**
* **GPU:** NVIDIA RTX A6000 (48GB GDDR6 VRAM, Ampere Architecture).
* **Capabilities:** Massive VRAM allows concurrent serving of LLMs (Pi0) and Vision models. Supports 3rd Gen Tensor Cores for accelerated INT8/BF16.
* **Role:** High-throughput inference, Triton Server host, calibration/export workhorse.

### **Edge (The Target)**
* **Device:** NVIDIA Jetson AGX Orin (64GB Unified Memory).
* **Capabilities:** Dedicated DLA (Deep Learning Accelerator) for INT8 offloading.
* **Role:** Power-constrained deployment, efficiency profiling (FPS/Watt).

---

## 3. Supported Models & Backends

| Model | Task | Backends to Test | Quantization | Special Notes |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8** | Object Detection | PyTorch, ORT-CPU, ORT-OpenVINO, ORT-CUDA, TRT-Native | FP32, FP16, INT8 | Baseline model. Easy to quantize. |
| **SAMURAI** | Visual Tracking | PyTorch, ORT-CUDA | FP16 | **Stateful.** Requires Triton Sequence Batcher to maintain object ID across frames. |
| **Pi0** | VLA (Robotics) | PyTorch (Python Backend) | BF16 / INT4 | **Multimodal.** Requires complex input handling (Text + Image + State) and iterative output (Action Chunks). |

---

## 4. Phase 1: The YOLOv8 Baseline (Immediate)

### **A. Export & Optimization**
1.  **Native Export:** Convert `.pt` to `.onnx` (Opset 17).
2.  **TRT Engine Building:**
    * Use `trtexec` on A6000 for server-side engines.
    * **Crucial:** Must rebuild `.plan` files on Jetson (TensorRT engines are not portable across GPU architectures).
3.  **INT8 Calibration:**
    * Use `MinMax` calibration for weights.
    * Use `Entropy` calibration for activations (better for detecting outliers in bounding boxes).

### **B. Triton Deployment Strategy**
* **Dynamic Batching:** Enable `dynamic_batching { preferred_batch_size: [ 4, 8 ] }` to saturate the A6000 CUDA cores.
* **Ensemble Model:**
    * *Step 1:* Image Pre-processing (Resize/Norm) -> **DALI Backend** (GPU accelerated).
    * *Step 2:* Inference -> **TensorRT Backend**.
    * *Step 3:* Post-processing (NMS) -> **Python Backend** (easier to implement) or **Custom C++ Backend** (max speed).

---

## 5. Phase 2: Advanced Models (Future)

### **A. SAMURAI (Visual Object Tracking)**
* **Challenge:** The model needs "Memory" of the previous frame.
* **Solution:** Use **Triton Sequence Batcher**.
    * Client sends a `CORRID` (Correlation ID) with every frame.
    * Triton routes all frames with the same `CORRID` to the same model instance to preserve the hidden state.

### **B. Pi0 (Vision-Language-Action)**
* **Challenge:** VLA models output "Action Chunks" (e.g., 50 steps of robot arm movement) and are extremely heavy.
* **Strategy:**
    * Use **Python Backend** in Triton to wrap the `transformers` / `vllm` logic.
    * **Memory Management:** The A6000 (48GB) is perfect for this. We can load Pi0 (approx 7-10GB) alongside YOLOv8 without swapping.

---

## 6. Phase 3: Benchmarking & Profiling

### **Metrics Definition**
1.  **Latency P99:** The time it takes for 99% of requests to finish. Critical for robotics (stutter = crash).
2.  **Throughput:** Inferences per second (IPS).
3.  **mAP @ .50-.95:** Accuracy loss due to quantization.
4.  **Energy Efficiency:** `Inferences / Joule`.

### **Tools**
* **Desktop:** `trtexec`, `perf_analyzer` (Triton tool).
* **Jetson:** `tegrastats`, `jtop`.

---

## 7. Execution Roadmap

- [ ] **Setup:** Build Docker containers for x86 (Server) and aarch64 (Jetson).
- [ ] **YOLO-Local:** Run local `benchmark_local.py` script on A6000 (PyTorch vs ORT).
- [ ] **YOLO-Triton:** Deploy YOLOv8 on Triton (Localhost) and test with `triton_client.py`.
- [ ] **Edge Port:** Copy repo to Jetson Orin. Re-export TensorRT engines.
- [ ] **Power Study:** Run `monitor.sh` on Jetson while flooding it with requests.
- [ ] **Visuals:** Generate Latency vs. Power graphs.
- [ ] **Advanced:** Attempt SAMURAI export to ONNX (High difficulty).

---

## 8. Directory Reference
* `src/export/`: Scripts to create .onnx and .plan files.
* `triton_repo/`: The strict file structure Triton requires.
* `docker/`: Environment definitions to keep versions sane.