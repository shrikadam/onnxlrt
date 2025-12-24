import os
import shutil
import time
import glob
import cv2
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = 'yolo11n-seg'  
SOURCE = 0                     # 0 for webcam, or path to video file
IMG_SIZE = 640
DATA_YAML = 'coco8-seg.yaml'  
EXPORT_DIR = 'exported_models'

# Ensure export directory exists (Do NOT delete it if we want to skip re-exports)
os.makedirs(EXPORT_DIR, exist_ok=True)

def rename_and_save_artifact(original_path, new_name):
    """Helper to rename exported models so they don't overwrite each other."""
    new_path = os.path.join(EXPORT_DIR, new_name)
   
    # Handle directories (like OpenVINO) and files (like ONNX/Engine)
    if os.path.isdir(original_path):
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        shutil.move(original_path, new_path)
    elif os.path.isfile(original_path):
        if os.path.exists(new_path):
            os.remove(new_path)
        shutil.move(original_path, new_path)
   
    return new_path

def get_openvino_bin(folder_path):
    """Finds the .bin file inside the OpenVINO folder."""
    if not os.path.exists(folder_path): return None
    bin_files = glob.glob(os.path.join(folder_path, '**', '*.bin'), recursive=True)
    if bin_files:
        return bin_files[0]
    return None

def export_all_formats():
    print(f"\n{'='*20} PHASE 1: CHECKING/EXPORTING MODELS {'='*20}")
    model = YOLO(f'{MODEL_NAME + ".pt"}')
    paths = {}
   
    # ==========================
    # 1. ONNX
    # ==========================
    target_onnx = os.path.join(EXPORT_DIR, f'{MODEL_NAME + ".onnx"}')
    if os.path.exists(target_onnx):
        print(f"  [Found existing] ONNX: {target_onnx}")
        paths['ONNX'] = target_onnx
    else:
        print("\n--- Exporting ONNX (Static) ---")
        model.export(format='onnx', dynamic=False, simplify=True)
        paths['ONNX'] = rename_and_save_artifact(f'{MODEL_NAME + ".onnx"}', f'{MODEL_NAME + ".onnx"}')

    # ==========================
    # 2. OpenVINO FP32
    # ==========================
    target_ov_fp32 = os.path.join(EXPORT_DIR, f'{MODEL_NAME + "_openvino_model_fp32"}')
    if get_openvino_bin(target_ov_fp32):
        print(f"  [Found existing] OpenVINO FP32: {target_ov_fp32}")
        # paths['OpenVINO FP32'] = get_openvino_bin(target_ov_fp32)
        paths['OpenVINO FP32'] = target_ov_fp32
    else:
        print("\n--- Exporting OpenVINO (FP32) ---")
        model.export(format='openvino', half=False)
        base_folder = rename_and_save_artifact(f'{MODEL_NAME + "_openvino_model"}', f'{MODEL_NAME + "_openvino_model_fp32"}')
        # paths['OpenVINO FP32'] = get_openvino_bin(base_folder)
        paths['OpenVINO FP32'] = base_folder

    # ==========================
    # 3. OpenVINO FP16
    # ==========================
    target_ov_fp16 = os.path.join(EXPORT_DIR, f'{MODEL_NAME + "_openvino_model_fp16"}')
    if get_openvino_bin(target_ov_fp16):
        print(f"  [Found existing] OpenVINO FP16: {target_ov_fp16}")
        # paths['OpenVINO FP16'] = get_openvino_bin(target_ov_fp16)
        paths['OpenVINO FP16'] = target_ov_fp16
    else:
        print("\n--- Exporting OpenVINO (FP16) ---")
        model.export(format='openvino', half=True)
        base_folder = rename_and_save_artifact(f'{MODEL_NAME + "_openvino_model"}', f'{MODEL_NAME + "_openvino_model_fp16"}')
        # paths['OpenVINO FP16'] = get_openvino_bin(base_folder)
        paths['OpenVINO FP16'] = base_folder

    # ==========================
    # 4. OpenVINO INT8
    # ==========================
    target_ov_int8 = os.path.join(EXPORT_DIR, f'{MODEL_NAME + "_openvino_model_int8"}')
    if get_openvino_bin(target_ov_int8):
        print(f"  [Found existing] OpenVINO INT8: {target_ov_int8}")
        # paths['OpenVINO INT8'] = get_openvino_bin(target_ov_int8)
        paths['OpenVINO INT8'] = target_ov_int8
    else:
        print("\n--- Exporting OpenVINO (INT8) ---")
        model.export(format='openvino', int8=True, data=DATA_YAML)
       
        # Handle inconsistent naming from Ultralytics
        possible_names = [f'{MODEL_NAME + "_int8_openvino_model"}', f'{MODEL_NAME + "_openvino_model"}']
        found_export = None
        for p in possible_names:
            if os.path.exists(p):
                found_export = p
                break
               
        if found_export:
            base_folder = rename_and_save_artifact(found_export, f'{MODEL_NAME + "_openvino_model_int8"}')
            # paths['OpenVINO INT8'] = get_openvino_bin(base_folder)
            paths['OpenVINO INT8'] = base_folder

    # ==========================
    # GPU EXPORTS (TensorRT)
    # ==========================
    if torch.cuda.is_available():
        # 5. TensorRT FP32
        target_trt_fp32 = os.path.join(EXPORT_DIR, f'{MODEL_NAME + "_fp32.engine"}')
        if os.path.exists(target_trt_fp32):
            print(f"  [Found existing] TensorRT FP32: {target_trt_fp32}")
            paths['TensorRT FP32'] = target_trt_fp32
        else:
            try:
                print("\n--- Exporting TensorRT (FP32) ---")
                model.export(format='engine', device=0, half=False, simplify=True)
                paths['TensorRT FP32'] = rename_and_save_artifact(f'{MODEL_NAME + ".engine"}', f'{MODEL_NAME + "_fp32.engine"}')
            except Exception as e: print(f"SKIP TensorRT FP32: {e}")

        # 6. TensorRT FP16
        target_trt_fp16 = os.path.join(EXPORT_DIR, f'{MODEL_NAME + "_fp16.engine"}')
        if os.path.exists(target_trt_fp16):
            print(f"  [Found existing] TensorRT FP16: {target_trt_fp16}")
            paths['TensorRT FP16'] = target_trt_fp16
        else:
            try:
                print("\n--- Exporting TensorRT (FP16) ---")
                model.export(format='engine', device=0, half=True, simplify=True)
                paths['TensorRT FP16'] = rename_and_save_artifact(f'{MODEL_NAME + ".engine"}', f'{MODEL_NAME + "_fp16.engine"}')
            except Exception as e: print(f"SKIP TensorRT FP16: {e}")

        # 7. TensorRT INT8
        # target_trt_int8 = os.path.join(EXPORT_DIR, 'yolo11_seg_int8.engine')
        # if os.path.exists(target_trt_int8):
        #     print(f"  [Found existing] TensorRT INT8: {target_trt_int8}")
        #     paths['TensorRT INT8'] = target_trt_int8
        # else:
        #     try:
        #         print("\n--- Exporting TensorRT (INT8) ---")
        #         if os.path.exists("calibration.cache"): os.remove("calibration.cache")
        #         model.export(format='engine', device=0, int8=True, data=DATA_YAML)
        #         paths['TensorRT INT8'] = rename_and_save_artifact(f'{MODEL_NAME.replace(".pt", ".engine")}', 'yolo11_seg_int8.engine')
        #     except Exception as e: print(f"SKIP TensorRT INT8: {e}")

    return paths

def run_inference_benchmark(model_path, format_name, device):
    print(f"\n>>> TESTING: {format_name}")
    print(f"    Path: {model_path}")
    print(f"    Device: {device}")
   
    try:
        if format_name == "Eager (PyTorch CPU)":
             model = YOLO(MODEL_NAME)
        elif format_name == "Native CUDA":
             model = YOLO(MODEL_NAME)
        else:
             model = YOLO(model_path, task='segment')

        # Warmup
        print("    Warming up...")
        dummy_input = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        if str(device) == '4': dummy_input = dummy_input.to('cuda')
       
        try:
            model(dummy_input, verbose=False, device=device)
        except:
            pass

        print("    Starting Stream... (Press 'q' to skip to next model)")
       
        cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return 0

        frame_count = 0
        start_time = time.time()
        fps_log = []

        while True:
            ret, frame = cap.read()
            if not ret: break

            t0 = time.time()
            results = model.predict(frame, device=device, verbose=False, imgsz=IMG_SIZE, half=False)
            t1 = time.time()

            frame_count += 1
            inference_time = t1 - t0
            if inference_time > 0:
                fps = 1.0 / inference_time
                fps_log.append(fps)

                annotated_frame = results[0].plot()
                avg_fps = sum(fps_log[-50:]) / len(fps_log[-50:])
                cv2.putText(annotated_frame, f"Mode: {format_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Comparative Inference', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if time.time() - start_time > 15: break

        cap.release()
        cv2.destroyAllWindows()
       
        avg_fps_final = sum(fps_log) / len(fps_log) if fps_log else 0
        print(f"    RESULT: Average FPS for {format_name}: {avg_fps_final:.2f}")
        return avg_fps_final

    except Exception as e:
        print(f"    [!] Failed to run {format_name}: {e}")
        return 0

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
   
    # 1. Export Phase
    exported_models = export_all_formats()
   
    # 2. Define Progression (Name, Path, Device)
    progression = []

    # CPU Tests
    progression.append(("Eager (PyTorch CPU)", MODEL_NAME, "cpu"))
    if 'ONNX' in exported_models:
        progression.append(("ONNX (CPU)", exported_models['ONNX'], "cpu"))
    if 'OpenVINO FP32' in exported_models:
        progression.append(("OpenVINO FP32", exported_models['OpenVINO FP32'], "cpu"))
    if 'OpenVINO FP16' in exported_models:
        progression.append(("OpenVINO FP16", exported_models['OpenVINO FP16'], "cpu"))
    if 'OpenVINO INT8' in exported_models:
        progression.append(("OpenVINO INT8", exported_models['OpenVINO INT8'], "cpu"))

    # GPU Tests
    if torch.cuda.is_available():
        progression.append(("Native CUDA", MODEL_NAME, 0))
       
        if 'TensorRT FP32' in exported_models:
            progression.append(("TensorRT FP32", exported_models['TensorRT FP32'], 0))
        if 'TensorRT FP16' in exported_models:
            progression.append(("TensorRT FP16", exported_models['TensorRT FP16'], 0))
        if 'TensorRT INT8' in exported_models:
            progression.append(("TensorRT INT8", exported_models['TensorRT INT8'], 0))

    # 3. Inference Phase
    print(f"\n{'='*20} PHASE 2: PROGRESSIVE INFERENCE {'='*20}")
   
    results = {}
    for name, path, dev in progression:
        fps = run_inference_benchmark(path, name, dev)
        results[name] = fps
        time.sleep(1)

    # 4. Final Report
    print(f"\n{'='*20} FINAL BENCHMARK REPORT {'='*20}")
    print(f"{'Format':<25} | {'FPS':<10}")
    print("-" * 40)
    for name, fps in results.items():
        print(f"{name:<25} | {fps:.2f}")
    print("-" * 40)

