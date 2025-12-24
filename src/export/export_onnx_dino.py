import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
# We use 'dinov2_vits14' (Small) because it's the best CPU candidate.
# Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14
MODEL_NAME = "dinov2_vits14" 
ONNX_PATH = "dinov2_s14.onnx"
QUANTIZED_PATH = "dinov2_s14_int8.onnx"
IMAGE_SIZE = 518  # Standard DINOv2 resolution (must be multiple of 14)

class Dinov2Wrapper(nn.Module):
    """Wrapper to handle DINOv2 output cleanly."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # forward_features returns a dict. We want the patch tokens and CLS token.
        # "x_norm_clstoken" is the global vector (1x384)
        # "x_norm_patchtokens" are the spatial features (1x1369x384)
        out = self.model.forward_features(x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

def export_and_quantize_dino():
    print(f"1. Loading DINOv2 ({MODEL_NAME}) from Torch Hub...")
    # Loading from official Repo
    base_model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
    base_model.to('cpu')
    base_model.eval()
    
    model = Dinov2Wrapper(base_model)
    
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("2. Exporting to ONNX (FP32)...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=['image'],
        output_names=['cls_token', 'patch_tokens'],
        opset_version=18,
        do_constant_folding=True,
        # Dynamic axes allow you to change image size at runtime
        dynamic_axes={
            'image': {2: 'height', 3: 'width'},
            'patch_tokens': {1: 'num_patches'}
        }
    )
    
    size_fp32 = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"   -> FP32 Size: {size_fp32:.2f} MB")

    print("3. Quantizing to INT8...")
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=QUANTIZED_PATH,
        weight_type=QuantType.QUInt8
    )
    
    size_int8 = os.path.getsize(QUANTIZED_PATH) / (1024 * 1024)
    print(f"   -> INT8 Size: {size_int8:.2f} MB")
    print(f"   -> Reduction: {size_fp32 / size_int8:.1f}x smaller")

def benchmark_dino():
    print("\n4. Benchmarking DINOv2 Inference (Avg of 10 runs)...")
    input_data = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    session_fp32 = onnxruntime.InferenceSession(ONNX_PATH, sess_opts, providers=['CPUExecutionProvider'])
    session_int8 = onnxruntime.InferenceSession(QUANTIZED_PATH, sess_opts, providers=['CPUExecutionProvider'])

    def run_inference(session, name):
        # Warmup
        session.run(None, {'image': input_data})
        start = time.time()
        for _ in range(10):
            session.run(None, {'image': input_data})
        end = time.time()
        print(f"   [{name}] Avg Time: {(end - start)/10:.4f} s/img")

    run_inference(session_fp32, "FP32 Standard")
    run_inference(session_int8, "INT8 Quantized")

def test_inference_dino(image_path):
    print(f"\n--- Running Inference on {image_path} ---")
    
    # 1. Preprocess
    # DINOv2 expects normalized ImageNet stats
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_img = cv2.imread(image_path)
    if original_img is None: raise ValueError("Image not found!")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    img_tensor = transform(original_img).unsqueeze(0).numpy() # Shape: (1, 3, 518, 518)
    
    # 2. Run ONNX Inference
    session = onnxruntime.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {'image': img_tensor})
    
    # Output[1] is patch_tokens: Shape (1, 1369, 384)
    patch_tokens = outputs[1]
    
    # 3. Visualization via PCA (384 dims -> 3 dims for RGB)
    # Remove batch dim -> (1369, 384)
    features = patch_tokens[0]
    
    # Fit PCA to get the main 3 components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    
    # Normalize to 0-255 for visualization
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = (pca_features * 255).astype(np.uint8)
    
    # Reshape back to grid. 
    # Logic: 518 / 14 (patch size) = 37. Grid is 37x37.
    grid_size = IMAGE_SIZE // 14
    vis_image = pca_features.reshape(grid_size, grid_size, 3)
    
    # Resize up to original image size for overlay
    vis_image_resized = cv2.resize(vis_image, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 4. Save Side-by-Side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img)
    ax[0].set_title("Original")
    ax[0].axis('off')
    
    ax[1].imshow(vis_image_resized)
    ax[1].set_title("DINOv2 PCA Features")
    ax[1].axis('off')
    
    out_name = "vis_dino_result.jpg"
    plt.savefig(out_name, bbox_inches='tight')
    print(f"Saved visualization to {out_name}")

if __name__ == "__main__":
    export_and_quantize_dino()
    benchmark_dino()
    test_inference_dino("bus.jpg")