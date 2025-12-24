import os
import time
import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
# Adjust these paths to where you saved your SAM 2 files
CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
CONFIG = "sam2_hiera_t.yaml" 
ONNX_PATH = "sam2_tiny_encoder.onnx"
QUANTIZED_PATH = "sam2_tiny_encoder_int8.onnx"

class DynamicSam2Wrapper(nn.Module):
    """
    A smart wrapper that adapts to however many features 
    the model actually returns (1, 3, or 4).
    """
    def __init__(self, sam2_model, num_features):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder
        self.num_features = num_features

    def forward(self, x):
        # The encoder returns a dict: {"vision_features": [...], ...}
        out = self.image_encoder(x)
        feats = out["vision_features"]
        
        # Dynamically return exactly what exists
        if self.num_features == 1:
            return feats[0]
        elif self.num_features == 2:
            return feats[0], feats[1]
        elif self.num_features == 3:
            return feats[0], feats[1], feats[2]
        else:
            return tuple(feats)

def export_and_quantize_sam2():
    print("1. Loading SAM 2...")
    try:
        sam2_model = build_sam2(CONFIG, CHECKPOINT, device='cpu')
    except Exception as e:
        print(f"\nCRITICAL ERROR Loading Model: {e}")
        print(f"Ensure {CONFIG} and {CHECKPOINT} exist.")
        return

    # 2. DIAGNOSIS STEP: Run one pass to see what the model actually outputs
    print("\n2. Running Diagnosis Pass (Eager Mode)...")
    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        out = sam2_model.image_encoder(dummy_input)
        feats = out["vision_features"]
        num_feats = len(feats)
        print(f"   -> Model returned {num_feats} feature map(s).")
        for i, f in enumerate(feats):
            print(f"      Feature {i} shape: {f.shape}")

    # 3. Create Wrapper based on diagnosis
    model = DynamicSam2Wrapper(sam2_model, num_features=num_feats)
    model.eval()

    # 4. Generate Output Names list dynamically
    output_names = [f'feat_{i}' for i in range(num_feats)]
    print(f"   -> Will export with output names: {output_names}")

    print("\n3. Exporting to ONNX (FP32)...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=['image'],
        output_names=output_names,
        opset_version=18, 
        do_constant_folding=True
    )
    
    size_fp32 = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"   -> FP32 Size: {size_fp32:.2f} MB")

    print("\n4. Quantizing to INT8...")
    quantize_dynamic(
        model_input=ONNX_PATH,
        model_output=QUANTIZED_PATH,
        weight_type=QuantType.QUInt8
    )
    
    size_int8 = os.path.getsize(QUANTIZED_PATH) / (1024 * 1024)
    print(f"   -> INT8 Size: {size_int8:.2f} MB")

def test_inference_sam2(image_path):
    print(f"\n--- Running SAM 2 Hybrid Inference on {image_path} ---")

    # 1. Load Original PyTorch Model (Wrapper for Decoder)
    # We still need the original model object to handle the decoding logic
    sam2_model = build_sam2(CONFIG, CHECKPOINT, device='cpu')
    predictor = SAM2ImagePredictor(sam2_model)

    # 2. Preprocess Image
    image = cv2.imread(image_path)
    if image is None: raise ValueError("Image not found")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 1024x1024 (SAM standard)
    input_size = 1024
    img_resized = cv2.resize(image, (input_size, input_size))
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    x = (img_resized - mean) / std
    x = x.transpose(2, 0, 1).astype(np.float32)[None, :, :, :] # (1, 3, 1024, 1024)

    # 3. Run ONNX Encoder (The Heavy Part)
    print("Running Encoder via ONNX...")
    ort_sess = onnxruntime.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    # Run inference
    features = ort_sess.run(None, {"image": x})
    
    # SAM 2 PyTorch expects features as a dict or specific list format
    # The ONNX outputs are [feat_0, feat_1, feat_2] (High res -> Low res)
    # We manually inject these features back into the PyTorch predictor
    # ADAPTIVE LOGIC:
    if len(features) == 1:
        # If model only gave 1 feature, we might need to duplicate it or 
        # the decoder might expect a list of 3 identical ones?
        # Usually, if Tiny outputs 1, the config knows how to handle it.
        # But for the PyTorch decoder wrapper, we must format it as a dict.
        
        # NOTE: You might need to inspect features[0] shape to know which level it is.
        # Assuming it's the high-res one.
        features_dict = {
            "image_embed": features[0], 
            "high_res_feats": [features[0], features[0]] # Dummy fill if needed
        }
    else:
        # Standard 3-feature behavior
        features_dict = {
            "image_embed": features[2],
            "high_res_feats": [features[0], features[1]]
        }
    # 4. Run PyTorch Decoder (The Light Part)
    # We bypass the predictor.set_image() which normally runs the heavy encoder
    # and directly set the features.
    
    # Hack: We manually set the features on the predictor
    predictor._features = features_dict
    predictor._is_image_set = True
    predictor._orig_hw = [image.shape[0], image.shape[1]]
    
    # We need to manually set the input size for coordinate transformations
    # SAM 2 internally tracks transforms. We simply overwrite the internal state 
    # to match our resize operation.
    from sam2.utils.transforms import SAM2Transforms
    predictor._transforms = SAM2Transforms(
        resolution=input_size, 
        mask_threshold=0.0, 
        max_hole_area=0.0, 
        max_sprinkle_area=0.0
    )

    # 5. Prompting (Simulate a click in the middle of the image)
    print("Prompting center point...")
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1]) # 1 = foreground click

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    
    # 6. Save Result
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    plt.plot(input_point[:, 0], input_point[:, 1], 'go') # Green dot for click
    plt.axis('off')
    plt.title(f"SAM 2 Result (Score: {scores[0]:.2f})")
    plt.savefig("vis_sam2_result.jpg", bbox_inches='tight')
    print("Saved visualization to vis_sam2_result.jpg")

if __name__ == "__main__":
    # Hide GPU from PyTorch for this script to prevent "no kernel image" errors
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    export_and_quantize_sam2()
    benchmark_sam2()
    test_inference_sam2("bus.jpg")