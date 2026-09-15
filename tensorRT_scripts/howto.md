# Walkthrough: TensorRT Integration (`adgbc_trt`) in Pupil 2D Detector

## Overview of Changes
We have integrated NVIDIA TensorRT engine support into the Pupil Core 2D neural network detector plugin on branch `features/trt`.

### 1. Zero-Copy TensorRT Wrapper
* Created [`trt_detector_wrapper.py`](file:///D:/School/4-3/pupil/pupil_src/shared_modules/pupil_detector_plugins/trt_detector_wrapper.py) implementing `TRTDetectorModule`.
* **Zero Host Transfers**: Directly binds PyTorch CUDA tensor memory addresses to the TensorRT context (`execute_async_v3` for modern TensorRT 10.x/11.x, with backward-compatible fallback to `execute_async_v2`).
* **Metadata Detection**: Automatically parses input/output tensor names and extracts expected spatial resolution (`expected_h=192`, `expected_w=192`) and class count from the engine.

### 2. 2D Neural Network Detector Plugin
* Updated [`detector_2d_nn_plugin.py`](file:///D:/School/4-3/pupil/pupil_src/shared_modules/pupil_detector_plugins/detector_2d_nn_plugin.py):
  - Added `("adgbc_trt", "AD-GBC (TensorRT)")` to `AVAILABLE_MODELS`.
  - In `_load_model()`, added engine resolution logic that searches `model_ckpts/adgbc_nn_best.engine` (with fallbacks to `adgbc.engine` and `nnunet_gbc_backbone.engine`).
  - In `_detect_nn()`, added adaptive spatial rescaling:
    - Downsamples the camera frame ($640 \times 400 \to 192 \times 192$) for engine input.
    - Scales contour coordinates back as `float32` prior to `cv2.convexHull()` and `cv2.fitEllipse()`, ensuring sub-pixel precision and exact alignment with downstream Pupil gaze normalization.
  - Preserved `adgbc_400` as requested.

---

## Server Deployment & Verification Instructions
*(Execute these steps on your remote GPU server: RTX A6000 for PoC, RTX 4090 for production)*

### Step 1: Convert Model to TensorRT Engine
In your remote server environment:
```bash
conda activate nnunet_mlu
cd /home/iulab9/PycharmProjects/nnUNet

# 1. Export PyTorch checkpoint to ONNX
python pth_to_onnx.py \
    --model_folder /mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_16__nnUNetPlans__2d/ \
    --fold 0 \
    --checkpoint_name checkpoint_final.pth \
    --output_file nnunet_gbc_backbone.onnx \
    --opset 17 \
    --device cuda:0
    
# 1.1 Command for iulab9 CUDA=1
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python tensorRT_scripts/pth_to_onnx.py --model_folder /mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_16__nnUNetPlans__2d/ --fold 0 --checkpoint_name checkpoint_final.pth --output_file nn_GBC_S_16.onnx --opset 17 --device cuda:0

# 2. Build TensorRT Engine (FP16, 192x192)
python onnx_to_engine.py \
    --onnx_file nnunet_gbc_backbone.onnx \
    --output_engine adgbc_nn_best.engine \
    --precision fp16 \
    --min_batch 1 --opt_batch 1 --max_batch 4 \
    --device 0
```

### Step 2: Place Engine in Pupil Plugin Checkpoints
Copy the built `adgbc_nn_best.engine` into Pupil's checkpoint directory:
```bash
cp adgbc_nn_best.engine /path/to/pupil/pupil_src/shared_modules/pupil_detector_plugins/model_ckpts/adgbc_nn_best.engine
```

### Step 3: Run Engine Self-Test
Verify that the engine and wrapper load cleanly in the Pupil environment:
```bash
python -c "
import torch
from pupil_detector_plugins.trt_detector_wrapper import TRTDetectorModule
engine_file = 'pupil_src/shared_modules/pupil_detector_plugins/model_ckpts/adgbc_nn_best.engine'
module = TRTDetectorModule(engine_file, torch.device('cuda:0'))
dummy = torch.randn(1, 1, module.expected_h, module.expected_w, device='cuda:0')
out = module(dummy)
print('Inference test succeeded! Output shape:', out.shape)
assert out.shape == (1, 4, module.expected_h, module.expected_w)
"
```

### Step 4: Launch Pupil Capture
```bash
python main.py capture
```
1. Open the plugin manager and enable **Pupil Detector 2D (Neural Net)**.
2. Under the **Model** dropdown, select **AD-GBC (TensorRT)**.
3. Observe the pupil outline overlay, confidence bar graph, and timing waterfall metrics.
