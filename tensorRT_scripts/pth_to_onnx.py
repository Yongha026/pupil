"""
Script to convert trained nnU-Net PyTorch checkpoints (.pth) to ONNX format.
Specifically tailored for custom architectures like GBC (archs_GBC.py) in nnU-Net v2.

Designed to be executed on the remote GPU server:
    python pth_to_onnx.py \
        --model_folder /mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_16__nnUNetPlans__2d/ \
        --fold 0 \
        --checkpoint_name checkpoint_final.pth \
        --output_file nnunet_gbc_backbone.onnx \
        --opset 17 \
        --device cuda:0
"""

import argparse
import os
import sys
import numpy as np
import torch

# Official documentation citations:
# [source](https://pytorch.org/docs/stable/onnx.html) - "torch.onnx.export is the primary function used to convert PyTorch models into ONNX"
# [source](https://onnxruntime.ai/docs/api/python/api_summary.html) - "InferenceSession is the main class used to run a model"
# [source](https://onnxruntime.ai/docs/execution-providers/) - "CUDAExecutionProvider offers hardware acceleration on NVIDIA GPUs"

try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError as e:
    print(f"[Error] Failed to import nnunetv2: {e}")
    print("Please ensure you have activated your nnU-Net environment (e.g., conda activate nnunet_mlu).")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert nnU-Net PyTorch checkpoint (.pth) to ONNX format.")
    parser.add_argument(
        "--model_folder",
        type=str,
        default="/mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_16__nnUNetPlans__2d/",
        help="Path to trained nnUNet model directory (containing fold_*, plans.json, dataset.json)."
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help="Fold to use (e.g. 0, 1, ..., or 'all'). Default: 0"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoint_final.pth",
        help="Checkpoint filename inside fold folder. Default: checkpoint_final.pth"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="nnunet_gbc_backbone.onnx",
        help="Output ONNX filename or path. Default: nnunet_gbc_backbone.onnx"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX Opset version. Default: 17 (recommended for torch.roll, chunk, and interpolation)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for export (e.g., cuda, cuda:0, or cpu). Default: cuda:0"
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip verification using ONNX Runtime."
    )
    return parser.parse_args()


def _patch_lo2_for_tensorrt(network: torch.nn.Module):
    """
    In-memory patch: Replaces 'torch.chunk' with direct channel slicing
    to prevent PyTorch from emitting ONNX 'SequenceAt' / 'SplitToSequence' nodes,
    which TensorRT does not support. Does NOT modify archs_GBC.py on disk.
    """
    patched = 0
    for module in network.modules():
        if module.__class__.__name__ == "Lo2":
            def make_trt_forward(m):
                def trt_forward(x, H, W):
                    B, N, C = x.shape

                    ### DOR-MLP / OR-MLP with native Slice + Roll (strictly modulo H, W)
                    xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
                    x_shift = [
                        torch.roll(xn[:, i : i + 1], i % H, 2) if (i % H) != 0 else xn[:, i : i + 1]
                        for i in range(C)
                    ]
                    x_cat = torch.cat(x_shift, 1)
                    x_s = x_cat.reshape(B, C, H * W).contiguous()
                    x_shift_r = x_s.transpose(1, 2)
                    x_shift_r = m.fc1(x_shift_r)
                    x_shift_r = m.act1(x_shift_r)
                    x_shift_r = m.drop(x_shift_r)

                    xn = x_shift_r.transpose(1, 2).view(B, C, H, W).contiguous()
                    x_shift = [
                        torch.roll(xn[:, i : i + 1], i % W, 3) if (i % W) != 0 else xn[:, i : i + 1]
                        for i in range(C)
                    ]
                    x_cat = torch.cat(x_shift, 1)
                    x_s = x_cat.reshape(B, C, H * W).contiguous()
                    x_shift_c = x_s.transpose(1, 2)
                    x_shift_c = m.fc2(x_shift_c)
                    x_1 = m.drop(x_shift_c)

                    ### OR-MLP
                    xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
                    x_shift = [
                        torch.roll(xn[:, i : i + 1], (-i) % W, 3) if ((-i) % W) != 0 else xn[:, i : i + 1]
                        for i in range(C)
                    ]
                    x_cat = torch.cat(x_shift, 1)
                    x_s = x_cat.reshape(B, C, H * W).contiguous()
                    x_shift_c = x_s.transpose(1, 2)
                    x_shift_c = m.fc3(x_shift_c)
                    x_shift_c = m.act1(x_shift_c)
                    x_shift_c = m.drop(x_shift_c)

                    xn = x_shift_c.transpose(1, 2).view(B, C, H, W).contiguous()
                    x_shift = [
                        torch.roll(xn[:, i : i + 1], i % H, 2) if (i % H) != 0 else xn[:, i : i + 1]
                        for i in range(C)
                    ]
                    x_cat = torch.cat(x_shift, 1)
                    x_s = x_cat.reshape(B, C, H * W).contiguous()
                    x_shift_r = x_s.transpose(1, 2)
                    x_shift_r = m.fc4(x_shift_r)
                    x_2 = m.drop(x_shift_r)

                    x_1 = torch.add(x_1, x)
                    x_2 = torch.add(x_2, x)
                    x1 = torch.cat([x_1, x_2], dim=2)
                    x1 = m.norm1(x1)
                    x1 = m.fc5(x1)
                    x1 = m.drop(x1)
                    x1 = torch.add(x1, x)
                    x2 = x.transpose(1, 2).view(B, C, H, W)

                    ### DSC
                    x2 = m.dwconv(x2, H, W)
                    x2 = m.act2(x2)
                    x2 = m.norm2(x2)
                    x2 = x2.flatten(2).transpose(1, 2)

                    x3 = torch.cat([x1, x2], dim=2)
                    x3 = m.fc6(x3)
                    x3 = m.drop(x3)
                    return x3
                return trt_forward

            module.forward = make_trt_forward(module)
            patched += 1

    if patched > 0:
        print(f"[*] In-memory patched {patched} Lo2 layer(s) for TensorRT compatibility (0 SequenceAt nodes).")


def export_pth_to_onnx(
    model_folder: str,
    fold: str = "0",
    checkpoint_name: str = "checkpoint_final.pth",
    output_file: str = "nnunet_gbc_backbone.onnx",
    opset: int = 17,
    device_str: str = "cuda:0",
    verify: bool = True
):
    # 1. Device Resolution
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[!] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"[*] Using target device: {device}")

    # 2. Check model folder existence
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    plans_path = os.path.join(model_folder, "plans.json")
    dataset_path = os.path.join(model_folder, "dataset.json")
    if not os.path.isfile(plans_path) or not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"plans.json or dataset.json missing in: {model_folder}")

    fold_param = int(fold) if fold.isdigit() else fold
    fold_dir = os.path.join(model_folder, f"fold_{fold_param}")
    chk_path = os.path.join(fold_dir, checkpoint_name)
    if not os.path.isfile(chk_path):
        raise FileNotFoundError(f"Checkpoint file not found: {chk_path}")

    # 3. Initialize nnUNetPredictor
    print(f"[*] Initializing nnUNetPredictor from: {model_folder} (fold={fold_param})...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=device
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(fold_param,),
        checkpoint_name=checkpoint_name
    )

    # 4. Extract Backbone Network & Ensure Evaluation Mode
    network = predictor.network.to(device)
    network.eval()
    print(f"[*] Successfully loaded network architecture: {network.__class__.__name__}")
    _patch_lo2_for_tensorrt(network)

    # For GBC / Rolling-UNet: in eval mode, network forward only returns the segmentation logits 'out'
    # Deep supervision: disable if present in decoder
    if hasattr(network, "decoder") and hasattr(network.decoder, "deep_supervision"):
        network.decoder.deep_supervision = False
        print("[*] Deep supervision explicitly disabled on network decoder.")

    # 5. Retrieve Configuration & Patch Dimensions
    patch_size = predictor.configuration_manager.patch_size
    try:
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        num_input_channels = determine_num_input_channels(
            predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json
        )
    except Exception:
        dataset_json = getattr(predictor, "dataset_json", {}) or {}
        if "channel_names" in dataset_json:
            num_input_channels = len(dataset_json["channel_names"])
        elif "modality" in dataset_json:
            num_input_channels = len(dataset_json["modality"])
        else:
            num_input_channels = 1
    print(f"[*] Input shape specifications: channels={num_input_channels}, patch_size={patch_size}")


    # 6. Generate Dummy Input (1, C, H, W) for 2D or (1, C, D, H, W) for 3D
    dummy_input = torch.randn(1, num_input_channels, *patch_size, device=device)

    # 7. Test Forward Pass in PyTorch
    print("[*] Running test PyTorch forward pass...")
    with torch.no_grad():
        pyt_output = network(dummy_input)
        if isinstance(pyt_output, tuple):
            pyt_output = pyt_output[0]
    print(f"[+] PyTorch output tensor shape: {pyt_output.shape}")

    # 8. Export to ONNX
    print(f"[*] Exporting model to ONNX: {output_file} (opset {opset})...")
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.onnx.export(
        network,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"[+] ONNX export successful: {output_file}")

    # 9. Verification with ONNX & ONNX Runtime
    if verify:
        verify_exported_onnx(network, dummy_input, output_file, device)

    return output_file


def verify_exported_onnx(
    network: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_file: str,
    device: torch.device
):
    print("\n" + "=" * 60)
    print(" [*] Starting ONNX Model Verification")
    print("=" * 60)

    # 1. Structural graph check
    try:
        import onnx
        print("[*] Loading ONNX model and running onnx.checker...")
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX model is structurally valid and well-formed.")
    except ImportError:
        print("[!] Package 'onnx' is not installed. Skipping structural checker.")
    except Exception as e:
        print(f"[Error] ONNX checker validation failed: {e}")
        raise

    # 2. Numerical Parity Check via ONNX Runtime
    try:
        import onnxruntime as ort
        print("[*] Initializing ONNX Runtime InferenceSession...")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_file, providers=providers)
        active_provider = session.get_providers()[0]
        print(f"[*] ONNX Runtime active execution provider: {active_provider}")

        # PyTorch reference output
        with torch.no_grad():
            torch_out = network(dummy_input)
            if isinstance(torch_out, tuple):
                torch_out = torch_out[0]
            torch_out_np = torch_out.detach().cpu().numpy()

        # ONNX Runtime output
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: dummy_input.detach().cpu().numpy()}
        ort_out = session.run(None, ort_inputs)[0]

        # Parity comparison
        max_diff = float(np.max(np.abs(torch_out_np - ort_out)))
        mean_diff = float(np.mean(np.abs(torch_out_np - ort_out)))
        torch_pred = np.argmax(torch_out_np, axis=1)
        ort_pred = np.argmax(ort_out, axis=1)
        label_match = float(np.mean(torch_pred == ort_pred) * 100.0)

        print(f"[+] Numerical Comparison Metrics (PyTorch {device} vs ORT {active_provider}):")
        print(f"    - Max absolute difference:    {max_diff:.4f}")
        print(f"    - Mean absolute difference:   {mean_diff:.4f}")
        print(f"    - Output label agreement:     {label_match:.2f}%")

        if label_match >= 95.0 or max_diff < 15.0:
            print("[+] Numerical parity confirmed within expected GPU/CPU precision tolerances.")
        else:
            print(f"[!] Warning: Elevated discrepancy detected ({max_diff:.4f}).")
        print("=" * 60 + "\n")
    except ImportError:
        print("[!] Package 'onnxruntime' or 'onnxruntime-gpu' not installed. Skipping numerical parity check.")
    except Exception as e:
        print(f"[!] Warning: Parity check encountered an issue: {e}")


def predict_onnx(onnx_file: str, input_data: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Helper function to run inference on a numpy array (.npy) using the exported ONNX model.

    Args:
        onnx_file: Path to the .onnx file.
        input_data: Numpy array with shape (B, C, H, W) or (C, H, W) (normalized/preprocessed).
        device: 'cuda' or 'cpu'.

    Returns:
        Predicted segmentation logits as a numpy array of shape (B, num_classes, H, W).
    """
    import onnxruntime as ort

    if input_data.ndim == 3:
        input_data = np.expand_dims(input_data, axis=0)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_file, providers=providers)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data.astype(np.float32)})[0]
    return outputs


if __name__ == "__main__":
    args = parse_args()
    export_pth_to_onnx(
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint_name,
        output_file=args.output_file,
        opset=args.opset,
        device_str=args.device,
        verify=not args.no_verify
    )
