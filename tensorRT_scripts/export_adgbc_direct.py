"""
Direct Checkpoint to ONNX Exporter for AD-GBC.
Exports directly from the exact .pth file loaded by Pupil Core (e.g. adgbc_nn_best.pth),
ensuring 100% weight identity between the PyTorch detector and the exported ONNX model.
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add Pupil shared_modules to path
current_dir = os.path.dirname(os.path.abspath(__file__))
pupil_shared_dir = os.path.abspath(os.path.join(current_dir, "..", "pupil_src", "shared_modules"))
if pupil_shared_dir not in sys.path:
    sys.path.insert(0, pupil_shared_dir)

try:
    from pupil_detector_plugins.adgbc.archs_GBC import GBC_Rolling_Unet_S
except ImportError as e:
    print(f"[Error] Failed to import GBC_Rolling_Unet_S from {pupil_shared_dir}: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Directly export AD-GBC .pth checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pupil_src/shared_modules/pupil_detector_plugins/model_ckpts/adgbc_nn_best.pth",
        help="Path to .pth checkpoint file."
    )
    parser.add_argument(
        "--output_onnx",
        type=str,
        default="pupil_src/shared_modules/pupil_detector_plugins/model_ckpts/adgbc_nn_best.onnx",
        help="Path to output .onnx file."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=192,
        help="Input spatial height. Default: 192"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=192,
        help="Input spatial width. Default: 192"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for export (e.g. cuda:0 or cpu)."
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip ONNX Runtime numerical verification."
    )
    return parser.parse_args()


def _patch_lo2_for_tensorrt(network: torch.nn.Module):
    """
    In-memory patch: Replaces 'torch.chunk' with direct channel slicing
    to prevent PyTorch from emitting ONNX 'SequenceAt' / 'SplitToSequence' nodes.
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
        print(f"[*] In-memory patched {patched} Lo2 layer(s) for TensorRT compatibility.")


def export_adgbc(
    checkpoint_path: str,
    output_onnx_path: str,
    height: int = 192,
    width: int = 192,
    opset: int = 17,
    device_str: str = "cuda:0",
    verify: bool = True
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")
    print(f"[*] Target device: {device}")
    print(f"[*] Loading checkpoint from: {checkpoint_path}...")

    raw_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        raw_ckpt["network_weights"]
        if (isinstance(raw_ckpt, dict) and "network_weights" in raw_ckpt)
        else raw_ckpt
    )

    # Detect gbc_num_balls from centers shape if present
    num_balls = 32
    if "gbc.centers" in state_dict:
        num_balls = state_dict["gbc.centers"].shape[0]
        print(f"[*] Auto-detected gbc_num_balls={num_balls} from checkpoint state_dict.")

    print(f"[*] Instantiating GBC_Rolling_Unet_S (num_classes=4, input_channels=1, gbc_num_balls={num_balls}, img_size={height})...")
    model = GBC_Rolling_Unet_S(
        num_classes=4,
        input_channels=1,
        deep_supervision=False,
        img_size=height,
        gbc_num_balls=num_balls
    ).to(device)

    load_res = model.load_state_dict(state_dict, strict=False)
    print(f"[+] Loaded weights into model. Missing keys: {len(load_res.missing_keys)}, Unexpected keys: {len(load_res.unexpected_keys)}")
    if load_res.missing_keys:
        print(f"    Missing keys preview: {load_res.missing_keys[:5]}")
    if load_res.unexpected_keys:
        print(f"    Unexpected keys preview: {load_res.unexpected_keys[:5]}")

    model.eval()

    # Apply in-memory patch for ONNX export
    _patch_lo2_for_tensorrt(model)

    dummy_input = torch.randn(1, 1, height, width, device=device)

    print("[*] Running test PyTorch forward pass...")
    with torch.no_grad():
        pyt_out = model(dummy_input)
        if isinstance(pyt_out, tuple):
            pyt_out = pyt_out[0]
    print(f"[+] PyTorch forward successful. Output shape: {pyt_out.shape}")

    out_dir = os.path.dirname(os.path.abspath(output_onnx_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Exporting ONNX to: {output_onnx_path} (opset {opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
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
    print(f"[+] Export complete: {output_onnx_path}")

    if verify:
        try:
            import onnxruntime as ort
            print("\n" + "=" * 60)
            print(" [*] Verifying Export with ONNX Runtime")
            print("=" * 60)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(output_onnx_path, providers=providers)
            ort_out = session.run(None, {"input": dummy_input.detach().cpu().numpy()})[0]

            pyt_np = pyt_out.detach().cpu().numpy()
            max_diff = float(np.max(np.abs(pyt_np - ort_out)))
            mean_diff = float(np.mean(np.abs(pyt_np - ort_out)))
            agree = float(np.mean(np.argmax(pyt_np, axis=1) == np.argmax(ort_out, axis=1)) * 100.0)

            print(f"[+] Numerical Parity (PyTorch vs ORT {session.get_providers()[0]} on Random Tensor):")
            print(f"    - Max absolute difference:  {max_diff:.6f}")
            print(f"    - Mean absolute difference: {mean_diff:.6f}")
            print(f"    - Class label agreement:    {agree:.2f}%")

            # Verify on real eye image if available
            real_img_candidates = [
                os.path.join(pupil_shared_dir, "..", "..", "jw_192.png"),
                os.path.join(pupil_shared_dir, "..", "..", "jw.png"),
            ]
            real_img_path = next((p for p in real_img_candidates if os.path.isfile(p)), None)
            if real_img_path:
                import cv2
                import torchvision
                import PIL.Image
                raw_im = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)
                if raw_im is not None:
                    if raw_im.shape[:2] != (height, width):
                        raw_im = cv2.resize(raw_im, (width, height), interpolation=cv2.INTER_AREA)
                    table = (float(255) * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    enhanced = clahe.apply(cv2.LUT(raw_im, table))
                    tf = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize([0.5], [0.5]),
                    ])
                    real_tensor = tf(PIL.Image.fromarray(enhanced)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pyt_real = model(real_tensor)
                        if isinstance(pyt_real, tuple):
                            pyt_real = pyt_real[0]
                    ort_real = session.run(None, {"input": real_tensor.detach().cpu().numpy()})[0]
                    pyt_real_np = pyt_real.detach().cpu().numpy()
                    real_max_diff = float(np.max(np.abs(pyt_real_np - ort_real)))
                    real_agree = float(np.mean(np.argmax(pyt_real_np, axis=1) == np.argmax(ort_real, axis=1)) * 100.0)
                    p_pyt = int((np.argmax(pyt_real_np, axis=1) == 3).sum())
                    p_ort = int((np.argmax(ort_real, axis=1) == 3).sum())
                    print(f"\n[+] Real Eye Image Parity ({os.path.basename(real_img_path)}):")
                    print(f"    - Max logit difference:     {real_max_diff:.6f}")
                    print(f"    - Class label agreement:    {real_agree:.2f}%")
                    print(f"    - Pupil pixels (PyT vs ORT): {p_pyt} vs {p_ort}")
            print("=" * 60 + "\n")
        except ImportError:
            print("[!] onnxruntime not installed, skipping verification.")
        except Exception as e:
            print(f"[!] Verification warning: {e}")


if __name__ == "__main__":
    args = parse_args()
    export_adgbc(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output_onnx,
        height=args.height,
        width=args.width,
        opset=args.opset,
        device_str=args.device,
        verify=not args.no_verify
    )
