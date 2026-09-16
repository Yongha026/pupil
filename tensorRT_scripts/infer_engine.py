#!/usr/bin/env python3
"""
Single-Image Parity & Inference Evaluation: TensorRT (.engine) vs PyTorch (.pth).

Runs inference on a single eye image through both the native PyTorch checkpoint
and the compiled TensorRT engine using the exact Pupil Core preprocessing pipeline.
Calculates IoU, class agreement, numerical logit differences, center drift,
ellipse geometry comparisons, and benchmark inference latencies.
Optionally renders a 4-panel diagnostic visualization.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision

# Setup import path for pupil_detector_plugins
current_dir = os.path.dirname(os.path.abspath(__file__))
pupil_root = os.path.abspath(os.path.join(current_dir, ".."))
pupil_shared = os.path.join(pupil_root, "pupil_src", "shared_modules")
if pupil_shared not in sys.path:
    sys.path.insert(0, pupil_shared)

try:
    from pupil_detector_plugins.adgbc.archs_GBC import GBC_Rolling_Unet_S
    from pupil_detector_plugins.trt_detector_wrapper import TRTDetectorModule
except ImportError as err:
    print(f"[Error] Failed to import Pupil detector modules: {err}")
    print(f"Make sure '{pupil_shared}' exists and is accessible.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("infer_engine")


def find_default_file(candidates: list) -> Optional[str]:
    """Finds the first existing file among candidate paths."""
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def parse_args():
    default_image_candidates = [
        os.path.join(pupil_root, "jw_192.png"),
        os.path.join(pupil_root, "jw.png"),
        os.path.join(pupil_root, "model_result_images", "jw780_adgbc.png"),
        os.path.join(pupil_root, "pupil.jpg"),
    ]
    default_pth_candidates = [
        os.path.join(pupil_shared, "pupil_detector_plugins", "model_ckpts", "adgbc_nn_best.pth"),
        os.path.join(pupil_root, "adgbc_nn_best.pth"),
    ]
    default_engine_candidates = [
        os.path.join(pupil_shared, "pupil_detector_plugins", "model_ckpts", "adgbc_nn_best.engine"),
        os.path.join(pupil_root, "adgbc_nn_best.engine"),
        os.path.join(pupil_root, "adgbc_nn_best_32.engine"),
        os.path.join(pupil_shared, "pupil_detector_plugins", "model_ckpts", "adgbc_nn_best_32.engine"),
    ]

    default_image = find_default_file(default_image_candidates)
    default_pth = find_default_file(default_pth_candidates)
    default_engine = find_default_file(default_engine_candidates)

    parser = argparse.ArgumentParser(
        description="Run single-image inference and parity evaluation between PyTorch (.pth) and TensorRT (.engine)."
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=default_image,
        help=f"Path to input image (default: {default_image})",
    )
    parser.add_argument(
        "--pth",
        type=str,
        default=default_pth,
        help=f"Path to PyTorch .pth checkpoint (default: {default_pth})",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=default_engine,
        help=f"Path to TensorRT .engine model (default: {default_engine})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=192,
        help="Spatial input height (default: 192)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=192,
        help="Spatial input width (default: 192)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations for latency benchmark (default: 10)",
    )
    parser.add_argument(
        "--bench_runs",
        type=int,
        default=50,
        help="Number of timed iterations for latency benchmark (default: 50)",
    )
    parser.add_argument(
        "--output_vis",
        type=str,
        default=os.path.join(pupil_root, "model_result_images", "parity_comparison.png"),
        help="Path to save diagnostic visualization figure",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Skip rendering and saving diagnostic visualization",
    )
    return parser.parse_args()


class PupilPreprocessor:
    """Standard Pupil Core preprocessor matching detector_2d_nn_plugin.py."""

    def __init__(self, target_h: int = 192, target_w: int = 192):
        self.target_h = target_h
        self.target_w = target_w
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.table = (float(255) * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

    def preprocess(self, img_gray: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Returns:
            tensor: (1, 1, H, W) normalized float32 torch.Tensor on device
            enhanced_gray: (H, W) uint8 image after resize, gamma LUT, and CLAHE
        """
        h, w = img_gray.shape[:2]
        if h != self.target_h or w != self.target_w:
            resized = cv2.resize(img_gray, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        else:
            resized = img_gray

        gamma = cv2.LUT(resized.astype(np.uint8), self.table)
        enhanced = self.clahe.apply(gamma)
        pil_img = PIL.Image.fromarray(enhanced)
        tensor = self.transform(pil_img).unsqueeze(0).to(device)
        return tensor, enhanced


def load_pytorch_model(pth_path: str, height: int, device: torch.device) -> torch.nn.Module:
    """Loads and initializes PyTorch GBC_Rolling_Unet_S model from checkpoint."""
    if not pth_path or not os.path.isfile(pth_path):
        raise FileNotFoundError(f"PyTorch checkpoint not found at: {pth_path}")

    raw_ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    state_dict = (
        raw_ckpt["network_weights"]
        if (isinstance(raw_ckpt, dict) and "network_weights" in raw_ckpt)
        else raw_ckpt
    )

    num_balls = 32
    if "gbc.centers" in state_dict:
        num_balls = state_dict["gbc.centers"].shape[0]

    model = GBC_Rolling_Unet_S(
        num_classes=4,
        input_channels=1,
        deep_supervision=False,
        img_size=height,
        gbc_num_balls=num_balls,
    ).to(device)

    load_res = model.load_state_dict(state_dict, strict=False)
    if load_res.missing_keys:
        logger.warning(f"PyTorch missing keys: {load_res.missing_keys[:5]}")
    if load_res.unexpected_keys:
        logger.warning(f"PyTorch unexpected keys: {load_res.unexpected_keys[:5]}")

    model.eval()
    return model


def fit_pupil_ellipse(pupil_mask: np.ndarray, raw_conf: float) -> Optional[Dict]:
    """Fits an ellipse to the pupil binary mask matching detector_2d_nn_plugin.py."""
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(best_contour)

    ellipse = None
    if len(hull) >= 5:
        try:
            ellipse = cv2.fitEllipse(hull)
        except Exception:
            ellipse = None
    if ellipse is None and len(best_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(best_contour)
        except Exception:
            ellipse = None

    if ellipse is not None:
        (cx, cy), (d1, d2), angle_deg = ellipse
        if any(np.isnan([cx, cy, d1, d2, angle_deg])) or d1 <= 0 or d2 <= 0:
            ellipse = None

    if ellipse is not None:
        (cx, cy), (d1, d2), angle_deg = ellipse
    elif len(best_contour) >= 3:
        (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
        d1 = d2 = float(radius * 2.0)
        angle_deg = 0.0
        if any(np.isnan([cx, cy, d1, d2, angle_deg])) or radius <= 0:
            return None
    else:
        return None

    if d1 > d2:
        minor_d, major_d = float(d2), float(d1)
        angle_deg = (angle_deg + 90.0) % 180.0
    else:
        minor_d, major_d = float(d1), float(d2)
        angle_deg = angle_deg % 180.0

    area = float(cv2.contourArea(best_contour))
    aspect_ratio = minor_d / (major_d + 1e-6)

    return {
        "center": (float(cx), float(cy)),
        "axes": (minor_d, major_d),
        "angle": float(angle_deg),
        "area": area,
        "aspect_ratio": aspect_ratio,
        "contour": best_contour,
    }


def benchmark_model(model_fn, tensor: torch.Tensor, device: torch.device, warmup: int = 10, runs: int = 50) -> Tuple[float, float]:
    """Measures mean latency and standard deviation in milliseconds."""
    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_fn(tensor)
            if is_cuda:
                torch.cuda.synchronize()

        timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model_fn(tensor)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    return float(np.mean(timings)), float(np.std(timings))


def render_visualization(
    enhanced_img: np.ndarray,
    pred_pyt: np.ndarray,
    pred_trt: np.ndarray,
    ellipse_pyt: Optional[Dict],
    ellipse_trt: Optional[Dict],
    metrics: Dict,
    output_path: str,
):
    """Generates a 4-panel diagnostic comparison figure."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=150)
    fig.patch.set_facecolor("#181818")

    # Colormap for eye segmentation: 0=Background, 1=Sclera, 2=Iris, 3=Pupil
    cmap_seg = plt.cm.get_cmap("viridis", 4)

    # Panel 1: Preprocessed Input
    ax = axes[0]
    ax.set_facecolor("#181818")
    ax.imshow(enhanced_img, cmap="gray")
    ax.set_title("Input (Gamma + CLAHE)", color="white", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Panel 2: PyTorch (.pth) Prediction
    ax = axes[1]
    ax.set_facecolor("#181818")
    ax.imshow(enhanced_img, cmap="gray", alpha=0.6)
    mask_pyt_pupil = (pred_pyt == 3).astype(np.float32)
    ax.imshow(mask_pyt_pupil, cmap="autumn", alpha=0.5 * mask_pyt_pupil)
    if ellipse_pyt is not None:
        (cx, cy) = ellipse_pyt["center"]
        (minor_d, major_d) = ellipse_pyt["axes"]
        angle = ellipse_pyt["angle"]
        e = Ellipse((cx, cy), width=major_d, height=minor_d, angle=angle, fill=False, edgecolor="cyan", linewidth=2)
        ax.add_patch(e)
        ax.plot(cx, cy, "c+", markersize=10, markeredgewidth=2)
        info_txt = f"Pixels: {int(mask_pyt_pupil.sum())}\nCenter: ({cx:.1f}, {cy:.1f})\nAxes: ({minor_d:.1f}, {major_d:.1f})\nTime: {metrics['pyt_time_ms']:.2f} ms"
    else:
        info_txt = "No Ellipse Fitted"
    ax.text(0.04, 0.05, info_txt, transform=ax.transAxes, color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75, edgecolor="cyan"))
    ax.set_title(f"PyTorch (.pth) [Conf: {metrics['pyt_conf']:.3f}]", color="cyan", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Panel 3: TensorRT (.engine) Prediction
    ax = axes[2]
    ax.set_facecolor("#181818")
    ax.imshow(enhanced_img, cmap="gray", alpha=0.6)
    mask_trt_pupil = (pred_trt == 3).astype(np.float32)
    ax.imshow(mask_trt_pupil, cmap="autumn", alpha=0.5 * mask_trt_pupil)
    if ellipse_trt is not None:
        (cx, cy) = ellipse_trt["center"]
        (minor_d, major_d) = ellipse_trt["axes"]
        angle = ellipse_trt["angle"]
        e = Ellipse((cx, cy), width=major_d, height=minor_d, angle=angle, fill=False, edgecolor="magenta", linewidth=2)
        ax.add_patch(e)
        ax.plot(cx, cy, "m+", markersize=10, markeredgewidth=2)
        info_txt = f"Pixels: {int(mask_trt_pupil.sum())}\nCenter: ({cx:.1f}, {cy:.1f})\nAxes: ({minor_d:.1f}, {major_d:.1f})\nTime: {metrics['trt_time_ms']:.2f} ms ({metrics['speedup']:.1f}x)"
    else:
        info_txt = "No Ellipse Fitted"
    ax.text(0.04, 0.05, info_txt, transform=ax.transAxes, color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75, edgecolor="magenta"))
    ax.set_title(f"TensorRT (.engine) [Conf: {metrics['trt_conf']:.3f}]", color="magenta", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Panel 4: Parity / Overlap Difference Map
    ax = axes[3]
    ax.set_facecolor("#181818")
    diff_rgb = np.zeros((*pred_pyt.shape, 3), dtype=np.float32)
    p_pupil = pred_pyt == 3
    t_pupil = pred_trt == 3
    diff_rgb[p_pupil & t_pupil] = [0.0, 1.0, 0.0]     # Green = Agreement (TP)
    diff_rgb[p_pupil & ~t_pupil] = [1.0, 0.2, 0.2]    # Red = PyTorch Only (FN)
    diff_rgb[~p_pupil & t_pupil] = [0.2, 0.4, 1.0]    # Blue = TensorRT Only (FP)

    ax.imshow(enhanced_img, cmap="gray", alpha=0.4)
    ax.imshow(diff_rgb, alpha=0.75)

    # Plot center drift vector
    if ellipse_pyt is not None and ellipse_trt is not None:
        p_c = ellipse_pyt["center"]
        t_c = ellipse_trt["center"]
        ax.annotate(
            "", xy=t_c, xytext=p_c,
            arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5, mutation_scale=15)
        )
        ax.plot(p_c[0], p_c[1], "co", markersize=6, label="PyTorch Center")
        ax.plot(t_c[0], t_c[1], "mo", markersize=6, label="TensorRT Center")

    diff_summary = (
        f"Pupil IoU:        {metrics['iou_pct']:.2f}%\n"
        f"All-Class Agree:  {metrics['agreement_pct']:.2f}%\n"
        f"Center Drift:     {metrics['center_drift_px']:.2f} px\n"
        f"Logit Max Diff:   {metrics['max_logit_diff']:.4f}\n"
        f"Logit Mean Diff:  {metrics['mean_logit_diff']:.4f}"
    )
    ax.text(0.04, 0.05, diff_summary, transform=ax.transAxes, color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75, edgecolor="yellow"))
    ax.set_title("Parity (Green=Match, Red=PyT, Blue=TRT)", color="yellow", fontsize=12, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"[+] Diagnostic visual report saved to: {output_path}")


def run_inference():
    args = parse_args()

    if not args.image or not os.path.isfile(args.image):
        logger.error(f"Image path does not exist: {args.image}")
        sys.exit(1)
    if not args.pth or not os.path.isfile(args.pth):
        logger.error(f"PyTorch checkpoint path does not exist: {args.pth}")
        sys.exit(1)
    if not args.engine or not os.path.isfile(args.engine):
        logger.error(f"TensorRT engine path does not exist: {args.engine}")
        sys.exit(1)

    device = torch.device(args.device)
    logger.info("=" * 70)
    logger.info(" AD-GBC Single Image Parity Evaluation: PyTorch vs TensorRT")
    logger.info("=" * 70)
    logger.info(f"Target Device:     {device}")
    logger.info(f"Input Image:       {args.image}")
    logger.info(f"PyTorch (.pth):    {args.pth}")
    logger.info(f"TensorRT (.engine):{args.engine}")
    logger.info(f"Target Shape:      ({args.height}, {args.width})")

    # 1. Load Raw Image & Preprocess
    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        logger.error(f"Failed to load image via OpenCV: {args.image}")
        sys.exit(1)

    preprocessor = PupilPreprocessor(target_h=args.height, target_w=args.width)
    tensor, enhanced_img = preprocessor.preprocess(img_gray, device)
    logger.info(f"Input image loaded: orig_shape={img_gray.shape} -> preprocessed_tensor={tuple(tensor.shape)}")

    # 2. PyTorch Inference
    logger.info("[*] Loading PyTorch model...")
    pyt_mod = load_pytorch_model(args.pth, height=args.height, device=device)
    with torch.no_grad():
        out_pyt = pyt_mod(tensor)
        if isinstance(out_pyt, tuple):
            out_pyt = out_pyt[0]

    # Benchmark PyTorch
    pyt_time_ms, pyt_std_ms = benchmark_model(pyt_mod, tensor, device, warmup=args.warmup, runs=args.bench_runs)

    # 3. TensorRT Inference
    logger.info("[*] Loading TensorRT engine...")
    trt_mod = TRTDetectorModule(engine_path=args.engine, device=device)
    out_trt = trt_mod(tensor)

    # Benchmark TensorRT
    trt_time_ms, trt_std_ms = benchmark_model(trt_mod, tensor, device, warmup=args.warmup, runs=args.bench_runs)
    speedup = pyt_time_ms / max(1e-4, trt_time_ms)

    # 4. Post-processing & Metric Computations
    probs_pyt = torch.softmax(out_pyt, dim=1)
    probs_trt = torch.softmax(out_trt, dim=1)

    pred_pyt = torch.argmax(probs_pyt, dim=1)[0].detach().cpu().numpy()
    pred_trt = torch.argmax(probs_trt, dim=1)[0].detach().cpu().numpy()

    pyt_pupil_pixels = (pred_pyt == 3)
    trt_pupil_pixels = (pred_trt == 3)

    pyt_pupil_count = int(pyt_pupil_pixels.sum())
    trt_pupil_count = int(trt_pupil_pixels.sum())

    pyt_conf = float(np.mean(probs_pyt[0, 3].detach().cpu().numpy()[pyt_pupil_pixels])) if pyt_pupil_count > 0 else 0.0
    trt_conf = float(np.mean(probs_trt[0, 3].detach().cpu().numpy()[trt_pupil_pixels])) if trt_pupil_count > 0 else 0.0

    # IoU calculation for pupil class (3)
    intersection = int((pyt_pupil_pixels & trt_pupil_pixels).sum())
    union = int((pyt_pupil_pixels | trt_pupil_pixels).sum())
    iou_pct = (intersection / float(union) * 100.0) if union > 0 else (100.0 if pyt_pupil_count == trt_pupil_count == 0 else 0.0)

    # Overall multi-class agreement across all pixels
    agreement_pct = float(np.mean(pred_pyt == pred_trt) * 100.0)

    # Numerical differences on unnormalized logits
    out_pyt_np = out_pyt.detach().cpu().numpy()
    out_trt_np = out_trt.detach().cpu().numpy()
    max_logit_diff = float(np.max(np.abs(out_pyt_np - out_trt_np)))
    mean_logit_diff = float(np.mean(np.abs(out_pyt_np - out_trt_np)))
    rmse_logit = float(np.sqrt(np.mean((out_pyt_np - out_trt_np) ** 2)))

    # Ellipse fitting
    mask_pyt = np.zeros_like(pred_pyt, dtype=np.uint8)
    mask_pyt[pyt_pupil_pixels] = 255
    ellipse_pyt = fit_pupil_ellipse(mask_pyt, pyt_conf)

    mask_trt = np.zeros_like(pred_trt, dtype=np.uint8)
    mask_trt[trt_pupil_pixels] = 255
    ellipse_trt = fit_pupil_ellipse(mask_trt, trt_conf)

    # Geometric drift
    center_drift_px = 0.0
    if ellipse_pyt is not None and ellipse_trt is not None:
        cx_p, cy_p = ellipse_pyt["center"]
        cx_t, cy_t = ellipse_trt["center"]
        center_drift_px = float(np.sqrt((cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2))

    # 5. Print Results Table
    print("\n" + "=" * 70)
    print("                      EVALUATION RESULTS")
    print("=" * 70)
    print(f" [A] PyTorch (.pth) Inference:")
    print(f"     - Pupil Pixel Count:   {pyt_pupil_count} pixels")
    print(f"     - Mean Confidence:     {pyt_conf:.4f}")
    print(f"     - Latency:             {pyt_time_ms:.2f} ms (+/- {pyt_std_ms:.2f} ms)")
    if ellipse_pyt:
        print(f"     - Fitted Center (X,Y): ({ellipse_pyt['center'][0]:.2f}, {ellipse_pyt['center'][1]:.2f})")
        print(f"     - Fitted Axes (m, M):  ({ellipse_pyt['axes'][0]:.2f}, {ellipse_pyt['axes'][1]:.2f}) px")
        print(f"     - Ellipse Angle:       {ellipse_pyt['angle']:.1f} deg")
    else:
        print(f"     - Fitted Ellipse:      None (rejection triggered)")

    print("-" * 70)
    print(f" [B] TensorRT (.engine) Inference:")
    print(f"     - Pupil Pixel Count:   {trt_pupil_count} pixels")
    print(f"     - Mean Confidence:     {trt_conf:.4f}")
    print(f"     - Latency:             {trt_time_ms:.2f} ms (+/- {trt_std_ms:.2f} ms)")
    print(f"     - Speedup:             {speedup:.1f}x vs PyTorch")
    if ellipse_trt:
        print(f"     - Fitted Center (X,Y): ({ellipse_trt['center'][0]:.2f}, {ellipse_trt['center'][1]:.2f})")
        print(f"     - Fitted Axes (m, M):  ({ellipse_trt['axes'][0]:.2f}, {ellipse_trt['axes'][1]:.2f}) px")
        print(f"     - Ellipse Angle:       {ellipse_trt['angle']:.1f} deg")
    else:
        print(f"     - Fitted Ellipse:      None (rejection triggered)")

    print("-" * 70)
    print(f" [C] Parity & Concordance Metrics:")
    print(f"     - Pupil IoU:           {iou_pct:.2f}% (Intersection={intersection}, Union={union})")
    print(f"     - All-Class Agreement: {agreement_pct:.2f}%")
    print(f"     - Center Drift:        {center_drift_px:.2f} px")
    print(f"     - Logit Max Diff:      {max_logit_diff:.6f}")
    print(f"     - Logit Mean Diff:     {mean_logit_diff:.6f}")
    print(f"     - Logit RMSE:          {rmse_logit:.6f}")
    print("=" * 70 + "\n")

    # 6. Render Visualization
    metrics = {
        "pyt_conf": pyt_conf,
        "trt_conf": trt_conf,
        "pyt_time_ms": pyt_time_ms,
        "trt_time_ms": trt_time_ms,
        "speedup": speedup,
        "iou_pct": iou_pct,
        "agreement_pct": agreement_pct,
        "center_drift_px": center_drift_px,
        "max_logit_diff": max_logit_diff,
        "mean_logit_diff": mean_logit_diff,
    }

    if not args.no_vis:
        render_visualization(
            enhanced_img=enhanced_img,
            pred_pyt=pred_pyt,
            pred_trt=pred_trt,
            ellipse_pyt=ellipse_pyt,
            ellipse_trt=ellipse_trt,
            metrics=metrics,
            output_path=args.output_vis,
        )


if __name__ == "__main__":
    run_inference()
