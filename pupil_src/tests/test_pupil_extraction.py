import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt

# Add pupil_src/shared_modules/pupil_detector_plugins to path
plugin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared_modules', 'pupil_detector_plugins'))
sys.path.append(plugin_dir)

import adgbc
from utils import get_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Test pupil extraction using AD-GBC offline on static images")
    parser.add_argument('--img_path', required=True, help="Path to input eye image")
    parser.add_argument('--ckpt_path', default=os.path.join(plugin_dir, "adgbc_nn_best.pth"), help="Path to AD-GBC weights")
    # parser.add_argument('--arch', default="GBC_Rolling_Unet_L", choices=["GBC_Rolling_Unet_S", "GBC_Rolling_Unet_M", "GBC_Rolling_Unet_L"], help="Model architecture")
    parser.add_argument('--channels', type=int, default=1, choices=[1, 3], help="Model input channels")
    parser.add_argument('--save_path', default="test_adgbc.png", help="Path to save visual results")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")
    img = cv2.imread(args.img_path)
    img_bgr = cv2.resize(img, (192,192),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)


    # 1. Instantiate model
    # print(f"Creating model: {args.arch} with {args.channels} input channels")
    # if args.arch == "GBC_Rolling_Unet_L":
    #     model = adgbc.GBC_Rolling_Unet_L(num_classes=4, input_channels=args.channels, deep_supervision=False)
    # elif args.arch == "GBC_Rolling_Unet_M":
    #     model = adgbc.GBC_Rolling_Unet_M(num_classes=4, input_channels=args.channels, deep_supervision=False)
    # else:
    #     model = adgbc.GBC_Rolling_Unet_S(num_classes=4, input_channels=args.channels, deep_supervision=False)
    print(f"Creating model: GBC_Rolling_UNet_L with {args.channels} input channels")
    model = adgbc.GBC_Rolling_Unet_L(num_classes=4, input_channels=args.channels, deep_supervision=False)
    # 2. Load weights
    if not os.path.exists(args.ckpt_path):
        print(f"Warning: Checkpoint not found at {args.ckpt_path}. Running with random weights.")
    else:
        print(f"Loading weights from: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "network_weights" in checkpoint:
            state_dict = checkpoint["network_weights"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Preprocess image (Gamma + CLAHE + ToTensor + Normalize)
    # Gamma Correction
    table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    img_gamma = cv2.LUT(gray, table.astype(np.uint8))
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gamma)
    # Convert to Tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5]),
    ])
    pil_img = PIL.Image.fromarray(img_clahe)
    tensor = transform(pil_img).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Replicate channels if model was trained with 3 channels
    if args.channels == 3:
        tensor = tensor.repeat(1, 3, 1, 1)

    # 4. Model Inference
    print("Running inference...")
    with torch.no_grad():
        output = model(tensor)
    
    # Extract prediction map
    predict = get_predictions(output)  # (1, H, W)
    predict_2d = predict[0].cpu().numpy()  # (H, W)

    # 5. Extract pupil mask (class index 3)
    pupil_mask = np.zeros_like(predict_2d, dtype=np.uint8)
    pupil_mask[predict_2d == 3] = 255

    # 6. Fit Ellipse
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipse_found = False
    result_ellipse = None
    if contours:
        best_contour = max(contours, key=cv2.contourArea)
        if len(best_contour) >= 5:
            ellipse = cv2.fitEllipse(best_contour)  # ((cx, cy), (MA, ma), angle)
            result_ellipse = ellipse
            ellipse_found = True
            print("Pupil ellipse fit found:")
            print(f"  Center:     {ellipse[0]}")
            print(f"  Axes sizes: {ellipse[1]}")
            print(f"  Angle:      {ellipse[2]} degrees")
        else:
            print("Best pupil contour has fewer than 5 points; cannot fit ellipse.")
    else:
        print("No pupil contours found.")

    # 7. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Frame")
    axes[0].axis('off')
    
    # Prediction segmentation map
    axes[1].imshow(predict_2d, cmap='tab10')
    axes[1].set_title("AD-GBC Segmentation Map")
    axes[1].axis('off')
    
    # Original Image with overlay
    overlay = img_bgr.copy()
    # Draw segmentation mask as semi-transparent red overlay
    overlay[predict_2d == 3] = [0, 0, 255]
    img_result = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
    
    if ellipse_found:
        # Draw the fitted ellipse in green
        cv2.ellipse(img_result, result_ellipse, (0, 255, 0), 2)
        # Draw center point
        center = (int(result_ellipse[0][0]), int(result_ellipse[0][1]))
        cv2.circle(img_result, center, 3, (255, 0, 0), -1)

    axes[2].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Fitted Pupil Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Results saved to: {args.save_path}")

if __name__ == '__main__':
    main()
