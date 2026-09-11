import torch
import torchvision.models as models
from ptflops import get_model_complexity_info
import os
import cv2
import sys
import numpy as np
import glob
import argparse

from sklearn.manifold import TSNE
tests_dir = os.path.dirname(os.path.abspath(__file__))
plugins_dir = os.path.abspath(os.path.join(tests_dir, "..", "shared_modules", "pupil_detector_plugins"))
if plugins_dir not in sys.path:
    sys.path.append(plugins_dir)


plugin_dir = "../shared_modules/pupil_detector_plugins"
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

model_path_adgbc = os.path.join(plugin_dir, "adgbc_s_nn_best.pth")

# Import AD-GBC Model
import adgbc

# adgbc_encoder.py 만들어서 인코더까지만 로드

try:
    model = adgbc.GBC_S_EncDec(num_classes=4, input_channels=1, deep_supervision=False).to(device)
    if os.path.exists(model_path_adgbc):
        checkpoint = torch.load(model_path_adgbc, map_location=device, weights_only=False)
        state_dict = checkpoint["network_weights"] if (
                    isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(f"ADGBC ckpt file not found at {model_path_adgbc}")
except Exception as e:
    print(f"Error loading adgbc: {e}")
    raise e

model.eval()


def get_img(self, img: np.ndarray) -> torch.Tensor:
    import torchvision
    import PIL.Image

    img_resized = cv2.resize(img, (192,192),interpolation=cv2.INTER_AREA)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )
    table = float(255) * (np.linspace(0, 1, 256) ** 0.8)
    img_gamma = cv2.LUT(img_resized.astype(np.uint8), table.astype(np.uint8))
    img_clahe = self.clahe.apply(img_gamma)
    pil_img = PIL.Image.fromarray(img_clahe)
    return transform(pil_img)

parser = argparse.ArgumentParser()
parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
args = parser.parse_args()

image_path = os.path.join(args.IMG_PATH, "*.png")
images = glob.glob(image_path)
# print(len(images))



