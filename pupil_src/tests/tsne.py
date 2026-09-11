import torch

import os
import cv2
import sys
import numpy as np
import glob
import argparse
import torchvision
import PIL.Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

clahe = cv2.createCLAHE(
    clipLimit=1.5, tileGridSize=(8, 8)
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

tests_dir = os.path.dirname(os.path.abspath(__file__))
plugins_dir = os.path.abspath(os.path.join(tests_dir, "..", "shared_modules", "pupil_detector_plugins"))
if plugins_dir not in sys.path:
    sys.path.append(plugins_dir)


device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

model_path_adgbc = os.path.join(plugins_dir, "model_ckpts","adgbc_nn_best.pth")

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

model.eval().to(device_str)


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
        self.table = float(255) * (np.linspace(0, 1, 256) ** 0.8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
        img_gamma = cv2.LUT(img_resized.astype(np.uint8), self.table.astype(np.uint8))
        img_clahe = self.clahe.apply(img_gamma)
        pil_img = PIL.Image.fromarray(img_clahe)
        return self.transform(pil_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
    args = parser.parse_args()

    image_path = os.path.join(args.IMG_PATH, "*.png")
    full_images = glob.glob(image_path)
    rand_images = random.sample(full_images, 96)

    dataset = ImageDataset(rand_images)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

    enc_features = []
    dec_features = []

    with torch.no_grad():
        for batch_imgs in tqdm(dataloader):
            batch_imgs = batch_imgs.to(device)
            _, enc, dec = model(batch_imgs)

            # t-SNE를 위해 (Batch, Channel, H, W) 형태를 (Batch, Features) 2D 형태로 변환
            enc_flat = enc.view(enc.size(0), -1).cpu().numpy()
            dec_flat = dec.view(dec.size(0), -1).cpu().numpy()

            enc_features.append(enc_flat)
            dec_features.append(dec_flat)

    encoders = np.vstack(enc_features)
    decoders = np.vstack(dec_features)

    # PCA를 활용한 차원 축소 (t-SNE 속도 대폭 향상)
    pca = PCA(n_components=50)
    encoders_pca = pca.fit_transform(encoders)
    decoders_pca = pca.fit_transform(decoders)

    # n_jobs=-1 로 멀티코어 연산 활용
    tsne_enc = TSNE(n_components=2, n_jobs=-1).fit_transform(encoders_pca)
    tsne_dec = TSNE(n_components=2, n_jobs=-1).fit_transform(decoders_pca)


    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range


    # 인코더 플롯 저장
    tx_enc = scale_to_01_range(tsne_enc[:, 0])
    ty_enc = scale_to_01_range(tsne_enc[:, 1])
    plt.figure()
    plt.scatter(tx_enc, ty_enc, color='skyblue', label='CULane')
    plt.legend()
    plt.savefig("Enc_features.png")
    plt.close()

    # 디코더 플롯 저장
    tx_dec = scale_to_01_range(tsne_dec[:, 0])
    ty_dec = scale_to_01_range(tsne_dec[:, 1])
    plt.figure()
    plt.scatter(tx_dec, ty_dec, color='skyblue', label='CULane')
    plt.legend()
    plt.savefig("Dec_features.png")
    plt.close()