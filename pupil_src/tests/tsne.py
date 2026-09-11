import torch
import torch.nn.functional as F

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
from matplotlib.colors import ListedColormap
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
        # self.mask_paths = str(image_paths).replace("images","labels").replace("png","npy")
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
        self.table = float(255) * (np.linspace(0, 1, 256) ** 0.8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Image
        img_path = self.image_paths[idx]
        if img_path.endswith("_0000.png"):
            img_path.replace("_0000","")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
        img_gamma = cv2.LUT(img_resized.astype(np.uint8), self.table.astype(np.uint8))
        img_clahe = self.clahe.apply(img_gamma)
        pil_img = PIL.Image.fromarray(img_clahe)

        # Mask
        msk_path = self.image_paths[idx].replace("images","labels").replace("png","npy")
        msk = np.load(msk_path).astype(np.uint8)
        msk_resized = cv2.resize(msk, (48, 48), interpolation=cv2.INTER_NEAREST)
        pil_mask = PIL.Image.fromarray(msk_resized)
        return self.transform(pil_img), msk_resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
    parser.add_argument("--datas", default=16, type=int, help="Number of images to use")
    args = parser.parse_args()

    image_path = os.path.join(args.IMG_PATH, "*.png")
    full_images = glob.glob(image_path)
    try:
        rand_images = random.sample(full_images, args.datas)
    except ValueError:
        rand_images = full_images

    dataset = ImageDataset(rand_images)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    SAMPLES_PER_CLASS = 16  # 클래스가 4개면 총 8,000개의 포인트로 t-SNE 수행
    NUM_CLASSES = 4

    features_by_class = {c: [] for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for batch_imgs, batch_masks in tqdm(dataloader):
            batch_imgs = batch_imgs.to(device)
            _, enc, _ = model(batch_imgs)  # [B, 64, 48, 48]

            B, C, H, W = enc.shape


            # 2. [B, C, H, W] -> [B * H * W, C] 형태로 변환 (각 픽셀이 64차원 벡터)
            enc_pixels = enc.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels_pixels = batch_masks.view(-1).cpu().numpy()

            # 3. 클래스별로 필터링하여 리스트에 축적
            for c in range(NUM_CLASSES):
                current_len = sum(len(x) for x in features_by_class[c])
                if current_len < SAMPLES_PER_CLASS:
                    c_feats = enc_pixels[labels_pixels == c]
                    if len(c_feats) > 0:
                        features_by_class[c].append(c_feats)

    # 클래스별로 모인 feature를 샘플링 및 병합
    selected_feats = []
    selected_labels = []

    for c in range(NUM_CLASSES):
        c_all = np.vstack(features_by_class[c])
        if len(c_all) > SAMPLES_PER_CLASS:
            idx = np.random.choice(len(c_all), SAMPLES_PER_CLASS, replace=False)
            c_all = c_all[idx]
        selected_feats.append(c_all)
        selected_labels.append(np.full(len(c_all), c))

    X = np.vstack(selected_feats)  # [총 점 수 (약 8,000), 64]
    y = np.concatenate(selected_labels)

    # 차원이 이미 64이므로 PCA 없이 바로 t-SNE 수행
    tsne = TSNE(n_components=2, n_jobs=-1, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # custom cmap from tab10
    tab10_cmap = plt.cm.get_cmap('tab10')
    four_colours_cmap = ListedColormap(tab10_cmap.colors[0:4])

    # 시각화
    plt.figure(figsize=(10, 8))

    # 클래스가 0, 1, 2, 3 이므로 vmin, vmax를 명시하여 색상을 4개 구간으로 깔끔하게 매핑
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y,
        cmap=four_colours_cmap,
        # vmin=-0.5,
        # vmax=9.5,  # tab10 컬러맵의 전체 인덱스 범위 지정
        alpha=0.5,
        s=5
    )

    # 1. ticks에는 숫자를 전달
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3])

    # 2. 텍스트 라벨은 set_yticklabels로 별도 지정
    cbar.ax.set_yticklabels(['Background', 'Sclera', 'Iris', 'Pupil'])
    cbar.set_label('Class ID')

    plt.title("Pixel-wise Deep Feature Clustering (t-SNE)")

    plt.savefig("PupilLabs_Enc_pixel_features_tsne.png", dpi=300)
    plt.close()