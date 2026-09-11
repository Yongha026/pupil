import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

tests_dir = os.path.dirname(os.path.abspath(__file__))
plugins_dir = os.path.abspath(os.path.join(tests_dir, "..", "shared_modules", "pupil_detector_plugins"))
if plugins_dir not in sys.path:
    sys.path.append(plugins_dir)

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

model_path_adgbc = os.path.join(plugins_dir, "model_ckpts", "adgbc_nn_best.pth")

import adgbc

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

def get_img(img_path: str) -> torch.Tensor:
    import torchvision
    import PIL.Image

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5]),
    ])
    table = float(255) * (np.linspace(0, 1, 256) ** 0.8)
    img_gamma = cv2.LUT(img_resized.astype(np.uint8), table.astype(np.uint8))
    img_clahe = clahe.apply(img_gamma)
    pil_img = PIL.Image.fromarray(img_clahe)
    return transform(pil_img)

def get_msk(mask_path: str) -> np.ndarray:
    msk = np.load(mask_path).astype(np.uint8)
    msk_resized = cv2.resize(msk, (48, 48), interpolation=cv2.INTER_NEAREST)
    return msk_resized

IMGPATH = "/mnt/nas/03. Gaze_tracking_2024 (IITP 2024-2027)/Gaze_public_dataset/near/OpenEDS/Openedsdata2019/Openedsdata2019/Semantic_Segmentation_Dataset/Semantic_Segmentation_Dataset/train/images/000000002610.png"
MSKPATH = "/mnt/nas/03. Gaze_tracking_2024 (IITP 2024-2027)/Gaze_public_dataset/near/OpenEDS/Openedsdata2019/Openedsdata2019/Semantic_Segmentation_Dataset/Semantic_Segmentation_Dataset/train/labels/000000002610.npy"

# 1. 추론 및 피처맵 추출
img_tensor = get_img(IMGPATH).unsqueeze(0).to(device)
with torch.no_grad():
    _, enc, _ = model(img_tensor)

# shape: (64, 48, 48)
enc_features = enc.squeeze(0).detach().cpu().numpy()

# 2. 마스크 로드 및 RGB 컬러 마스크 생성
mask = get_msk(MSKPATH)
h, w = mask.shape

# Matplotlib 표시용 RGB 색상 (배경: 검정, sclera: 빨강, iris: 초록, pupil: 파랑)
colors = [
    [0, 0, 0],       # Class 0: Background
    [255, 0, 0],     # Class 1: Sclera (Red)
    [0, 255, 0],     # Class 2: Iris (Green)
    [0, 0, 255],     # Class 3: Pupil (Blue)
]

color_mask = np.zeros((h, w, 3), dtype=np.uint8)
for class_idx, color in enumerate(colors):
    color_mask[mask == class_idx] = color

# 3. 8x8 그리드 시각화
fig, axes = plt.subplots(8, 8, figsize=(20, 20))
alpha = 0.6
beta = 1.0 - alpha

for idx, ax in enumerate(axes.flat):
    feat = enc_features[idx]

    # 채널별 피처를 [0, 255] uint8로 Min-Max 스케일링
    feat_min, feat_max = feat.min(), feat.max()
    if feat_max > feat_min:
        feat_norm = ((feat - feat_min) / (feat_max - feat_min) * 255.0).astype(np.uint8)
    else:
        feat_norm = np.zeros_like(feat, dtype=np.uint8)

    # 1채널 흑백을 3채널 RGB로 확장
    feat_rgb = cv2.cvtColor(feat_norm, cv2.COLOR_GRAY2RGB)

    # 마스크와 알파 블렌딩
    blended = cv2.addWeighted(feat_rgb, alpha, color_mask, beta, 0)

    ax.imshow(blended)
    ax.set_title(f"Ch {idx}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("blended_features_8x8.png", dpi=200, bbox_inches="tight")
plt.close()