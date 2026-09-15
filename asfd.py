PYTHONPATH=pupil_src/shared_modules/ python -c "
import torch, cv2, numpy as np
from pupil_detector_plugins.trt_detector_wrapper import TRTDetectorModule
import torchvision, PIL.Image

device = torch.device('cuda:0')
engine_path = 'pupil_src/shared_modules/pupil_detector_plugins/model_ckpts/adgbc_nn_best.engine'
img = cv2.imread('jw_192.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    img = cv2.imread('model_result_images/jw780_adgbc.png', cv2.IMREAD_GRAYSCALE)
img = cv2.flip(img, -1)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
table = float(255) * (np.linspace(0, 1, 256) ** 0.8)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5]),
])
img_gamma = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))
img_clahe = clahe.apply(img_gamma)
tensor = transform(PIL.Image.fromarray(img_clahe)).unsqueeze(0).to(device)

trt_mod = TRTDetectorModule(engine_path, device)
out_trt = trt_mod(tensor)
probs = torch.softmax(out_trt, dim=1)
pred = torch.argmax(probs, dim=1)[0].cpu().numpy()

pupil_pixels = (pred == 3)
pupil_probs = probs[0, 3].detach().cpu().numpy()[pupil_pixels]
raw_conf = float(np.mean(pupil_probs)) if np.any(pupil_pixels) else 0.0

print(f'[*] Pupil pixels before blur: {int(pupil_pixels.sum())}')
print(f'[*] Raw confidence:          {raw_conf:.4f} (threshold is 0.60)')

# Gate 1: Check Gaussian Blur
mask_raw = np.zeros_like(pred, dtype=np.uint8)
mask_raw[pupil_pixels] = 255

c_raw, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'[*] Contours WITHOUT blur:   {len(c_raw)}')

blur_mask = cv2.GaussianBlur(mask_raw, (5, 5), 0)
_, thresh_mask = cv2.threshold(blur_mask, 127, 255, cv2.THRESH_BINARY)
c_blur, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'[*] Contours WITH (5,5) blur:{len(c_blur)}')

if c_raw:
    cnt = max(c_raw, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    print(f'[*] Hull point count:        {len(hull)} (needs >= 5 for ellipse)')
    if len(hull) >= 5:
        ellipse = cv2.fitEllipse(hull)
        print(f'[+] Ellipse fit success! Center={ellipse[0]}, Axes={ellipse[1]}')
"