import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import argparse
import cv2
import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

# 1. Setup system-level mocks to bypass Cython/OpenGL binary incompatibilities
sys.modules['gl_utils'] = MagicMock()
sys.modules['pyglui'] = MagicMock()
sys.modules['pyglui.cygl.utils'] = MagicMock()
sys.modules['glfw'] = MagicMock()

# Mock deepvog and edgaze so they aren't imported or run
sys.modules['pupil_detector_plugins.deepvog'] = MagicMock()
sys.modules['pupil_detector_plugins.edgaze'] = MagicMock()

# 2. Add pupil_src/shared_modules to sys.path
pupil_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
shared_modules_dir = os.path.join(pupil_src_dir, 'shared_modules')
if shared_modules_dir not in sys.path:
    sys.path.append(shared_modules_dir)

# Now safely import Detector2DPlugin
from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin

# # 3. Check and prepare checkpoint file matching plugin's expected ckpt name
# plugin_dir = os.path.join(shared_modules_dir, 'pupil_detector_plugins')
# expected_ckpt = os.path.join(plugin_dir, 'ckpt_adgbc.pth')
# source_ckpt = os.path.join(plugin_dir, 'adgbc_nn_best.pth')

# if not os.path.exists(expected_ckpt):
#     if os.path.exists(source_ckpt):
#         print(f"Copying {source_ckpt} to expected plugin checkpoint location: {expected_ckpt}")
#         shutil.copy(source_ckpt, expected_ckpt)
#     else:
#         print(f"Warning: Neither {expected_ckpt} nor {source_ckpt} was found. Instantiating model with random weights.")

# 4. Mock classes matching Pupil Labs plugin interfaces
class MockRoi:
    def __init__(self):
        self.bounds = (0, 0, 640, 480)

class MockGPool:
    def __init__(self):
        self.eye_id = 0
        self.debug = False
        self.display_mode = "algorithm"
        self.roi = MockRoi()

class MockFrame:
    def __init__(self, bgr_img, timestamp=0.0):
        self.bgr = bgr_img
        self.gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        self.height, self.width = bgr_img.shape[:2]
        self.timestamp = timestamp
        # self.jpeg_buffer = [0]

def parse_args():
    parser = argparse.ArgumentParser(description="Test Detector2DPlugin pupil extraction offline using plugin functions")
    parser.add_argument('--img_path', default="pupil.jpg", required=True, help="Path to input eye image")
    parser.add_argument('--save_path', default="_", help="Path to save visual results")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load image
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Input image not found: {args.img_path}")
    img_bgr = cv2.imread(args.img_path)
    img_bgr = cv2.resize(img_bgr, (192, 192), interpolation=cv2.INTER_LINEAR)

    # Instantiate Mock global pool and Frame
    g_pool = MockGPool()
    frame = MockFrame(img_bgr)

    # 5. Instantiate the Plugin
    print("Instantiating Detector2DPlugin...")
    plugin = Detector2DPlugin(g_pool=g_pool)
    print(f"Selected pupil detection model: {plugin.active_model}")

    # 6. Perform detection using the plugin's unified detect function
    print("Running detection using plugin.detect...")
    datum = plugin.detect(frame)

    # Print results
    print("\n--- Detection Result Datum ---")
    for k, v in datum.items():
        print(f"  {k}: {v}")
    print("------------------------------\n")

    # 7. Visualization
    ellipse_info = datum.get("ellipse")
    ellipse_found = ellipse_info and ellipse_info.get("axes") != (0.0, 0.0)

    # Draw result overlay
    img_result = img_bgr.copy()
    if ellipse_found:
        cx, cy = ellipse_info["center"]
        axes = ellipse_info["axes"]
        angle = ellipse_info["angle"]
        
        # Draw the green fitted ellipse
        cv2.ellipse(img_result, ((cx, cy), axes, angle), (0, 255, 0), 2)
        # Draw the blue center point
        cv2.circle(img_result, (int(cx), int(cy)), 3, (255, 0, 0), -1)


    # Discard first & last inferecne time (Loading model / Garbage collection)
    plugin.cleanup()
    for i in range(100):
        plugin.detect(frame)
        plugin.cleanup()


    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Frame")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Plugin Extraction Result")
    axes[1].axis('off')

    plt.tight_layout()
    if args.save_path.endswith((".png",".jpg")):
        plt.savefig(args.save_path, bbox_inches='tight', dpi=150)
        print(f"Visual results successfully commented out to: {args.save_path}")
    else: print("Result not saved. Set --save_path to save result image.")
    plt.close()

if __name__ == '__main__':
    main()

'''
np.mean(adgbc)
np.float64(1091.0975400358438) # ms 단위
np.var(adgbc)
np.float64(37212.80494607989)

np.mean(rit)
np.float64(980.738199991174) # ms 단위
np.var(rit)
np.float64(266492.158255929)
'''