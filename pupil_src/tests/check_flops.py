import torch
import torchvision.models as models
from ptflops import get_model_complexity_info
import os
import argparse
import sys
from unittest.mock import MagicMock

# Setup system-level mocks to bypass Cython/OpenGL binary incompatibilities
sys.modules['gl_utils'] = MagicMock()
sys.modules['pyglui'] = MagicMock()
sys.modules['pyglui.cygl.utils'] = MagicMock()
sys.modules['glfw'] = MagicMock()

# Add the specific pupil_detector_plugins directory to sys.path
tests_dir = os.path.dirname(os.path.abspath(__file__))
plugins_dir = os.path.abspath(os.path.join(tests_dir, "..", "shared_modules", "pupil_detector_plugins"))
if plugins_dir not in sys.path:
    sys.path.append(plugins_dir)

parser=  argparse.ArgumentParser()
parser.add_argument("DETECT_MODEL", type=str, help="adgbc | nn_ritnet | nn_unext | mambaliteunet | rollingunet | ulvmunet")
args = parser.parse_args()

plugin_dir = "../shared_modules/pupil_detector_plugins"
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)



model_path_adgbc = os.path.join(plugin_dir, "adgbc_nn_best.pth")
model_path_ritnet_orig = os.path.join(plugin_dir, "best_model.pkl")
model_path_nn_ritnet = os.path.join(plugin_dir, "ritnet_nn_best.pth")
model_path_nn_unext = os.path.join(plugin_dir, "unext_nn_best.pth")
model_path_mambaliteunet = os.path.join(plugin_dir, "mambaliteunet_nn_best.pth")
model_path_rollingunet = os.path.join(plugin_dir, "rollingunet_nn_best.pth")
model_path_ulvmunet = os.path.join(plugin_dir, "ulvm_nn_best.pth")

# Load only the specified DETECT_MODEL to save memory and startup time
if args.DETECT_MODEL == "adgbc":
    # 1) AD-GBC Model
    import adgbc
    try:
        model = adgbc.GBC_Rolling_Unet_L(num_classes=4, input_channels=1, deep_supervision=False).to(device)
        if os.path.exists(model_path_adgbc):
            checkpoint = torch.load(model_path_adgbc, map_location=device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        else:
            print(f"ADGBC ckpt file not found at {model_path_adgbc}")
    except Exception as e:
        print(f"Error loading adgbc: {e}")
        raise e

elif args.DETECT_MODEL == "nn_ritnet":
    # 3) nnRITnet Model
    import nn_ritnet
    try:
        model = nn_ritnet.DenseNet2D(in_channels=1, out_channels=4, dropout=True, prob=0.2, deep_supervision=False).to(device)
        if os.path.exists(model_path_nn_ritnet):
            checkpoint = torch.load(model_path_nn_ritnet, map_location=device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        else:
            print(f"nnRITnet ckpt file not found at {model_path_nn_ritnet}")
    except Exception as e:
        print(f"Error loading nn_ritnet: {e}")
        raise e

elif args.DETECT_MODEL == "nn_unext":
    # 4) UNeXt Model
    import nn_unext
    try:
        model = nn_unext.UNext(num_classes=4, input_channels=1, deep_supervision=False).to(device)
        if os.path.exists(model_path_nn_unext):
            checkpoint = torch.load(model_path_nn_unext, map_location=device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        else:
            print(f"UNeXt ckpt file not found at {model_path_nn_unext}")
    except Exception as e:
        print(f"Error loading nn_unext: {e}")
        raise e

elif args.DETECT_MODEL == "mambaliteunet":
    # 5) MambaLiteUNet Model
    import mambaliteunet
    try:
        model = mambaliteunet.MambaLiteUNet(num_classes=4, input_channels=1).to(device)
        if os.path.exists(model_path_mambaliteunet):
            checkpoint = torch.load(model_path_mambaliteunet, map_location=device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        else:
            print(f"MambaLiteUNet ckpt file not found at {model_path_mambaliteunet}")
    except Exception as e:
        print(f"Error loading mambaliteunet: {e}")
        raise e

elif args.DETECT_MODEL == "rollingunet":
    # 6) Rolling_Unet_L Model
    class RollingUnetDummy:
        pass
    rollingunet = RollingUnetDummy()
    from adgbc.archs_GBC import Rolling_Unet_L
    rollingunet.Rolling_Unet_L = Rolling_Unet_L
    try:
        model = rollingunet.Rolling_Unet_L(num_classes=4, input_channels=1, deep_supervision=False).to(device)
        if os.path.exists(model_path_rollingunet):
            checkpoint = torch.load(model_path_rollingunet, map_location=device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (
                        isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        else:
            print(f"Rolling_Unet_L ckpt file not found at {model_path_rollingunet}")
    except Exception as e:
        print(f"Error loading rollingunet: {e}")
        raise e

elif args.DETECT_MODEL == "ulvmunet":
    # 7) UltraLight_VM_UNet Model
    import ulvmunet
    try:
        model = ulvmunet.UltraLight_VM_UNet(num_classes=4, input_channels=1).to(device)
        if os.path.exists(model_path_ulvmunet):
            checkpoint = torch.load(model_path_ulvmunet, map_location=device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        else:
            print(f"ULVMUNet ckpt file not found at {model_path_ulvmunet}")
    except Exception as e:
        print(f"Error loading ulvmunet: {e}")
        raise e

macs, params = get_model_complexity_info(
    model, (1, 224, 224), as_strings=True, print_per_layer_stat=True
)
print(f"{macs}, {params}")

