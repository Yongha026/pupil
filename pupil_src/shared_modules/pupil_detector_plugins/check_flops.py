import torch
import torchvision.models as models
from ptflops import get_model_complexity_info
import os
import argparse

import adgbc, mambaliteunet, rollingunet, ulvmunet, nn_ritnet, nn_unext
parser=  argparse.ArgumentParser()
parser.add_argument("DETECT_MODEL", type=str, help="adgbc | nn_ritnet | nn_unext | mambaliteunet | rollingunet | ulvmunet")
args = parser.parse_args()

plugin_dir = "./model_ckpts"
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
    try:
        model = adgbc.GBC_Rolling_Unet_L(num_classes=4, input_channels=1, deep_supervision=False).to(device)
        if os.path.exists(model_path_adgbc):
            checkpoint = torch.load(model_path_adgbc, map_location=device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
        else:
            pass
    except Exception as e:
        pass


elif args.DETECT_MODEL == "nn_ritnet":
    # 3) nnRITnet Model
    try:
        model = nn_ritnet.DenseNet2D(in_channels=1, out_channels=4, dropout=True, prob=0.2, deep_supervision=False).to(self.device)
        if os.path.exists(model_path_nn_ritnet):
            checkpoint = torch.load(model_path_nn_ritnet, map_location=self.device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

        else:
            pass
    except Exception as e:
        pass

elif args.DETECT_MODEL == "nn_unext":
    # 4) UNeXt Model
    try:
        model = nn_unext.UNext(num_classes=4, input_channels=1, deep_supervision=False).to(self.device)
        if os.path.exists(model_path_nn_unext):
            checkpoint = torch.load(model_path_nn_unext, map_location=self.device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

        else:
            pass
    except Exception as e:
        pass
elif args.DETECT_MODEL == "mambaliteunet":
    # 5) MambaLiteUNet Model
    try:
        model = mambaliteunet.MambaLiteUNet(num_classes=4, input_channels=1).to(self.device)
        if os.path.exists(model_path_nn_unext):
            checkpoint = torch.load(model_path_mambaliteunet, map_location=self.device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

        else:
            pass
    except Exception as e:
        pass

elif args.DETECT_MODEL == "rollingunet":
    # 6) Rolling_Unet_L Model
    try:
        model = rollingunet.Rolling_Unet_L(num_classes=4, input_channels=1, deep_supervision=False).to(self.device)
        if os.path.exists(model_path_nn_unext):
            checkpoint = torch.load(model_path_rollingunet, map_location=self.device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (
                        isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

        else:
            pass
    except Exception as e:
        pass

elif args.DETECT_MODEL == "ulvmunet":
    # 7) UltraLight_VM_UNet Model
    try:
        model = ulvmunet.UltraLight_VM_UNet(num_classes=4, input_channels=1).to(self.device)
        if os.path.exists(model_path_nn_unext):
            checkpoint = torch.load(model_path_ulvmunet, map_location=self.device, weights_only=False)
            state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
            model.load_state_dict(state_dict)
            model.eval()

        else:
            print(f"ULVMUNet ckpt file not found at {model_path_ulvmunet}")
    except Exception as e:
        pass



macs, params = get_model_complexity_info(
    model, (3, 224, 224), as_strings=True, print_per_layer_stat=True
)
print(f"{macs}, {params}")

