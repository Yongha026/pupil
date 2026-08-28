cd ./pupil_src/shared_modules/pupil_detector_plugins/model_ckpts

ADGBC="adgbc_nn_best.pth"
RITNET="ritnet_nn_best.pth"
UNEXT="unext_nn_best.pth"
MLU="mambaliteunet_nn_best.pth"
RUL="rollingunet_nn_best.pth"
ULVM="ulvm_nn_best.pth"

# 1. AD-GBC
if [ -f "$ADGBC" ]; then
  echo "Already have ckpt for AD-GBC"
else
  gdown 1By1SsLnPVxvQ6CiJ5sBThfqcW-0o5OpN
  echo "Downloaded ckpt for AD-GBC"
fi

# 2. RITnet
if [ -f "$RITNET" ]; then
  echo "Already have ckpt for nn_RITnet"
else
  gdown 1AvLUJj7e4Rfj61BNLzvDGl0pciJAfubq
  echo "Downloaded ckpt for nn_RITnet"
fi

# 3. UNeXt
if [ -f "$UNEXT" ]; then
  echo "Already have ckpt for UNeXt"
else
  gdown 1wRdTBIjoCzbOPh0EW_-bwBQEECGBEMRW
  echo "Downloaded ckpt for UNeXt"
fi

# 4. MambaLiteUNet
if [ -f "$MLU" ]; then
  echo "Already have ckpt for MambaLiteUNet"
else
  gdown 1qunzvgNxDB06vWZl_3hRcydVb29W37jU
  echo "Downloaded ckpt for MambaLiteUNet"
fi

# 5. RollingUNet
if [ -f "$RUL" ]; then
  echo "Already have ckpt for RollingUNet"
else
  gdown 1ToMEQg9SFRAPqP3XfxeORVKflrOEnRGV
  echo "Downloaded ckpt for RollingUNet"
fi
# 6. UltraLight_VMUNet
if [ -f "$ULVM" ]; then
  echo "Already have ckpt for UltraLight_VMUNet"
else
  gdown 1djqaLKjhOvqfDVHsYd_5m-9bsX3NFV46
  echo "Downloaded ckpt for UltraLight_VMUNet"
fi

cd ../../../../
