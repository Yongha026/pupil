cd ./pupil_src/shared_modules/pupil_detector_plugins/model_ckpts

ADGBC="adgbc_nn_best.pth"
RITNET="ritnet_nn_best.pth"
UNEXT="unext_nn_best.pth"
MLU="mambaliteunet_nn_best.pth"
RUL="rollingunet_nn_best.pth"
ULVM="ulvm_nn_best.pth"
UKAN="ukan_nn_best.pth"
PMR="pmr_nn_best.pth"

# 1. AD-GBC
if [ -f "$ADGBC" ]; then
  echo "Already have ckpt for AD-GBC"
else
  gdown 1V_bRnKT8cErFppsdRrX4R0hXdf_RnwLB
  echo "Downloaded ckpt for AD-GBC"
fi

# 2. RITnet
if [ -f "$RITNET" ]; then
  echo "Already have ckpt for nn_RITnet"
else
  gdown 1mZEDirn74gLKd3Cn_i9qTEzzMREKTodY
  echo "Downloaded ckpt for nn_RITnet"
fi

# 3. UNeXt
if [ -f "$UNEXT" ]; then
  echo "Already have ckpt for UNeXt"
else
  gdown 1lrnr6HJrBS4QDqW0g2Gc7VTa48uFbd21
  echo "Downloaded ckpt for UNeXt"
fi

# 4. MambaLiteUNet
if [ -f "$MLU" ]; then
  echo "Already have ckpt for MambaLiteUNet"
else
  gdown 1YcBL6xqqri2FSAap7VMUk00UOVrXHqN5
  echo "Downloaded ckpt for MambaLiteUNet"
fi

# 5. RollingUNet
if [ -f "$RUL" ]; then
  echo "Already have ckpt for RollingUNet"
else
  gdown 1ZIfrefk9hg06YsPt9NwEk7j2Dy0S3vTy
  echo "Downloaded ckpt for RollingUNet"
fi

# 6. UltraLight_VMUNet
if [ -f "$ULVM" ]; then
  echo "Already have ckpt for UltraLight_VMUNet"
else
  gdown 1tRCnzwXm4g8O9POy_gtxGDn4G4Q0MPeP
  echo "Downloaded ckpt for UltraLight_VMUNet"
fi

# 7. U-KAN
if [ -f "$UKAN" ]; then
  echo "Already have ckpt for UltraLight_VMUNet"
else
  gdown 1eNi3Z3xeqsEgxA1x5oaJ-3ZiiPdLDu6F
  echo "Downloaded ckpt for UltraLight_VMUNet"
fi


# 8. PMRNet
if [ -f "$PMR" ]; then
  echo "Already have ckpt for PMRNet"
else
  gdown 1QCoBeasJJ00C7yB2N14Zb0NkfJ32gQP5
  echo "Downloaded ckpt for PMRNet"
fi

cd ../../../../