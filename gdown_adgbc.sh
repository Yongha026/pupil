cd ./pupil_src/shared_modules/pupil_detector_plugins

ADGBC="adgbc_nn_best.pth"
RITNET="ritnet_nn_best.pth"
UNEXT="unext_nn_best.pth"

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

cd ../../../
