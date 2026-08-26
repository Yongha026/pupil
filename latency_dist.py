import numpy as np
import csv
import argparse

from skimage.util import dtype

parser = argparse.ArgumentParser()
parser.add_argument("CSVPATH", type=str, help="Path to csv file")
args = parser.parse_args()

adgbc = np.array([])
ritnet = np.array([])

with open(args.CSVPATH, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["method"] == "AD-GBC":
            adgbc = np.append(adgbc, float(row["processing_latency_ms"]))
        elif row["method"] == "RITnet":
            ritnet = np.append(ritnet, float(row["processing_latency_ms"]))


print(f"AD-GBC result with total {len(adgbc)} data: {np.mean(adgbc):.4f} ± {np.std(adgbc):.4f}")
print(f"RITnet result with total {len(ritnet)} data: {np.mean(ritnet):.4f} ± {np.std(ritnet):.4f}")
