import numpy as np
import csv
import argparse

from skimage.util import dtype

parser = argparse.ArgumentParser()
parser.add_argument("CSVPATH", type=str, help="Path to csv file")
args = parser.parse_args()

adgbc_process = np.array([])
adgbc_start_timestamp = np.array([])
ritnet_process = np.array([])
ritnet_start_timestamp = np.array([])

with open(args.CSVPATH, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["method"] == "AD-GBC":
            adgbc_process = np.append(adgbc_process, float(row["processing_latency_ms"]))
            adgbc_start_timestamp = np.append(adgbc_start_timestamp, float(row["t_start"]))
        elif row["method"] == "RITnet":
            ritnet_process = np.append(ritnet_process, float(row["processing_latency_ms"]))
            ritnet_start_timestamp = np.append(ritnet_start_timestamp, float(row["t_start"]))

print(f"AD-GBC result with total {len(adgbc_process)} data: {np.mean(adgbc_process):.4f} ± {np.std(adgbc_process):.4f}")
print(f"AD-GBC result average start timestamp: {np.mean(adgbc_start_timestamp)}\n")
print(f"RITnet result with total {len(ritnet_process)} data: {np.mean(ritnet_process):.4f} ± {np.std(ritnet_process):.4f}")
print(f"RITnet result average start timestamp: {np.mean(ritnet_start_timestamp)}")
