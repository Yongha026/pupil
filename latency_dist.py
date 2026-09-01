import numpy as np
import csv
import argparse

from skimage.util import dtype

parser = argparse.ArgumentParser()
parser.add_argument("CSVPATH", type=str, help="Path to csv file")
args = parser.parse_args()

# adgbc_process = np.array([])
# adgbc_start_timestamp = np.array([])
# ritnet_process = np.array([])
# ritnet_start_timestamp = np.array([])

# with open(args.CSVPATH, 'r', newline='') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         if row["method"] == "AD-GBC":
#             adgbc_process = np.append(adgbc_process, float(row["processing_latency_ms"]))
#             adgbc_start_timestamp = np.append(adgbc_start_timestamp, float(row["t_start"]))
#         elif row["method"] == "RITnet":
#             ritnet_process = np.append(ritnet_process, float(row["processing_latency_ms"]))
#             ritnet_start_timestamp = np.append(ritnet_start_timestamp, float(row["t_start"]))
#
# print(f"AD-GBC result with total {len(adgbc_process)} data: {np.mean(adgbc_process):.4f} ± {np.std(adgbc_process):.4f}")
# print(f"AD-GBC result average start timestamp: {np.mean(adgbc_start_timestamp)}\n")
# print(f"RITnet result with total {len(ritnet_process)} data: {np.mean(ritnet_process):.4f} ± {np.std(ritnet_process):.4f}")
# print(f"RITnet result average start timestamp: {np.mean(ritnet_start_timestamp)}")
latencies = np.array([])
start_timestamps = np.array([])
with open(args.CSVPATH, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 1:
            latencies = np.append(latencies, float(row['processing_latency_ms']))
            start_timestamps = np.append(start_timestamps, float(row['t_start']))
latencies = latencies[:-1]
start_timestamps = start_timestamps[:-1]
# starts = starts[:-1]
print(f"With total {len(latencies)} data: {np.mean(latencies):.4f} ± {np.std(latencies):.4f}")
print(f"Average start timestamp: {np.mean(start_timestamps)}")


#################### 100 loops in iulab9 #################
# adgbc
# With total 99 data: 102.7918 ± 22.2038
# Average start timestamp: 2167746.3643981838
# ritnet
# With total 99 data: 5.5401 ± 0.1775
# Average start timestamp: 2167949.2887510182
# nn_ritnet
# With total 99 data: 5.4053 ± 0.2919
# Average start timestamp: 2167908.442596336
# nn_unext
# With total 99 data: 6.4965 ± 0.4118
# Average start timestamp: 2167986.227508659
# 2dcpp
# With total 99 data: 1.4109 ± 0.0637
# Average start timestamp: 2168494.829089349
# mambaliteunet
# With total 99 data: 50.3218 ± 1.1478
# Average start timestamp: 2231818.8394683036
# rollingunet
# With total 99 data: 97.6275 ± 20.2625
# Average start timestamp: 2234496.729014041
# UltraLight_VMUNet
# With total 99 data: 17.9419 ± 0.5686
# Average start timestamp: 2255016.0028650993
# UKAN
# With total 99 data: 18.8477 ± 0.3765
# Average start timestamp: 2517877.3691836223
# PMRNet
# With total 99 data: 17.9731 ± 1.8053
# Average start timestamp: 2587212.066454949