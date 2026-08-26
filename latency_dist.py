import numpy as np
import csv
import argparse

from skimage.util import dtype

parser = argparse.ArgumentParser()
parser.add_argument("CSVPATH", type=str, help="Path to csv file")
args = parser.parse_args()

f = open(args.CSVPATH, 'r')
reader = csv.reader(f)

adgbc = np.array([])
ritnet = np.array([])

for row in reader:
    if row[0] == "AD-GBC":
        adgbc = np.append(adgbc, float(row[2]))
    elif row[0] == "RITnet":
        ritnet = np.append(ritnet, float(row[2]))


print(f"AD-GBC result with total {len(adgbc)} data: {np.mean(adgbc):.4f} ± {np.std(adgbc):.4f}")
print(f"RITnet result with total {len(ritnet)} data: {np.mean(ritnet):.4f} ± {np.std(ritnet):.4f}")
