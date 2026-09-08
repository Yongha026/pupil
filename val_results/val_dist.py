import argparse
import csv
import glob
import os
from typing import Dict, List, Optional
import numpy as np
import re


def find_latest_validation_csv(base_dir: Optional[str] = None) -> Optional[str]:
    """Find the most recently modified validation CSV file in Val_results or current directory."""
    if base_dir is None or not os.path.isdir(base_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates: List[str] = []
    # Search in current script directory, parent/Val_results, and workspace root
    search_dirs = [
        base_dir,
        os.path.join(base_dir, "Val_results"),
        os.path.join(base_dir, "..", "Val_results"),
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            candidates.extend(glob.glob(os.path.join(d, "validation_results_*.csv")))

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Print model, mean accuracy, and mean precision from validation CSV.")
    parser.add_argument("CSVPATH", nargs="?", default=None, help="Path to validation CSV file (optional).")
    args = parser.parse_args()

    csvfile = args.CSVPATH
    if not csvfile or not os.path.exists(csvfile):
        csvfile = find_latest_validation_csv()

    if not csvfile or not os.path.exists(csvfile):
        print("Error: No validation CSV file found.")
        return

    # Group measurements by model: {model_name: {'acc': [...], 'prec': [...]}}
    model_stats: Dict[str, Dict[str, List[float]]] = {}

    with open(csvfile, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:  # DictReader starts directly on data rows (no need for i >= 1)
            if re.search("#", str(row)):
                note = str(row['date']).replace('# "','').replace('"','')
                continue
            acc_str = row.get("accuracy_deg", "").strip()
            prec_str = row.get("precision_deg", "").strip()
            model = row.get("model", "unknown").strip()

            if not acc_str or not prec_str:
                continue

            try:
                acc_val = float(acc_str)
                prec_val = float(prec_str)
            except ValueError:
                continue

            if model not in model_stats:
                model_stats[model] = {"acc": [], "prec": []}

            model_stats[model]["acc"].append(acc_val)
            model_stats[model]["prec"].append(prec_val)

    if not model_stats:
        print("No valid accuracy/precision records found in the CSV.")
        return

    # Print results per model
    for model, stats in model_stats.items():
        runs = len(stats["acc"])
        mean_acc = np.mean(stats["acc"])
        std_acc = np.std(stats["acc"])
        mean_prec = np.mean(stats["prec"])
        std_prec = np.std(stats["prec"])
        print(f"model: {model}, total {runs} runs, accuracy={mean_acc:.4f} ± {std_acc:.4f}, precision={mean_prec:.4f} ± {std_prec:.4f}")
    try:
        print(note)
    except:
        pass

if __name__ == "__main__":
    main()