"""inject_val_v3.py — 从 injected_events_val.csv 重建 V3 val labels（seed=123）"""
from __future__ import annotations
import os, sys, json, csv

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_week5_root = os.path.dirname(_script_dir)
sys.path.insert(0, _week5_root)

from config import DATA_DIR, VAL_HOURS, N_CELLS


def build_labels_from_csv(csv_path: str, T: int, N: int) -> np.ndarray:
    """从 V2 injected_events CSV 重建 labels。"""
    labels = np.zeros((T, N), dtype=bool)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_start = int(row["t_start"])
            t_end = int(row["t_end"])
            # event_cells_injected 可能是短列表字符串 [...]
            cells_raw = row["event_cells_injected"].strip("[]")
            # parse ints - might be comma or comma+space separated
            if cells_raw:
                try:
                    cells = [int(x.strip()) for x in cells_raw.split(",") if x.strip()]
                except ValueError:
                    # if it's too long/truncated, skip
                    cells = []
            else:
                cells = []
            for t in range(t_start, t_end + 1):
                if 0 <= t < T:
                    for c in cells:
                        if 0 <= c < N:
                            labels[t, c] = True
    return labels


def main():
    print("[inject_val_v3] start")

    csv_path = os.path.join(DATA_DIR, "injected_events_val.csv")
    print(f"  reading {csv_path}")

    # VAL_HOURS=504, N_CELLS=1024
    labels = build_labels_from_csv(csv_path, VAL_HOURS, N_CELLS)

    print(f"  labels: {labels.shape}, sum={labels.sum()}, frac={labels.mean():.4f}")

    out_path = os.path.join(DATA_DIR, "anomaly_labels_val_v3.npy")
    np.save(out_path, labels.astype(np.float32))
    print(f"  saved {out_path}")

    # count by type
    type_counts = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

    summary = {
        "seed": 123,
        "method": "rebuilt_from_injected_events_val.csv",
        "total_labels": int(labels.sum()),
        "anomaly_ratio": round(float(labels.mean()), 5),
        "n_events": sum(type_counts.values()),
        "type_counts": type_counts,
    }
    with open(os.path.join(DATA_DIR, "injection_summary_v3.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("[inject_val_v3] done")


if __name__ == "__main__":
    main()
