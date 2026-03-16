#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_std(values: np.ndarray):
    if values.size == 0:
        return "", ""
    return float(values.mean()), float(values.std())


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate xp_* csv results")
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    results_root = args.results_root
    output_csv = args.output_csv or (results_root / "xp_summary.csv")

    rows_out: List[Dict[str, object]] = []

    for layout_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        for combo_dir in sorted([p for p in layout_dir.iterdir() if p.is_dir() and p.name.startswith("xp_")]):
            csv_path = combo_dir / "csv" / "xp_pairs.csv"
            if not csv_path.exists():
                continue

            rows = _read_csv_rows(csv_path)
            if not rows:
                continue

            rewards = np.asarray([float(r["reward_mean"]) for r in rows], dtype=np.float32)
            algo0 = rows[0].get("algo0", "")
            algo1 = rows[0].get("algo1", "")
            same_algo = (algo0 != "") and (algo0 == algo1)

            sp_mean, sp_std = "", ""
            xp_mean, xp_std = "", ""
            if same_algo:
                sp_rewards = np.asarray(
                    [float(r["reward_mean"]) for r in rows if r.get("run0", "") == r.get("run1", "")],
                    dtype=np.float32,
                )
                xp_rewards = np.asarray(
                    [float(r["reward_mean"]) for r in rows if r.get("run0", "") != r.get("run1", "")],
                    dtype=np.float32,
                )
                sp_mean, sp_std = _mean_std(sp_rewards)
                xp_mean, xp_std = _mean_std(xp_rewards)
            else:
                xp_mean, xp_std = _mean_std(rewards)

            rows_out.append(
                {
                    "layout": layout_dir.name,
                    "xp_folder": combo_dir.name,
                    "algo0": algo0,
                    "algo1": algo1,
                    "num_rows": int(len(rows)),
                    "sp": sp_mean,
                    "sp_std": sp_std,
                    "xp": xp_mean,
                    "xp_std": xp_std,
                    "reward_mean": float(rewards.mean()),
                    "reward_var": float(rewards.var()),
                    "reward_std": float(rewards.std()),
                    "source_csv": str(csv_path),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "xp_folder",
        "layout",
        "algo0",
        "algo1",
        "num_rows",
        "sp",
        "sp_std",
        "xp",
        "xp_std",
        "reward_mean",
        "reward_var",
        "reward_std",
        "source_csv",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} summary rows to {output_csv}")


if __name__ == "__main__":
    main()
