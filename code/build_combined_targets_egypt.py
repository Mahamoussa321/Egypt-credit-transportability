#!/usr/bin/env python3
"""
build_combined_targets_egypt.py

Create the final modeling file used by the Egypt credit transportability project.

This script combines:
  1) egypt_wbes_pooled_clean_improved.csv
  2) egypt_alt_targets_pooled_clean.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_COMMON_COLS = [
    "survey_year",
    "firm_age",
    "log_size",
    "log_sales",
    "survey_year_cat",
    "b4",
    "e11",
    "l1",
    "b5",
    "b2b",
    "d2",
    "j2",
    "k82a",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Build combined Egypt target file.")
    parser.add_argument("--base-dir", type=str, required=True, help="Directory containing the processed CSV files.")
    return parser.parse_args()

def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")

def main():
    args = parse_args()
    base = Path(args.base_dir)

    source_a = base / "egypt_wbes_pooled_clean_improved.csv"
    source_b = base / "egypt_alt_targets_pooled_clean.csv"
    output = base / "egypt_combined_targets_pooled_clean.csv"

    print(f"[read] source_a = {source_a}")
    print(f"[read] source_b = {source_b}")
    print(f"[write] output  = {output}")

    if not source_a.exists():
        raise FileNotFoundError(f"Could not find source_a: {source_a}")
    if not source_b.exists():
        raise FileNotFoundError(f"Could not find source_b: {source_b}")

    a = pd.read_csv(source_a)
    b = pd.read_csv(source_b)

    print(f"[shape] a = {a.shape}")
    print(f"[shape] b = {b.shape}")

    require_columns(a, DEFAULT_COMMON_COLS + ["target"], "source_a")
    require_columns(b, DEFAULT_COMMON_COLS + ["target_formal_credit", "target_applied_credit"], "source_b")

    aligned = a[DEFAULT_COMMON_COLS].reset_index(drop=True).equals(
        b[DEFAULT_COMMON_COLS].reset_index(drop=True)
    )
    print(f"[check] aligned = {aligned}")

    if not aligned:
        raise ValueError("The two source files are not row-aligned on the common columns.")

    out = b.copy()
    out["target_k30"] = a["target"]

    preferred_order = ["target_k30", "target_formal_credit", "target_applied_credit"]
    existing = [c for c in out.columns if c not in preferred_order]
    out = out[existing + preferred_order]

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(f"[done] wrote: {output}")
    print(f"[done] output shape: {out.shape}")
    print(f"[done] target columns present: {[c for c in preferred_order if c in out.columns]}")

if __name__ == "__main__":
    main()
