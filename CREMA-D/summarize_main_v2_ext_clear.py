#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path
from scipy.stats import t as tdist

METRICS = ["acc", "uar", "macro_f1", "spk_acc", "spk_uar"]

def paired_ttest(a, b):
    """two-sided paired t-test, d=a-b"""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        t = np.inf if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
    else:
        t = mean / (sd / sqrt(n))
        p = 2 * (1 - tdist.cdf(abs(t), df=n - 1))
    return mean, sd, float(t), float(p)

def fmt_pm(mean, std, nd=2):
    return f"{mean:.{nd}f}$\\pm${std:.{nd}f}"

def load_csvs(root: Path, seeds):
    rows = []
    for s in seeds:
        p = root / f"seed{s}" / "ablation_chain_summary_with_spk_uar.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        df = pd.read_csv(p)
        df["seed"] = int(s)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., /root/autodl-tmp/cremad_main_v2_ext_clear")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13,42,2026])
    ap.add_argument("--digits", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root)
    seeds = args.seeds
    nd = args.digits

    df = load_csvs(root, seeds)

    # ---- sanity: expected exp names ----
    expected = [
        "A_base",
        "C_gate",
        "E_gate_SAT_noCB",
        "C_gate_noTRF",
        "C_concat",
        "E_concat_SAT_noCB",
        "C_concat_noTRF",
    ]
    missing = [e for e in expected if e not in set(df["exp"])]
    if missing:
        print("⚠️ Missing exp(s) in CSVs:", missing)

    # ---- mean±std ----
    summ = df.groupby("exp")[METRICS].agg(["mean","std"])

    print("\n=== Mean±Std over 3 seeds (v2 main ext clear) ===")
    for exp in expected:
        if exp not in summ.index:
            continue
        m = summ.loc[exp]
        print(f"\n[{exp}]")
        for k in METRICS:
            mean = float(m[(k,"mean")])
            std  = float(m[(k,"std")]) if not np.isnan(m[(k,"std")]) else 0.0
            print(f"  {k:8s}: {mean:.4f} ± {std:.4f}")

    # ---- LaTeX rows ----
    print("\n=== LaTeX rows (copy into tables) ===")
    for exp in expected:
        if exp not in summ.index:
            continue
        m = summ.loc[exp]
        vals = [fmt_pm(float(m[(k,"mean")]), float(m[(k,"std")]) if not np.isnan(m[(k,"std")]) else 0.0, nd)
                for k in METRICS]
        # exp & Acc & UAR & MF1 & SpkAcc & SpkUAR
        print(f"{exp} & " + " & ".join(vals) + r" \\")

    # ---- paired t-tests ----
    def get(exp, metric):
        return df[df["exp"] == exp].sort_values("seed")[metric].to_list()

    comparisons = [
        ("C_gate", "C_gate_noTRF"),          # extraTRF effect (gate)
        ("C_concat", "C_concat_noTRF"),      # extraTRF effect (concat)
        ("C_gate", "C_concat"),              # fusion effect under C
        ("C_gate", "E_gate_SAT_noCB"),       # SAT effect (gate)
        ("C_concat", "E_concat_SAT_noCB"),   # SAT effect (concat)
    ]

    print("\n=== Paired t-tests (two-sided, n=3) | Δ = first - second ===")
    for a, b in comparisons:
        print(f"\n[{a}] vs [{b}]")
        for met in METRICS:
            mean, sd, t, p = paired_ttest(get(a, met), get(b, met))
            print(f"  {met:8s}: Δ={mean:+.4f} ± {sd:.4f} | t={t:.3f} | p={p:.6f}")

if __name__ == "__main__":
    main()