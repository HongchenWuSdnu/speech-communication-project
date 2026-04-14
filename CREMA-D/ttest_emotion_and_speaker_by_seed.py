#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ttest_emotion_and_speaker_by_seed.py

Compute:
1) mean ± std over seeds for each experiment and each metric
2) paired t-test by seed for selected comparisons
   - Emotion: Acc / UAR / MF1
   - Speaker: SpkAcc / SpkUAR (if available)

Input: multiple ablation_chain_summary.csv files (one per seed).
Each CSV is expected to have at least:
  exp,best_epoch,acc,uar,macro_f1,spk_acc,...
Optionally:
  spk_uar

Typical usage:
  python ttest_emotion_and_speaker_by_seed.py \
    /root/autodl-tmp/ablation_chain_seed13/ablation_chain_summary.csv \
    /root/autodl-tmp/ablation_chain_seed42/ablation_chain_summary.csv \
    /root/autodl-tmp/ablation_chain_seed2026/ablation_chain_summary.csv \
    --exp-a A_base --exp-c C_plus_specaug --exp-sat C_plus_SAT_noCB

Or pass directories (script will locate the CSV inside):
  python ttest_emotion_and_speaker_by_seed.py \
    /root/autodl-tmp/ablation_chain_seed13 \
    /root/autodl-tmp/ablation_chain_seed42 \
    /root/autodl-tmp/ablation_chain_seed2026 \
    --exp-a A_base --exp-c C_plus_specaug --exp-sat C_plus_SAT_noCB
"""

import os
import csv
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np

# --- try scipy for accurate p-values and CI ---
try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------
# helpers
# ----------------------------
def _is_dir(p: str) -> bool:
    return os.path.isdir(p)

def _resolve_csv_path(path_or_dir: str) -> str:
    """
    Accept either a CSV file path or a directory containing ablation_chain_summary.csv
    """
    if os.path.isfile(path_or_dir) and path_or_dir.endswith(".csv"):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        cand = os.path.join(path_or_dir, "ablation_chain_summary.csv")
        if os.path.isfile(cand):
            return cand
        # allow any *summary*.csv
        for fn in os.listdir(path_or_dir):
            if fn.endswith(".csv") and "summary" in fn:
                return os.path.join(path_or_dir, fn)
    raise FileNotFoundError(f"Cannot find summary CSV from: {path_or_dir}")

def _infer_seed_from_path(p: str) -> str:
    """
    Best-effort seed inference from path string. If cannot, return basename.
    Examples: ...seed2026/... -> '2026'
    """
    s = p
    # look for 'seed' followed by digits
    import re
    m = re.search(r"seed(\d+)", s)
    if m:
        return m.group(1)
    # fallback: directory name or file stem
    base = os.path.basename(p)
    if base.endswith(".csv"):
        base = os.path.splitext(base)[0]
    return base

def read_summary_csv(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize keys
            rr = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            rows.append(rr)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    if "exp" not in rows[0]:
        raise ValueError(f"CSV missing 'exp' column: {csv_path}")
    return rows

def index_rows_by_exp(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    mp = {}
    for r in rows:
        mp[r["exp"]] = r
    return mp

def _get_float(row: Dict[str, str], key: str) -> Optional[float]:
    if key not in row or row[key] is None or row[key] == "":
        return None
    try:
        return float(row[key])
    except Exception:
        return None

def mean_std(x: List[float]) -> Tuple[float, float]:
    arr = np.array(x, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0

def paired_ttest(diffs: np.ndarray) -> Dict[str, float]:
    """
    diffs: array of (x2 - x1) by seed
    return: mean_delta, t, p, ci_low, ci_high
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.size
    if n < 2:
        return {"mean_delta": float(diffs.mean()), "t": float("nan"), "p": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": n}

    md = diffs.mean()
    sd = diffs.std(ddof=1)
    se = sd / math.sqrt(n) if sd > 0 else 0.0

    # t-stat: md / se
    t = md / se if se > 0 else (0.0 if md == 0 else float("inf"))

    if _HAS_SCIPY:
        df = n - 1
        p = float(stats.t.sf(abs(t), df) * 2.0)
        tcrit = float(stats.t.ppf(0.975, df))
        ci_low = md - tcrit * se
        ci_high = md + tcrit * se
    else:
        # fallback: no scipy -> provide t but no reliable p/CI
        p = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")

    return {"mean_delta": float(md), "t": float(t), "p": float(p),
            "ci_low": float(ci_low), "ci_high": float(ci_high), "n": n}

def format_ms(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def _print_block(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def _print_ttest_line(label: str, res: Dict[str, float], unit: str = ""):
    md = res["mean_delta"]
    t = res["t"]
    p = res["p"]
    lo = res["ci_low"]
    hi = res["ci_high"]
    n = int(res["n"])
    if _HAS_SCIPY:
        print(f"{label:<16s}: meanΔ={md:+.3f}{unit} | t={t:.3f} | p={p:.6f} | 95%CI=[{lo:+.3f},{hi:+.3f}]{unit} | n={n}")
    else:
        print(f"{label:<16s}: meanΔ={md:+.3f}{unit} | t={t:.3f} | p=NA (no scipy) | 95%CI=NA | n={n}")

def collect_metric_by_seed(
    per_seed_exp: Dict[str, Dict[str, Dict[str, str]]],
    exp_name: str,
    metric_key: str,
) -> Tuple[List[str], List[float]]:
    """
    per_seed_exp[seed][exp] -> row dict
    returns aligned seeds list and values list
    """
    seeds = []
    vals = []
    for seed, exp_map in per_seed_exp.items():
        if exp_name not in exp_map:
            continue
        v = _get_float(exp_map[exp_name], metric_key)
        if v is None:
            continue
        seeds.append(seed)
        vals.append(v)
    return seeds, vals

def aligned_pair_values(
    per_seed_exp: Dict[str, Dict[str, Dict[str, str]]],
    exp1: str,
    exp2: str,
    metric_key: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Return seeds that have both exp1 and exp2 + the metric, and their values
    """
    seeds = []
    v1 = []
    v2 = []
    for seed, exp_map in per_seed_exp.items():
        if exp1 not in exp_map or exp2 not in exp_map:
            continue
        a = _get_float(exp_map[exp1], metric_key)
        b = _get_float(exp_map[exp2], metric_key)
        if a is None or b is None:
            continue
        seeds.append(seed)
        v1.append(a)
        v2.append(b)
    return seeds, np.array(v1, dtype=np.float64), np.array(v2, dtype=np.float64)


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="+",
        help="List of CSV paths OR directories containing ablation_chain_summary.csv (one per seed).",
    )
    ap.add_argument("--exp-a", default="A_base", help="Name for baseline A in CSV.")
    ap.add_argument("--exp-c", default="C_plus_specaug", help="Name for SpecAug model C in CSV.")
    ap.add_argument(
        "--exp-sat",
        default="C_plus_SAT_noCB",
        help="Name for SAT(noCB) model in CSV. (If not found, script will try common aliases.)",
    )
    ap.add_argument("--digits", type=int, default=3, help="Digits for mean±std display.")
    args = ap.parse_args()

    csv_paths = []
    for p in args.paths:
        csv_paths.append(_resolve_csv_path(p))

    # load all seeds
    per_seed_exp: Dict[str, Dict[str, Dict[str, str]]] = {}
    for cp in csv_paths:
        seed = _infer_seed_from_path(cp)
        rows = read_summary_csv(cp)
        exp_map = index_rows_by_exp(rows)
        per_seed_exp[seed] = exp_map

    # resolve SAT exp name (support aliases)
    sat_name = args.exp_sat
    sat_aliases = [
        sat_name,
        "C+SAT_noCB",
        "C_plus_SAT_noCB",
        "C_plus_SATramp_noCB",
        "C_plus_SAT",  # in case you used this
        "Csat_noCB",
    ]
    # find first alias present in at least one seed
    present_alias = None
    for alias in sat_aliases:
        for seed, exp_map in per_seed_exp.items():
            if alias in exp_map:
                present_alias = alias
                break
        if present_alias is not None:
            break
    if present_alias is None:
        # allow user to run without SAT exp; we will skip C vs SAT tests
        sat_name_resolved = None
    else:
        sat_name_resolved = present_alias

    # which metrics to summarize
    metrics = [
        ("acc", "Acc(%)"),
        ("uar", "UAR(%)"),
        ("macro_f1", "MF1(%)"),
        ("spk_acc", "SpkAcc(%)"),
        ("spk_uar", "SpkUAR(%)"),
    ]

    # check availability of spk_uar
    spk_uar_available = False
    for seed, exp_map in per_seed_exp.items():
        for expn in exp_map.keys():
            if _get_float(exp_map[expn], "spk_uar") is not None:
                spk_uar_available = True
                break
        if spk_uar_available:
            break

    _print_block("Loaded seeds")
    for seed, exp_map in sorted(per_seed_exp.items(), key=lambda x: x[0]):
        print(f"seed={seed} | exps={list(exp_map.keys())}")

    _print_block("Metric availability")
    print("spk_uar column:", "YES" if spk_uar_available else "NO (will skip SpkUAR tests/summary if missing)")

    # summarize each exp over seeds
    exps_to_report = [args.exp_a, args.exp_c]
    if sat_name_resolved is not None:
        exps_to_report.append(sat_name_resolved)

    _print_block("Mean ± std over seeds (from best checkpoints in each seed CSV)")
    # header
    header = ["EXP"] + [m[1] for m in metrics if (m[0] != "spk_uar" or spk_uar_available)]
    print(" | ".join([f"{h:>12s}" for h in header]))

    for expn in exps_to_report:
        row_out = [f"{expn:>12s}"]
        for key, disp in metrics:
            if key == "spk_uar" and not spk_uar_available:
                continue
            seeds, vals = collect_metric_by_seed(per_seed_exp, expn, key)
            if len(vals) == 0:
                row_out.append(f"{'NA':>12s}")
            else:
                mu, sd = mean_std(vals)
                row_out.append(f"{format_ms(mu, sd, digits=args.digits):>12s}")
        print(" | ".join(row_out))

    # paired t-tests
    _print_block("Paired t-tests by seed (exp2 - exp1)")
    print(f"Comparisons: A={args.exp_a}  C={args.exp_c}  SAT={sat_name_resolved if sat_name_resolved else 'NOT FOUND'}")

    def do_pair(exp1: str, exp2: str, metric_key: str, label: str):
        seeds, v1, v2 = aligned_pair_values(per_seed_exp, exp1, exp2, metric_key)
        if len(seeds) == 0:
            print(f"{label:<16s}: NA (missing metric/exp in CSVs)")
            return
        diffs = v2 - v1
        res = paired_ttest(diffs)
        unit = "%"  # all metrics are percentages in your CSV
        _print_ttest_line(label, res, unit=unit)
        print(f"  seeds used: {seeds}")
        print(f"  diffs ({exp2}-{exp1}): {diffs.tolist()}")

    # A vs C
    do_pair(args.exp_a, args.exp_c, "uar", "A vs C (UAR)")
    do_pair(args.exp_a, args.exp_c, "acc", "A vs C (Acc)")
    do_pair(args.exp_a, args.exp_c, "macro_f1", "A vs C (MF1)")
    do_pair(args.exp_a, args.exp_c, "spk_acc", "A vs C (SpkAcc)")
    if spk_uar_available:
        do_pair(args.exp_a, args.exp_c, "spk_uar", "A vs C (SpkUAR)")

    # C vs SAT
    if sat_name_resolved is not None:
        do_pair(args.exp_c, sat_name_resolved, "uar", "C vs SAT (UAR)")
        do_pair(args.exp_c, sat_name_resolved, "acc", "C vs SAT (Acc)")
        do_pair(args.exp_c, sat_name_resolved, "macro_f1", "C vs SAT (MF1)")
        do_pair(args.exp_c, sat_name_resolved, "spk_acc", "C vs SAT (SpkAcc)")
        if spk_uar_available:
            do_pair(args.exp_c, sat_name_resolved, "spk_uar", "C vs SAT (SpkUAR)")
    else:
        print("\nSAT experiment name not found in provided CSVs -> skipping C vs SAT tests.")

    _print_block("Done")
    if not _HAS_SCIPY:
        print("Note: scipy not found. t-stat shown, but p-values/CI are NA.")
        print("Install scipy if needed:  pip install scipy")


if __name__ == "__main__":
    main()