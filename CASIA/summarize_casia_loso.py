#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import math
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def mean_std(x: List[float]) -> Tuple[float, float]:
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))

def paired_ttest_two_sided(x: List[float], y: List[float]) -> float:
    """
    Paired two-sided t-test p-value WITHOUT scipy (n small).
    Uses Student-t distribution via mpmath if available; otherwise normal approx.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    d = x - y
    n = d.size
    if n < 2:
        return float("nan")
    md = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        # all diffs identical -> p ~ 0 if md!=0 else 1
        return 0.0 if md != 0 else 1.0
    t = md / (sd / math.sqrt(n))
    df = n - 1

    try:
        import mpmath as mp
        # two-sided p-value: 2*(1 - CDF(|t|))
        # Student-t CDF using betainc
        tt = abs(t)
        # CDF for t>0: 1 - 0.5*I_{df/(df+t^2)}(df/2, 1/2)
        xval = df / (df + tt**2)
        I = mp.betainc(df/2, 0.5, 0, xval, regularized=True)
        cdf = 1 - 0.5 * I
        p = 2 * (1 - cdf)
        return float(p)
    except Exception:
        # fallback: normal approx (rough but OK for quick sanity)
        import math as _m
        tt = abs(t)
        # normal cdf
        cdf = 0.5 * (1 + _m.erf(tt / _m.sqrt(2)))
        p = 2 * (1 - cdf)
        return float(p)

def load_one_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expected columns:
    # exp,best_epoch,acc,uar,macro_f1,spk_acc,spk_uar,ckpt
    need = {"exp","acc","uar","macro_f1","spk_acc","spk_uar"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    return df

def collect_runs(root: str) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    returns:
      runs[fold_name][seed] = dataframe(csv)
    expects:
      root/fold_test_xxx/seed13/ablation_chain_summary_with_spk_uar.csv
    """
    runs: Dict[str, Dict[int, pd.DataFrame]] = {}
    fold_dirs = sorted([d for d in glob.glob(os.path.join(root, "fold_test_*")) if os.path.isdir(d)])
    if not fold_dirs:
        raise RuntimeError(f"No fold_test_* directories under {root}")

    for fd in fold_dirs:
        fold_name = os.path.basename(fd)
        runs[fold_name] = {}
        seed_dirs = sorted([d for d in glob.glob(os.path.join(fd, "seed*")) if os.path.isdir(d)])
        for sd in seed_dirs:
            seed_str = os.path.basename(sd).replace("seed","")
            try:
                seed = int(seed_str)
            except:
                continue
            csv_path = os.path.join(sd, "ablation_chain_summary_with_spk_uar.csv")
            if os.path.exists(csv_path):
                runs[fold_name][seed] = load_one_csv(csv_path)
    return runs

def get_metric(df: pd.DataFrame, exp: str, metric: str) -> float:
    row = df[df["exp"] == exp]
    if row.empty:
        raise KeyError(f"exp={exp} not found in csv")
    return float(row.iloc[0][metric])

def summarize_fold_over_seeds(runs_fold: Dict[int, pd.DataFrame], exps: List[str], metrics: List[str]) -> pd.DataFrame:
    """
    For one fold:
      returns df with rows=exp and columns metric_mean, metric_std over seeds
    """
    seeds = sorted(runs_fold.keys())
    out_rows = []
    for exp in exps:
        row = {"exp": exp}
        for m in metrics:
            vals = [get_metric(runs_fold[s], exp, m) for s in seeds]
            mu, sd = mean_std(vals)
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sd
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def summarize_over_folds(fold_summaries: Dict[str, pd.DataFrame], exps: List[str], metrics: List[str]) -> pd.DataFrame:
    """
    Each fold_summary already has metric_mean (seed-mean).
    Now compute mean±std over folds using these metric_mean values.
    """
    out_rows = []
    fold_names = sorted(fold_summaries.keys())
    for exp in exps:
        row = {"exp": exp}
        for m in metrics:
            vals = []
            for fn in fold_names:
                df = fold_summaries[fn]
                v = float(df[df["exp"] == exp].iloc[0][f"{m}_mean"])
                vals.append(v)
            mu, sd = mean_std(vals)
            row[f"{m}_mean_over_folds"] = mu
            row[f"{m}_std_over_folds"] = sd
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def latex_table_main(df: pd.DataFrame, caption: str, label: str, use_cols: List[Tuple[str,str]]) -> str:
    """
    use_cols: list of (metric_base, display_name)
    expects df has metric_mean_over_folds and metric_std_over_folds
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    colspec = "l" + "c"*len(use_cols)
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    header = ["Method"] + [name for _, name in use_cols]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        exp = r["exp"]
        cells = [exp]
        for m, _disp in use_cols:
            mu = r[f"{m}_mean_over_folds"] * 100.0
            sd = r[f"{m}_std_over_folds"] * 100.0
            cells.append(f"{mu:.2f}$\\pm${sd:.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def latex_table_ttest(rows: List[Dict[str, float]], caption: str, label: str) -> str:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\toprule")
    lines.append("Comparison & Metric & $\\Delta$ (mean$\\pm$std) & $p$-value \\\\")
    lines.append("\\midrule")
    for rr in rows:
        comp = rr["comparison"]
        metric = rr["metric"]
        delta_mu = rr["delta_mu"] * 100.0
        delta_sd = rr["delta_sd"] * 100.0
        p = rr["p"]
        lines.append(f"{comp} & {metric} & {delta_mu:+.2f}$\\pm${delta_sd:.2f} & {p:.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Output root, e.g. /root/autodl-tmp/casia_main_v2_loso")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13,42,2026])
    ap.add_argument("--exp_a", default="C_gate")
    ap.add_argument("--exp_b", default="E_gate_SAT_noCB")
    ap.add_argument("--out_csv", default="casia_loso_summary.csv")
    ap.add_argument("--out_tex", default="casia_loso_tables.tex")
    args = ap.parse_args()

    exps = [args.exp_a, args.exp_b]
    metrics = ["acc","uar","macro_f1","spk_acc","spk_uar"]

    runs = collect_runs(args.root)

    # fold-level summary (over seeds)
    fold_summaries: Dict[str, pd.DataFrame] = {}
    for fold_name, runs_fold in runs.items():
        # ensure all seeds exist
        missing = [s for s in args.seeds if s not in runs_fold]
        if missing:
            raise RuntimeError(f"Fold {fold_name} missing seeds {missing}. Found {sorted(runs_fold.keys())}")
        fold_summaries[fold_name] = summarize_fold_over_seeds(runs_fold, exps, metrics)

    # overall summary over folds (using fold means)
    overall = summarize_over_folds(fold_summaries, exps, metrics)

    # Save a tidy csv
    overall.to_csv(os.path.join(args.root, args.out_csv), index=False)

    # ===== paired t-test over folds (fold-level means) =====
    fold_names = sorted(fold_summaries.keys())

    trows = []
    for m in ["uar","spk_acc","spk_uar","acc","macro_f1"]:
        A = []
        B = []
        for fn in fold_names:
            df = fold_summaries[fn]
            a = float(df[df["exp"]==args.exp_a].iloc[0][f"{m}_mean"])
            b = float(df[df["exp"]==args.exp_b].iloc[0][f"{m}_mean"])
            A.append(a); B.append(b)
        d = np.array(A) - np.array(B)
        dmu, dsd = mean_std(d.tolist())
        p = paired_ttest_two_sided(A, B)
        trows.append({
            "comparison": f"{args.exp_a} vs {args.exp_b}",
            "metric": m.upper() if m!="macro_f1" else "MF1",
            "delta_mu": dmu,
            "delta_sd": dsd,
            "p": p
        })

    # ===== LaTeX tables =====
    # Main table: report mean±std over folds (percent)
    main_tex = latex_table_main(
        overall,
        caption="CASIA LOSO generalization results (mean$\\pm$std over 4 folds; each fold averaged over 3 seeds).",
        label="tab:casia_loso_main",
        use_cols=[("acc","Acc (\\%)"),("uar","UAR (\\%)"),("macro_f1","MF1 (\\%)"),("spk_acc","SpkAcc (\\%)"),("spk_uar","SpkUAR (\\%)")]
    )
    ttest_tex = latex_table_ttest(
        trows,
        caption="Two-sided paired $t$-tests over 4 LOSO folds on CASIA (fold-level means; $\\Delta$=first-second).",
        label="tab:casia_loso_ttest"
    )

    tex_path = os.path.join(args.root, args.out_tex)
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by summarize_casia_loso.py\n")
        f.write("\\usepackage{booktabs}\n\n")
        f.write(main_tex + "\n\n")
        f.write(ttest_tex + "\n")
    print("Saved:")
    print("  CSV :", os.path.join(args.root, args.out_csv))
    print("  TeX :", tex_path)

    # Also print a readable summary
    print("\n=== Overall (over folds) ===")
    print(overall)

    print("\n=== Fold means (for sanity) ===")
    for fn in fold_names:
        print("\n---", fn, "---")
        print(fold_summaries[fn][["exp"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]])

if __name__ == "__main__":
    main()