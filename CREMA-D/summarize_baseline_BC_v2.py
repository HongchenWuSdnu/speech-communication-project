import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
from scipy.stats import t as tdist

paths = [
    "/root/autodl-tmp/cremad_baselines_BC_v2_noDeep/seed13/ablation_chain_summary_with_spk_uar.csv",
    "/root/autodl-tmp/cremad_baselines_BC_v2_noDeep/seed42/ablation_chain_summary_with_spk_uar.csv",
    "/root/autodl-tmp/cremad_baselines_BC_v2_noDeep/seed2026/ablation_chain_summary_with_spk_uar.csv",
]

def paired_ttest(a, b):
    # two-sided paired t-test, n=3
    d = np.array(a) - np.array(b)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    t = mean / (sd / sqrt(n)) if sd > 0 else np.inf
    p = 2 * (1 - tdist.cdf(abs(t), df=n-1)) if np.isfinite(t) else 0.0
    return mean, sd, t, p

dfs = []
for p in paths:
    seed = int(p.split("/seed")[1].split("/")[0])
    df = pd.read_csv(p)
    df["seed"] = seed
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

metrics = ["acc", "uar", "macro_f1", "spk_acc", "spk_uar"]

# ---- Table 2 summary (mean±std) ----
print("\n=== Mean±Std over 3 seeds ===")
summary = df.groupby("exp")[metrics].agg(["mean", "std"])
print(summary)

# ---- Key paired tests (same seeds) ----
print("\n=== Paired t-tests (two-sided), n=3 ===")
def get(exp, metric):
    s = df[df["exp"] == exp].sort_values("seed")[metric].to_list()
    return s

comparisons = [
    ("A_trad_only_MLP", "O_gate_C"),
    ("C_concat_C", "O_gate_C"),
    ("O_gate_C", "O_gate_C_SAT"),
    ("C_concat_C", "C_concat_C_SAT"),
]

for a, b in comparisons:
    print(f"\n[{a}] vs [{b}]  (Δ = {a} - {b})")
    for m in metrics:
        mean, sd, t, p = paired_ttest(get(a, m), get(b, m))
        print(f"  {m:8s}: Δ={mean:+.4f} ± {sd:.4f} | t={t:.3f} | p={p:.6f}")