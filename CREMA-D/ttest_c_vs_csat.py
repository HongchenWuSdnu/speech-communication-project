# ttest_c_vs_csat.py
import os
import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats

SEEDS = [13, 42, 2026]
ROOT_TEMPLATE = "/root/autodl-tmp/ablation_chain_CsatNoCB_seed{seed}/ablation_chain_summary.csv"

def load_uar(seed: int):
    path = ROOT_TEMPLATE.format(seed=seed)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    def get(exp_name: str) -> float:
        row = df[df["exp"] == exp_name]
        if len(row) != 1:
            raise ValueError(f"Seed {seed}: cannot find unique row for {exp_name}")
        return float(row["uar"].iloc[0])
    return {
        "A": get("A_base"),
        "C": get("C_plus_specaug"),
        "Csat": get("C_plus_SAT_noCB"),
    }

def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1))

def ci95_of_mean(d):
    # 95% CI for mean difference with df=n-1
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = d.mean()
    s = d.std(ddof=1)
    tcrit = stats.t.ppf(0.975, df=n-1)
    half = tcrit * s / sqrt(n)
    return float(m - half), float(m + half)

def main():
    A, C, Csat = [], [], []
    for s in SEEDS:
        u = load_uar(s)
        A.append(u["A"])
        C.append(u["C"])
        Csat.append(u["Csat"])

    A_m, A_s = mean_std(A)
    C_m, C_s = mean_std(C)
    Csat_m, Csat_s = mean_std(Csat)

    d_AC = np.array(C) - np.array(A)
    d_C_Csat = np.array(Csat) - np.array(C)

    t1 = stats.ttest_rel(C, A)
    t2 = stats.ttest_rel(Csat, C)

    ci1 = ci95_of_mean(d_AC)
    ci2 = ci95_of_mean(d_C_Csat)

    print("=== UAR (mean ± std) over seeds ===")
    print(f"A_base         : {A_m:.3f} ± {A_s:.3f}")
    print(f"C_plus_specaug : {C_m:.3f} ± {C_s:.3f}")
    print(f"C+SAT_noCB     : {Csat_m:.3f} ± {Csat_s:.3f}")
    print()

    print("=== Paired t-test on UAR (by seed) ===")
    print(f"A vs C:     meanΔ={d_AC.mean():.3f} | t={t1.statistic:.3f} | p={t1.pvalue:.6f} | 95%CI=[{ci1[0]:.3f},{ci1[1]:.3f}]")
    print(f"C vs C+SAT: meanΔ={d_C_Csat.mean():.3f} | t={t2.statistic:.3f} | p={t2.pvalue:.6f} | 95%CI=[{ci2[0]:.3f},{ci2[1]:.3f}]")
    print()

if __name__ == "__main__":
    main()