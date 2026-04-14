cd /root/autodl-tmp/IEMOCAP
cat > summarize_iemocap_loso.py <<'PY'
import os, re
import pandas as pd
import numpy as np

OLD_ROOT = "/root/autodl-tmp/IEMOCAP/iemocap_main_v2_loso"     # S1-S4
NEW_ROOT = "/root/IEMOCAP_OUT/iemocap_main_v2_loso"           # S5
OUT_DIR  = "/root/autodl-tmp/IEMOCAP"

FOLDS = ["S1","S2","S3","S4","S5"]
SEEDS = ["seed13","seed42","seed2026"]
EXPS  = ["C_gate","E_gate_SAT_noCB"]

# expected columns in your per-seed summary CSV (adjust if naming differs)
# We'll accept flexible naming by mapping
CAND = {
    "acc":      ["acc", "Acc"],
    "uar":      ["uar", "UAR"],
    "macro_f1": ["macro_f1", "mf1", "MacroF1", "MF1"],
    "spk_acc":  ["spk_acc", "SpkAcc", "speaker_acc"],
    "spk_uar":  ["spk_uar", "SpkUAR", "speaker_uar"],
}

def pick_col(df, key):
    for c in CAND[key]:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find a column for {key}. Have: {list(df.columns)}")

def root_for_fold(fold):
    return NEW_ROOT if fold == "S5" else OLD_ROOT

def load_seed_summary(fold, seed):
    root = root_for_fold(fold)
    p = os.path.join(root, fold, seed, "ablation_chain_summary_with_spk_uar.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    return df, p

def extract_metrics(df, exp_name):
    row = df[df["exp"] == exp_name]
    if len(row) != 1:
        raise ValueError(f"exp={exp_name} not uniquely found, got {len(row)}")
    row = row.iloc[0]
    return {
        "acc": float(row[pick_col(df,"acc")]),
        "uar": float(row[pick_col(df,"uar")]),
        "macro_f1": float(row[pick_col(df,"macro_f1")]),
        "spk_acc": float(row[pick_col(df,"spk_acc")]),
        "spk_uar": float(row[pick_col(df,"spk_uar")]),
    }

def mean_std(x):
    x = np.array(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=0))

def fmt(m, s):
    return f"{m*100:.2f}±{s*100:.2f}"

def main():
    # fold -> exp -> list of seed metrics dict
    fold_exp_seed = {fold:{exp:[] for exp in EXPS} for fold in FOLDS}
    paths_used = []

    for fold in FOLDS:
        for seed in SEEDS:
            df, p = load_seed_summary(fold, seed)
            paths_used.append(p)
            for exp in EXPS:
                fold_exp_seed[fold][exp].append(extract_metrics(df, exp))

    # fold means (avg over seeds)
    fold_means = {fold:{} for fold in FOLDS}
    for fold in FOLDS:
        for exp in EXPS:
            ms = fold_exp_seed[fold][exp]
            fold_means[fold][exp] = {k: np.mean([d[k] for d in ms]) for k in ms[0].keys()}

    # overall over folds
    rows = []
    for exp in EXPS:
        for key in ["acc","uar","macro_f1","spk_acc","spk_uar"]:
            pass
        vals = {key:[fold_means[fold][exp][key] for fold in FOLDS] for key in fold_means[FOLDS[0]][exp].keys()}
        m_acc,s_acc = mean_std(vals["acc"])
        m_uar,s_uar = mean_std(vals["uar"])
        m_f1,s_f1   = mean_std(vals["macro_f1"])
        m_sa,s_sa   = mean_std(vals["spk_acc"])
        m_su,s_su   = mean_std(vals["spk_uar"])
        rows.append({
            "exp": exp,
            "acc_mean": m_acc, "acc_std": s_acc,
            "uar_mean": m_uar, "uar_std": s_uar,
            "macro_f1_mean": m_f1, "macro_f1_std": s_f1,
            "spk_acc_mean": m_sa, "spk_acc_std": s_sa,
            "spk_uar_mean": m_su, "spk_uar_std": s_su,
            "Acc(%)": fmt(m_acc,s_acc),
            "UAR(%)": fmt(m_uar,s_uar),
            "MF1(%)": fmt(m_f1,s_f1),
            "SpkAcc(%)": fmt(m_sa,s_sa),
            "SpkUAR(%)": fmt(m_su,s_su),
        })

    out_csv = os.path.join(OUT_DIR, "iemocap_loso_summary.csv")
    out_tex = os.path.join(OUT_DIR, "iemocap_loso_tables.tex")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)

    # LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{IEMOCAP (6-class) results under session-LOSO. Mean$\pm$std over 5 folds; within each fold, results are averaged over 3 seeds.}")
    lines.append(r"\label{tab:iemocap_loso}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Acc (\%) & UAR (\%) & MF1 (\%) & SpkAcc (\%) & SpkUAR (\%) \\")
    lines.append(r"\midrule")
    for _, r in df_out.iterrows():
        lines.append(f"{r['exp']} & {r['Acc(%)']} & {r['UAR(%)']} & {r['MF1(%)']} & {r['SpkAcc(%)']} & {r['SpkUAR(%)']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", out_tex)
    print("[INFO] used per-seed CSVs:")
    for p in paths_used:
        print("  -", p)

if __name__ == "__main__":
    main()
