import os
import numpy as np
import pandas as pd

CSV = "/root/autodl-tmp/CASIA/metadata.csv"
OUTDIR = "/root/autodl-tmp/CASIA/v2"

label_map = {
    "Anger": 0,
    "Fear": 1,
    "Happiness": 2,
    "Neutral": 3,
    "Sadness": 4,
    "Surprise": 5,
}

df = pd.read_csv(CSV)
df = df[df["emotion"].isin(label_map)].copy()

# 固定排序，保证可复现
df = df.sort_values(["speaker", "emotion", "wav_path"]).reset_index(drop=True)

file_paths = df["wav_path"].astype(str).tolist()
y = np.array([label_map[e] for e in df["emotion"].tolist()], dtype=np.int64)
spk = np.array(df["speaker"].astype(str).tolist(), dtype=object)

os.makedirs(OUTDIR, exist_ok=True)
np.save(f"{OUTDIR}/CASIA_file_paths.npy", np.array(file_paths, dtype=object))
np.save(f"{OUTDIR}/CASIA_y_labels.npy", y)
np.save(f"{OUTDIR}/CASIA_speaker_ids.npy", spk, allow_pickle=True)

print("Saved to:", OUTDIR)
print("N =", len(y))
print("Speakers =", sorted(set(spk.tolist())))
print("Label counts:", {k: int((df.emotion==k).sum()) for k in label_map})