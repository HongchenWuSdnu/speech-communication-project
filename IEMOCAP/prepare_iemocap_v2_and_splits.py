# prepare_iemocap_v2_and_splits.py
import os, re, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

ROOT = "/root/autodl-tmp/IEMOCAP"

def find_one(root, filename):
    hits = []
    for dp, dn, fn in os.walk(root):
        if filename in fn:
            hits.append(os.path.join(dp, filename))
    if len(hits) == 0:
        raise FileNotFoundError(f"Cannot find {filename} under {root}")
    if len(hits) > 1:
        # choose the shortest path (usually the main one)
        hits = sorted(hits, key=lambda x: len(x))
    return hits[0]

def parse_session_and_speaker(file_id: str, audio_path: str):
    """
    file_id example: Ses05M_script03_2_M016
    parse session=5, speaker="Ses05M" (10 speakers total: Ses01F/Ses01M/.../Ses05F/Ses05M)
    """
    m = re.match(r"Ses(\d\d)([MF])", file_id)
    if m:
        sess = int(m.group(1))
        spk = f"Ses{m.group(1)}{m.group(2)}"
        return sess, spk

    # fallback: try from path like .../Session5/...
    m2 = re.search(r"/Session(\d)/", audio_path.replace("\\", "/"))
    sess = int(m2.group(1)) if m2 else -1
    # if no speaker, use unknown bucket
    spk = "UNK"
    return sess, spk

def main():
    os.makedirs(ROOT, exist_ok=True)
    csv_path = find_one(ROOT, "training_data.csv")
    print(f"[OK] training_data.csv = {csv_path}")

    df = pd.read_csv(csv_path)
    assert set(["file_id", "emotion", "audio_path"]).issubset(df.columns), df.columns

    # 6-class mapping
    emo_map = {
        "Anger": "ang",
        "Happiness": "hap",
        "Neutral": "neu",
        "Sadness": "sad",
        "Frustration": "fru",
        "Excited": "exc",
    }
    df = df[df["emotion"].isin(emo_map.keys())].copy()
    df["emo6"] = df["emotion"].map(emo_map)

    # resolve paths: Kaggle csv may contain /kaggle/working/...; we rewrite to local extracted root if needed
    # We try to locate the "audio_files" directory under ROOT and replace prefix up to "audio_files".
    audio_root = None
    for dp, dn, fn in os.walk(ROOT):
        if os.path.basename(dp) == "audio_files":
            audio_root = dp
            break
    if audio_root is None:
        # maybe it's directly in iemocap-audio-complete/audio_files
        hits = []
        for dp, dn, fn in os.walk(ROOT):
            if "audio_files" in dn:
                hits.append(os.path.join(dp, "audio_files"))
        if hits:
            audio_root = sorted(hits, key=len)[0]
    if audio_root is None:
        raise FileNotFoundError("Cannot find audio_files directory under /root/autodl-tmp/IEMOCAP after unzip.")

    print(f"[OK] audio_root = {audio_root}")

    def rewrite_path(p):
        p2 = p.replace("\\", "/")
        if "/audio_files/" in p2:
            tail = p2.split("/audio_files/", 1)[1]
            return os.path.join(audio_root, tail)
        # if already relative to extracted structure
        if p2.endswith(".wav") and os.path.exists(p2):
            return p2
        # last resort: assume it's already under audio_root
        return os.path.join(audio_root, os.path.basename(p2))

    df["wav_path"] = df["audio_path"].apply(rewrite_path)
    missing = df[~df["wav_path"].apply(os.path.exists)]
    if len(missing) > 0:
        print("[WARN] missing wav files:", len(missing))
        print(missing[["file_id", "wav_path"]].head(20).to_string(index=False))
        raise FileNotFoundError("Some wav files not found after path rewrite. Fix paths first.")

    # parse session_id + speaker_id
    sess_ids = []
    spk_ids = []
    for fid, ap in zip(df["file_id"].tolist(), df["wav_path"].tolist()):
        sess, spk = parse_session_and_speaker(str(fid), str(ap))
        sess_ids.append(sess)
        spk_ids.append(spk)
    df["session_id"] = sess_ids
    df["speaker_id"] = spk_ids

    # label ids (fixed order for reproducibility)
    label_order = ["ang", "hap", "neu", "sad", "fru", "exc"]
    label2id = {k:i for i,k in enumerate(label_order)}
    y = np.array([label2id[e] for e in df["emo6"].tolist()], dtype=np.int64)

    file_paths = np.array(df["wav_path"].tolist(), dtype=object)
    speaker_ids = np.array(df["speaker_id"].tolist(), dtype=object)
    session_ids = np.array(df["session_id"].tolist(), dtype=np.int64)

    # save v2
    v2_dir = os.path.join(ROOT, "iemocap_v2")
    os.makedirs(v2_dir, exist_ok=True)

    np.save(os.path.join(v2_dir, "SCI_file_paths.npy"), file_paths)
    np.save(os.path.join(v2_dir, "SCI_y_labels.npy"), y)
    np.save(os.path.join(v2_dir, "SCI_speaker_ids.npy"), speaker_ids, allow_pickle=True)
    np.save(os.path.join(v2_dir, "SCI_session_ids.npy"), session_ids)

    # dataset stats
    print("\n=== IEMOCAP-6 stats (after filtering) ===")
    print("N =", len(df))
    print("Class counts:", Counter(df["emo6"].tolist()))
    print("Session counts:", Counter(df["session_id"].tolist()))
    print("Speaker counts:", Counter(df["speaker_id"].tolist()))

    # make Session-LOSO splits (5 folds)
    splits_dir = os.path.join(ROOT, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    all_idx = np.arange(len(df), dtype=np.int64)
    for s in [1,2,3,4,5]:
        test_idx = all_idx[session_ids == s]
        train_idx = all_idx[session_ids != s]

        split = {
            "name": f"session_loso_S{s}",
            "test_session": s,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "label_order": label_order,
            "label2id": label2id,
        }
        out_json = os.path.join(splits_dir, f"session_loso_S{s}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {out_json} | train={len(train_idx)} test={len(test_idx)}")

    # also write a global meta json
    meta = {
        "label_order": label_order,
        "label2id": label2id,
        "v2_dir": v2_dir,
        "splits_dir": splits_dir,
    }
    with open(os.path.join(ROOT, "iemocap_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {os.path.join(ROOT, 'iemocap_meta.json')}")

if __name__ == "__main__":
    main()