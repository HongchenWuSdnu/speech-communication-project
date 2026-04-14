import os, json
import numpy as np

V2 = "/root/autodl-tmp/CASIA/v2"
OUT = "/root/autodl-tmp/CASIA/v2/splits_loso"

file_paths = np.load(f"{V2}/CASIA_file_paths.npy", allow_pickle=True)
y = np.load(f"{V2}/CASIA_y_labels.npy", allow_pickle=True)
spk = np.load(f"{V2}/CASIA_speaker_ids.npy", allow_pickle=True)

speakers = sorted(list(set(spk.tolist())))
os.makedirs(OUT, exist_ok=True)

print("speakers:", speakers)

for test_spk in speakers:
    test_idx = np.where(spk == test_spk)[0].tolist()
    train_idx = np.where(spk != test_spk)[0].tolist()

    obj = {
        "dataset": "CASIA",
        "split_type": "LOSO",
        "test_speaker": test_spk,
        "n_total": int(len(spk)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "label_set": "6-class",
        "label_map": {"Anger":0,"Fear":1,"Happiness":2,"Neutral":3,"Sadness":4,"Surprise":5},
        "train_idx": train_idx,
        "test_idx": test_idx
    }
    path = f"{OUT}/fold_test_{test_spk}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("saved", path, "| train", len(train_idx), "test", len(test_idx))