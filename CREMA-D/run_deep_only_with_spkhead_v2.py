#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_deep_only_with_spkhead_v2.py

Purpose:
  Deep-only baseline for CREMA-D (6-class) under speaker-independent split,
  WITH a speaker classifier head to measure leakage (SpkAcc / SpkUAR).

Key points:
  - Uses v2 aligned arrays:
      SCI_file_paths.npy, SCI_y_labels.npy, SCI_speaker_ids.npy (allow_pickle), SCI_X_trad_v2.npy (only for split)
  - Speaker-independent split: GroupShuffleSplit(test_size=0.2, random_state=42), groups=speaker_ids_raw
  - Wav2Vec2 loaded offline from MODEL_DIR; feature encoder frozen; (by default) ALL wav2vec2 params frozen.
  - Trainable parts: BiLSTM + AttnPool + Emotion head + Speaker head
  - Optional SAT: can be enabled, but by default OFF (we just want leakage measurement)

Outputs (per seed):
  - ckpt: <out_root>/seed<SEED>/deep_only_spkhead_best.pt
  - csv : <out_root>/seed<SEED>/deep_only_spkhead_summary.csv
"""

import os, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from transformers import Wav2Vec2Model

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# =============== v2 paths ===============
MODEL_DIR = "/root/autodl-tmp/wav2vec2-base-local"
FILE_PATHS_NPY = "/root/autodl-tmp/SCI_file_paths.npy"
X_TRAD_PATH    = "/root/autodl-tmp/SCI_X_trad_v2.npy"     # only for split
Y_EMO_PATH     = "/root/autodl-tmp/SCI_y_labels.npy"
SPK_ID_PATH    = "/root/autodl-tmp/SCI_speaker_ids.npy"

# =============== protocol ===============
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============== audio ===============
TARGET_SR = 16000
MAX_SEC   = 3.0

# =============== training ===============
NUM_EPOCHS = 25
BATCH_SIZE = 16
ACCUM_STEPS = 2
NUM_WORKERS = 4

LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

WARMUP_EPOCHS = 5

# SAT (optional)
SAT_WARMUP_EPOCHS = 8
SAT_ALPHA_MAX     = 0.02
SAT_LAMBDA_MAX    = 0.003


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def cosine_lr_factor(epoch_idx_1based: int, total_epochs: int, warmup_epochs: int) -> float:
    e = epoch_idx_1based
    warmup_epochs = max(1, int(warmup_epochs))
    if e <= warmup_epochs:
        return e / warmup_epochs
    t = (e - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * t))

# ---------- GRL ----------
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

# ---------- Dataset ----------
class AudioDataset(Dataset):
    def __init__(self, file_paths, y_emo, y_spk, target_sr=16000, max_sec=3.0):
        self.file_paths = list(file_paths)
        self.y_emo = torch.tensor(y_emo, dtype=torch.long)
        self.y_spk = torch.tensor(y_spk, dtype=torch.long)
        self.target_sr = target_sr
        self.target_len = int(target_sr * max_sec)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        import soundfile as sf
        wav, sr = sf.read(self.file_paths[idx], dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav)

        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(wav)

        n = wav.shape[0]
        if n < self.target_len:
            wav = F.pad(wav, (0, self.target_len - n))
        else:
            wav = wav[:self.target_len]

        return wav, self.y_emo[idx], self.y_spk[idx]

def collate_fn(batch):
    wavs = torch.stack([b[0] for b in batch], dim=0)  # (B,L)
    attn = torch.ones_like(wavs, dtype=torch.long)
    y_emo = torch.stack([b[1] for b in batch], dim=0)
    y_spk = torch.stack([b[2] for b in batch], dim=0)
    return wavs, attn, y_emo, y_spk

# ---------- Model ----------
class DeepOnlyWithSpkHead(nn.Module):
    def __init__(self, num_emotions: int, num_speakers: int, freeze_w2v: bool = True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_DIR, local_files_only=True)
        self.wav2vec2.freeze_feature_encoder()

        if freeze_w2v:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=768, hidden_size=64,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions),
        )
        self.speaker_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_speakers),
        )

    def forward(self, input_values, attention_mask, alpha: float = 0.0):
        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hs = out.last_hidden_state  # (B,T,768)
        lstm_out, _ = self.lstm(hs)  # (B,T,128)
        attn = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (B,T,1)
        h = torch.sum(lstm_out * attn, dim=1)  # (B,128)

        emo_logits = self.emotion_head(h)
        spk_logits = self.speaker_head(grad_reverse(h, alpha))
        return emo_logits, spk_logits

# ---------- Train/Eval ----------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    s_true, s_pred = [], []
    for wavs, attn, y_emo, y_spk in loader:
        wavs = wavs.to(device)
        attn = attn.to(device)
        emo_logits, spk_logits = model(wavs, attn, alpha=0.0)

        yhat = emo_logits.argmax(dim=1).cpu().numpy()
        shat = spk_logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y_emo.numpy()); y_pred.extend(yhat)
        s_true.extend(y_spk.numpy()); s_pred.extend(shat)

    acc = accuracy_score(y_true, y_pred) * 100
    uar = recall_score(y_true, y_pred, average="macro") * 100
    mf1 = f1_score(y_true, y_pred, average="macro") * 100
    spk_acc = accuracy_score(s_true, s_pred) * 100
    spk_uar = recall_score(s_true, s_pred, average="macro") * 100
    return acc, uar, mf1, spk_acc, spk_uar

def train_one_epoch(model, loader, optimizer, device, ce_emo, ce_spk,
                    alpha, lam, accum_steps=1, max_grad_norm=1.0, use_amp=False, scaler=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    emo_sum, spk_sum, n = 0.0, 0.0, 0

    for step, (wavs, attn, y_emo, y_spk) in enumerate(loader):
        wavs = wavs.to(device)
        attn = attn.to(device)
        y_emo = y_emo.to(device)
        y_spk = y_spk.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            emo_logits, spk_logits = model(wavs, attn, alpha=alpha)
            loss_emo = ce_emo(emo_logits, y_emo)
            loss_spk = ce_spk(spk_logits, y_spk)
            loss = (loss_emo + lam * loss_spk) / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if max_grad_norm is not None:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if use_amp:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        emo_sum += float(loss_emo.item())
        spk_sum += float(loss_spk.item())
        n += 1

    return emo_sum / max(n, 1), spk_sum / max(n, 1)

def run_seed(seed: int, out_root: str, use_sat: bool, freeze_w2v: bool):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Seed={seed} | Device={device} | SAT={use_sat} | freeze_w2v={freeze_w2v}")

    seed_dir = os.path.join(out_root, f"seed{seed}")
    ensure_dir(seed_dir)

    file_paths = np.load(FILE_PATHS_NPY, allow_pickle=True)
    X_trad = np.load(X_TRAD_PATH)  # for split only
    y_emo_raw = np.load(Y_EMO_PATH)
    spk_raw = np.load(SPK_ID_PATH, allow_pickle=True)

    emo_enc = LabelEncoder()
    y_emo = emo_enc.fit_transform(y_emo_raw)

    spk_enc = LabelEncoder()
    y_spk = spk_enc.fit_transform(spk_raw)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X_trad, y_emo, groups=spk_raw))

    print(f"Train samples: {len(tr_idx)} | Test samples: {len(te_idx)}")
    print(f"Train speakers: {len(np.unique(spk_raw[tr_idx]))} | Test speakers: {len(np.unique(spk_raw[te_idx]))}")

    train_ds = AudioDataset(file_paths[tr_idx], y_emo[tr_idx], y_spk[tr_idx], TARGET_SR, MAX_SEC)
    test_ds  = AudioDataset(file_paths[te_idx], y_emo[te_idx], y_spk[te_idx], TARGET_SR, MAX_SEC)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    model = DeepOnlyWithSpkHead(num_emotions=len(emo_enc.classes_), num_speakers=len(spk_enc.classes_),
                                freeze_w2v=freeze_w2v).to(device)

    ce_emo = nn.CrossEntropyLoss()
    ce_spk = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    use_amp = torch.cuda.is_available()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    base_lr = LR_HEAD
    best = {"uar": -1, "acc": -1, "mf1": -1, "spk_acc": -1, "spk_uar": -1, "epoch": -1}
    ckpt_path = os.path.join(seed_dir, "deep_only_spkhead_best.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        fac = cosine_lr_factor(epoch, NUM_EPOCHS, WARMUP_EPOCHS)
        for g in optimizer.param_groups:
            g["lr"] = base_lr * fac

        if (not use_sat):
            alpha, lam = 0.0, 0.0
        else:
            if epoch <= SAT_WARMUP_EPOCHS:
                alpha, lam = 0.0, 0.0
            else:
                t = (epoch - SAT_WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - SAT_WARMUP_EPOCHS))
                t = min(1.0, max(0.0, float(t)))
                alpha = SAT_ALPHA_MAX * t
                lam = SAT_LAMBDA_MAX * t

        tr_emo, tr_spk = train_one_epoch(model, train_loader, optimizer, device, ce_emo, ce_spk,
                                         alpha=alpha, lam=lam, accum_steps=ACCUM_STEPS,
                                         max_grad_norm=MAX_GRAD_NORM, use_amp=use_amp, scaler=amp_scaler)

        acc, uar, mf1, spk_acc, spk_uar = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"TrainEmo={tr_emo:.4f} TrainSpk={tr_spk:.4f} | "
              f"Val Acc={acc:.2f} UAR={uar:.2f} MF1={mf1:.2f} | "
              f"SpkAcc={spk_acc:.2f} SpkUAR={spk_uar:.2f} | alpha={alpha:.3f} lam={lam:.4f}")

        if uar > best["uar"]:
            best = {"uar": uar, "acc": acc, "mf1": mf1, "spk_acc": spk_acc, "spk_uar": spk_uar, "epoch": epoch}
            torch.save({
                "model": model.state_dict(),
                "best": best,
                "emo_classes": emo_enc.classes_,
                "spk_classes": spk_enc.classes_,
                "seed": seed,
                "split_random_state": SPLIT_RANDOM_STATE,
                "use_sat": use_sat,
                "freeze_w2v": freeze_w2v,
            }, ckpt_path)
            print(f"  ✅ Saved best ckpt: {ckpt_path} | best UAR={best['uar']:.2f} (epoch {best['epoch']})")

    # write a small csv
    csv_path = os.path.join(seed_dir, "deep_only_spkhead_summary.csv")
    with open(csv_path, "w") as f:
        f.write("seed,best_epoch,acc,uar,mf1,spk_acc,spk_uar,ckpt\n")
        f.write(f"{seed},{best['epoch']},{best['acc']:.4f},{best['uar']:.4f},{best['mf1']:.4f},"
                f"{best['spk_acc']:.4f},{best['spk_uar']:.4f},{ckpt_path}\n")

    print(f"\n✅ Seed {seed} done. CSV: {csv_path}")
    return csv_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    ap.add_argument("--use_sat", action="store_true", help="Enable SAT (optional). Default OFF.")
    ap.add_argument("--unfreeze_w2v", action="store_true", help="Unfreeze wav2vec2 (more compute). Default OFF.")
    args = ap.parse_args()

    ensure_dir(args.out_root)
    freeze_w2v = not args.unfreeze_w2v

    csvs = []
    for s in args.seeds:
        csvs.append(run_seed(s, args.out_root, use_sat=args.use_sat, freeze_w2v=freeze_w2v))

    print("\n✅ All done.")
    for c in csvs:
        print(" -", c)

if __name__ == "__main__":
    main()