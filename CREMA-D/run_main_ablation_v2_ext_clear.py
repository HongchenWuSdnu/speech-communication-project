#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_main_ablation_v2_ext_clear.py

CREMA-D SER main ablation (v2 aligned), clear naming, speaker leakage (SpkAcc + SpkUAR).

V2-aligned inputs:
  /root/autodl-tmp/SCI_file_paths.npy
  /root/autodl-tmp/SCI_y_labels.npy
  /root/autodl-tmp/SCI_speaker_ids.npy  (object dtype, allow_pickle=True)
  /root/autodl-tmp/SCI_X_trad_v2.npy    (aligned to file_paths)

Protocol:
  - Speaker-independent split: GroupShuffleSplit(test_size=0.2, random_state=42), groups=speaker_ids_raw
  - Best checkpoint selected by best UAR on test split (consistent with your previous scripts)
  - Report: Acc / UAR / MF1 + SpkAcc / SpkUAR

Experiments (clear naming):
  A_base
  C_gate              (extraTRF=1, SpecAug ON, fusion=gate)
  E_gate_SAT_noCB      (C_gate + SAT)
  C_gate_noTRF         (C_gate but extraTRF OFF)
  C_concat             (C_gate but fusion=concat)
  E_concat_SAT_noCB    (C_concat + SAT)
  C_concat_noTRF        (C_concat but extraTRF OFF)

Outputs per seed:
  <out_root>/seed<SEED>/ablation_chain_summary_with_spk_uar.csv
  and best ckpt per exp saved in the same seed folder.

"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torch.autograd import Function
from transformers import Wav2Vec2Model

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# =========================
# V2-aligned paths
# =========================
MODEL_DIR = "/root/autodl-tmp/wav2vec2-base-local"

FILE_PATHS_NPY = "/root/autodl-tmp/SCI_file_paths.npy"
X_TRAD_PATH    = "/root/autodl-tmp/SCI_X_trad_v2.npy"
Y_EMO_PATH     = "/root/autodl-tmp/SCI_y_labels.npy"
SPK_ID_PATH    = "/root/autodl-tmp/SCI_speaker_ids.npy"

# =========================
# Split protocol
# =========================
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2

# =========================
# Audio
# =========================
TARGET_SR = 16000
MAX_SEC   = 3.0

# =========================
# Training
# =========================
NUM_EPOCHS   = 30
BATCH_SIZE   = 16
ACCUM_STEPS  = 2
NUM_WORKERS  = 4

LR_W2V       = 1e-5
LR_HEAD      = 5e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

# Scheduler
WARMUP_EPOCHS = 5

# SAT schedule (noCB)
SAT_WARMUP_EPOCHS = 8
SAT_ALPHA_MAX     = 0.02
SAT_LAMBDA_MAX    = 0.003


# =========================
# Utils
# =========================
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

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# GRL
# =========================
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


# =========================
# Dataset / Collator
# =========================
class AudioTradDataset(Dataset):
    def __init__(self, file_paths, x_trad, y_emo, y_spk, target_sr=16000, max_sec=3.0):
        self.file_paths = list(file_paths)
        self.x_trad = torch.tensor(x_trad, dtype=torch.float32)
        self.y_emo  = torch.tensor(y_emo, dtype=torch.long)
        self.y_spk  = torch.tensor(y_spk, dtype=torch.long)
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

        return {
            "waveform": wav,
            "x_trad": self.x_trad[idx],
            "y_emo": self.y_emo[idx],
            "y_spk": self.y_spk[idx],
        }

class CollatorSimple:
    def __call__(self, batch):
        waveforms = torch.stack([b["waveform"] for b in batch], dim=0)  # (B, L)
        attention_mask = torch.ones_like(waveforms, dtype=torch.long)
        x_trad = torch.stack([b["x_trad"] for b in batch], dim=0)
        y_emo  = torch.stack([b["y_emo"] for b in batch], dim=0)
        y_spk  = torch.stack([b["y_spk"] for b in batch], dim=0)
        return waveforms, attention_mask, x_trad, y_emo, y_spk


# =========================
# Hidden SpecAug
# =========================
class HiddenSpecAug(nn.Module):
    def __init__(self, time_mask_prob=0.05, time_mask_len=10,
                 feat_mask_prob=0.05, feat_mask_len=64):
        super().__init__()
        self.time_mask_prob = time_mask_prob
        self.time_mask_len = time_mask_len
        self.feat_mask_prob = feat_mask_prob
        self.feat_mask_len = feat_mask_len

    def forward(self, x):
        if (not self.training) or (self.time_mask_prob <= 0 and self.feat_mask_prob <= 0):
            return x
        B, T, C = x.shape

        if self.time_mask_prob > 0:
            num_masks = max(1, int(T * self.time_mask_prob / max(1, self.time_mask_len)))
            for b in range(B):
                for _ in range(num_masks):
                    t0 = random.randint(0, max(0, T - self.time_mask_len))
                    x[b, t0:t0 + self.time_mask_len, :] = 0.0

        if self.feat_mask_prob > 0:
            num_masks = max(1, int(C * self.feat_mask_prob / max(1, self.feat_mask_len)))
            for b in range(B):
                for _ in range(num_masks):
                    c0 = random.randint(0, max(0, C - self.feat_mask_len))
                    x[b, :, c0:c0 + self.feat_mask_len] = 0.0

        return x


# =========================
# Extra Transformer Encoder
# =========================
class ExtraTransformerEncoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, key_padding_mask=None):
        return self.enc(x, src_key_padding_mask=key_padding_mask)


# =========================
# Config
# =========================
@dataclass
class ExpConfig:
    name: str
    fusion: str = "gate"          # "gate" | "concat"
    use_extra_trf: bool = True
    extra_trf_layers: int = 1
    use_hidden_specaug: bool = False
    use_sat: bool = False
    unfreeze_last_n: int = 2      # wav2vec2 last N transformer layers trainable


# =========================
# Model (fusion switchable)
# =========================
class E2EModel(nn.Module):
    def __init__(self, num_emotions: int, num_speakers: int, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_DIR, local_files_only=True)
        self.wav2vec2.freeze_feature_encoder()

        # freeze all by default
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

        # unfreeze last N transformer layers (wav2vec2-base has 12 layers)
        lastN = int(cfg.unfreeze_last_n)
        if lastN > 0:
            last_layers = list(range(12 - lastN, 12))
            for name, param in self.wav2vec2.named_parameters():
                if any(f"encoder.layers.{i}." in name for i in last_layers):
                    param.requires_grad = True

        self.specaug = HiddenSpecAug() if cfg.use_hidden_specaug else None
        self.extra_trf = ExtraTransformerEncoder(num_layers=cfg.extra_trf_layers) if cfg.use_extra_trf else None

        # deep temporal modeling
        self.lstm = nn.LSTM(
            input_size=768, hidden_size=64,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.temporal_attn = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

        # trad stream
        self.trad_stream = nn.Sequential(
            nn.Linear(45, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # gate params
        self.gate_trad = nn.Linear(128, 128)
        self.gate_deep = nn.Linear(128, 128)

        head_in = 256 if cfg.fusion == "concat" else 128

        self.emotion_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions),
        )
        self.speaker_head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_speakers),
        )

    def forward(self, input_values, attention_mask, x_trad, alpha: float):
        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hs = out.last_hidden_state  # (B,T,768)

        if self.specaug is not None:
            hs = self.specaug(hs)

        if self.extra_trf is not None:
            B, T, _ = hs.shape
            key_padding_mask = torch.zeros((B, T), dtype=torch.bool, device=hs.device)
            hs = self.extra_trf(hs, key_padding_mask=key_padding_mask)

        lstm_out, _ = self.lstm(hs)  # (B,T,128)
        attn = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (B,T,1)
        h_deep = torch.sum(lstm_out * attn, dim=1)  # (B,128)

        h_trad = self.trad_stream(x_trad)  # (B,128)

        if self.cfg.fusion == "gate":
            z = torch.sigmoid(self.gate_trad(h_trad) + self.gate_deep(h_deep))
            h_fusion = z * h_trad + (1 - z) * h_deep  # (B,128)
        elif self.cfg.fusion == "concat":
            h_fusion = torch.cat([h_deep, h_trad], dim=1)  # (B,256)
        else:
            raise ValueError(f"Unknown fusion: {self.cfg.fusion}")

        emo_logits = self.emotion_head(h_fusion)
        spk_logits = self.speaker_head(grad_reverse(h_fusion, alpha))
        return emo_logits, spk_logits


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    s_true, s_pred = [], []

    for wavs, attn, x_trad, y_emo, y_spk in loader:
        wavs = wavs.to(device)
        attn = attn.to(device)
        x_trad = x_trad.to(device)

        emo_logits, spk_logits = model(wavs, attn, x_trad, alpha=0.0)

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

def train_one_epoch(model, loader, optimizer, device,
                    ce_emo, ce_spk,
                    alpha, lam,
                    accum_steps=1, max_grad_norm=1.0,
                    use_amp=False, scaler: Optional[torch.cuda.amp.GradScaler] = None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    emo_sum, spk_sum, n = 0.0, 0.0, 0

    for step, (wavs, attn, x_trad, y_emo, y_spk) in enumerate(loader):
        wavs = wavs.to(device)
        attn = attn.to(device)
        x_trad = x_trad.to(device)
        y_emo = y_emo.to(device)
        y_spk = y_spk.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            emo_logits, spk_logits = model(wavs, attn, x_trad, alpha=alpha)
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


def run_one_exp(cfg: ExpConfig,
                file_paths_train, file_paths_test,
                X_trad_train, X_trad_test,
                y_emo_train, y_emo_test,
                y_spk_train, y_spk_test,
                emo_enc, spk_enc,
                device, out_dir: str) -> Dict[str, Any]:

    print(f"\n====================\n▶ {cfg.name}\n====================")
    train_ds = AudioTradDataset(file_paths_train, X_trad_train, y_emo_train, y_spk_train, TARGET_SR, MAX_SEC)
    test_ds  = AudioTradDataset(file_paths_test,  X_trad_test,  y_emo_test,  y_spk_test,  TARGET_SR, MAX_SEC)

    collate = CollatorSimple()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

    model = E2EModel(num_emotions=len(emo_enc.classes_), num_speakers=len(spk_enc.classes_), cfg=cfg).to(device)
    print(f"Trainable params: {count_trainable_params(model)/1e6:.2f}M | fusion={cfg.fusion} | extraTRF={cfg.use_extra_trf} | specaug={cfg.use_hidden_specaug} | SAT={cfg.use_sat}")

    ce_emo = nn.CrossEntropyLoss()
    ce_spk = nn.CrossEntropyLoss()

    w2v_params = [p for p in model.wav2vec2.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "wav2vec2" not in n]

    param_groups = []
    if len(w2v_params) > 0:
        param_groups.append({"params": w2v_params, "lr": LR_W2V})
    param_groups.append({"params": head_params, "lr": LR_HEAD})

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    base_lr_w2v = LR_W2V
    base_lr_head = LR_HEAD

    use_amp = torch.cuda.is_available()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"uar": -1, "acc": -1, "mf1": -1, "spk_acc": -1, "spk_uar": -1, "epoch": -1}
    ckpt_path = os.path.join(out_dir, f"{cfg.name}_best.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        fac = cosine_lr_factor(epoch, NUM_EPOCHS, WARMUP_EPOCHS)
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]["lr"] = base_lr_w2v * fac
            optimizer.param_groups[1]["lr"] = base_lr_head * fac
        else:
            optimizer.param_groups[0]["lr"] = base_lr_head * fac

        if (not cfg.use_sat) or (epoch <= SAT_WARMUP_EPOCHS):
            alpha, lam = 0.0, 0.0
        else:
            t = (epoch - SAT_WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - SAT_WARMUP_EPOCHS))
            t = min(1.0, max(0.0, float(t)))
            alpha = SAT_ALPHA_MAX * t
            lam = SAT_LAMBDA_MAX * t

        tr_emo, tr_spk = train_one_epoch(
            model, train_loader, optimizer, device,
            ce_emo, ce_spk,
            alpha=alpha, lam=lam,
            accum_steps=ACCUM_STEPS, max_grad_norm=MAX_GRAD_NORM,
            use_amp=use_amp, scaler=amp_scaler
        )

        acc, uar, mf1, spk_acc, spk_uar = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | TrainEmo={tr_emo:.4f} TrainSpk={tr_spk:.4f} | "
              f"Val Acc={acc:.2f} UAR={uar:.2f} MF1={mf1:.2f} | SpkAcc={spk_acc:.2f} SpkUAR={spk_uar:.2f} | alpha={alpha:.3f} lam={lam:.4f}")

        if uar > best["uar"]:
            best = {"uar": uar, "acc": acc, "mf1": mf1, "spk_acc": spk_acc, "spk_uar": spk_uar, "epoch": epoch}
            torch.save({"model": model.state_dict(), "best": best, "cfg": cfg.__dict__}, ckpt_path)
            print(f"  ✅ Saved best: {ckpt_path} | best UAR={best['uar']:.2f} (epoch {best['epoch']})")

    return {
        "exp": cfg.name,
        "best_epoch": best["epoch"],
        "acc": best["acc"],
        "uar": best["uar"],
        "macro_f1": best["mf1"],
        "spk_acc": best["spk_acc"],
        "spk_uar": best["spk_uar"],
        "ckpt": ckpt_path
    }


def run_seed(seed: int, out_root: str):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Seed={seed} | Device: {device}")

    seed_dir = os.path.join(out_root, f"seed{seed}")
    ensure_dir(seed_dir)

    wav_paths = np.load(FILE_PATHS_NPY, allow_pickle=True)
    X_trad = np.load(X_TRAD_PATH)
    y_emo_labels = np.load(Y_EMO_PATH)
    speaker_ids_raw = np.load(SPK_ID_PATH, allow_pickle=True)

    assert len(wav_paths) == len(X_trad) == len(y_emo_labels) == len(speaker_ids_raw), \
        f"Length mismatch: wav={len(wav_paths)}, X_trad={len(X_trad)}, y={len(y_emo_labels)}, spk={len(speaker_ids_raw)}"

    emo_enc = LabelEncoder()
    y_emo = emo_enc.fit_transform(y_emo_labels)

    spk_enc = LabelEncoder()
    y_spk = spk_enc.fit_transform(speaker_ids_raw)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_trad, y_emo, groups=speaker_ids_raw))

    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train speakers: {len(np.unique(speaker_ids_raw[train_idx]))} | Test speakers: {len(np.unique(speaker_ids_raw[test_idx]))}")

    scaler = StandardScaler()
    X_trad_train = scaler.fit_transform(X_trad[train_idx])
    X_trad_test  = scaler.transform(X_trad[test_idx])

    file_paths_train = wav_paths[train_idx]
    file_paths_test  = wav_paths[test_idx]

    y_emo_train = y_emo[train_idx]
    y_emo_test  = y_emo[test_idx]
    y_spk_train = y_spk[train_idx]
    y_spk_test  = y_spk[test_idx]

    # =========================
    # Experiments (clear naming)
    # =========================
    EXPS: List[ExpConfig] = [
        # Base
        ExpConfig(name="A_base", fusion="gate", use_extra_trf=False, use_hidden_specaug=False, use_sat=False, unfreeze_last_n=2),

        # Core C: extraTRF + SpecAug + gate
        ExpConfig(name="C_gate", fusion="gate", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),

        # E: C + SAT (noCB)
        ExpConfig(name="E_gate_SAT_noCB", fusion="gate", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=True, unfreeze_last_n=2),

        # Ablation: remove extraTRF (keep SpecAug + gate)
        ExpConfig(name="C_gate_noTRF", fusion="gate", use_extra_trf=False, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),

        # Fusion ablation at C-level: concat
        ExpConfig(name="C_concat", fusion="concat", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),

        # SAT under concat
        ExpConfig(name="E_concat_SAT_noCB", fusion="concat", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=True, unfreeze_last_n=2),

        # extraTRF ablation under concat
        ExpConfig(name="C_concat_noTRF", fusion="concat", use_extra_trf=False, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),
    ]

    results: List[Dict[str, Any]] = []
    for cfg in EXPS:
        r = run_one_exp(
            cfg=cfg,
            file_paths_train=file_paths_train,
            file_paths_test=file_paths_test,
            X_trad_train=X_trad_train,
            X_trad_test=X_trad_test,
            y_emo_train=y_emo_train,
            y_emo_test=y_emo_test,
            y_spk_train=y_spk_train,
            y_spk_test=y_spk_test,
            emo_enc=emo_enc,
            spk_enc=spk_enc,
            device=device,
            out_dir=seed_dir
        )
        results.append(r)

    csv_path = os.path.join(seed_dir, "ablation_chain_summary_with_spk_uar.csv")
    with open(csv_path, "w") as f:
        f.write("exp,best_epoch,acc,uar,macro_f1,spk_acc,spk_uar,ckpt\n")
        for r in results:
            f.write(f"{r['exp']},{r['best_epoch']},{r['acc']:.4f},{r['uar']:.4f},{r['macro_f1']:.4f},"
                    f"{r['spk_acc']:.4f},{r['spk_uar']:.4f},{r['ckpt']}\n")

    print("\n====================")
    print(f"✅ Seed {seed} done. Saved CSV: {csv_path}")
    print("====================\n")
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Output root, e.g. /root/autodl-tmp/cremad_main_v2_ext_clear")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
    args = ap.parse_args()

    ensure_dir(args.out_root)
    for s in args.seeds:
        run_seed(s, args.out_root)

    print("\n✅ All done.")

if __name__ == "__main__":
    main()