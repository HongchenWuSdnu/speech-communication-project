#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cremad_baselines_BC.py

CREMA-D (6-class) speaker-independent baselines + our fusion variants.
This script runs (by default) 3 seeds: 13/42/2026 on the SAME split protocol:
  GroupShuffleSplit(test_size=0.2, random_state=42), groups = raw speaker_ids.

It produces, for each seed, a CSV:
  <out_root>/seed<SEED>/ablation_chain_summary_with_spk_uar.csv

Then (optional) it can aggregate across seeds using your existing:
  ttest_emotion_and_speaker_by_seed.py

What it runs (you can toggle in EXPS list):
(B) Wav2Vec2-only baseline:
  - deep_only_C: extraTRF(1) + hidden SpecAug (same as your C), but NO trad branch in fusion
  - deep_only_C+SAT: same + SAT (no CB)

(C) Simple fusion baseline (Concat / late fusion):
  - concat_C: extraTRF(1) + hidden SpecAug + concat([deep, trad])
  - concat_C+SAT: same + SAT (no CB)

(Ours) gated fusion:
  - gate_C: your C_plus_specaug (gated fusion)
  - gate_C+SAT: your C_plus_SAT_noCB (gated fusion + SAT noCB)

Optional (A) Trad-only baseline:
  - trad_only_MLP: 45-d trad features only (MLP)

NOTE:
- This script reuses your 45-d trad features (SCI_X_trad.npy) and CREMA-D wav folder.
- All models output: Emotion(Acc/UAR/MF1) + Speaker(SpkAcc/SpkUAR)

Usage:
  python /root/autodl-tmp/run_cremad_baselines_BC.py \
    --out_root /root/autodl-tmp/cremad_baselines_BC \
    --seeds 13 42 2026

Optional: run t-test aggregation at end:
  --run_ttest --ttest_script /root/autodl-tmp/ttest_emotion_and_speaker_by_seed.py
"""

import os
import math
import random
import argparse
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

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
# Fixed paths (same as your CREMA-D setup)
# =========================
MODEL_DIR = "/root/autodl-tmp/wav2vec2-base-local"

DATA_WAV_DIR = "/root/autodl-tmp/AudioWAV"
X_TRAD_PATH  = "/root/autodl-tmp/SCI_X_trad.npy"
Y_EMO_PATH   = "/root/autodl-tmp/SCI_y_labels.npy"
SPK_ID_PATH  = "/root/autodl-tmp/SCI_speaker_ids.npy"

# =========================
# Fixed split protocol
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

# SAT schedule (weak ramp; same as your noCB SAT setup)
SAT_WARMUP_EPOCHS = 8
SAT_ALPHA_MAX     = 0.02
SAT_LAMBDA_MAX    = 0.003

# Scheduler
WARMUP_EPOCHS = 5


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

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_lr_factor(epoch_idx_1based: int, total_epochs: int, warmup_epochs: int) -> float:
    e = epoch_idx_1based
    warmup_epochs = max(1, int(warmup_epochs))
    if e <= warmup_epochs:
        return e / warmup_epochs
    t = (e - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * t))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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
        # use soundfile to avoid torchaudio torchcodec issue
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
            "waveform": wav,              # (L,)
            "x_trad": self.x_trad[idx],   # (45,)
            "y_emo": self.y_emo[idx],
            "y_spk": self.y_spk[idx],
        }

class CollatorSimple:
    def __call__(self, batch):
        waveforms = torch.stack([b["waveform"] for b in batch], dim=0)  # (B, L)
        attention_mask = torch.ones_like(waveforms, dtype=torch.long)   # (B, L)
        x_trad = torch.stack([b["x_trad"] for b in batch], dim=0)       # (B, 45)
        y_emo  = torch.stack([b["y_emo"] for b in batch], dim=0)        # (B,)
        y_spk  = torch.stack([b["y_spk"] for b in batch], dim=0)        # (B,)
        return waveforms, attention_mask, x_trad, y_emo, y_spk


# =========================
# Components: Hidden SpecAug
# =========================
class HiddenSpecAug(nn.Module):
    """
    SpecAugment-like masking on wav2vec2 hidden states: (B, T, C)
    """
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

        # time masking
        if self.time_mask_prob > 0:
            num_masks = max(1, int(T * self.time_mask_prob / max(1, self.time_mask_len)))
            for b in range(B):
                for _ in range(num_masks):
                    t0 = random.randint(0, max(0, T - self.time_mask_len))
                    x[b, t0:t0 + self.time_mask_len, :] = 0.0

        # feature masking
        if self.feat_mask_prob > 0:
            num_masks = max(1, int(C * self.feat_mask_prob / max(1, self.feat_mask_len)))
            for b in range(B):
                for _ in range(num_masks):
                    c0 = random.randint(0, max(0, C - self.feat_mask_len))
                    x[b, :, c0:c0 + self.feat_mask_len] = 0.0

        return x


# =========================
# Extra Transformer Encoder block
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
# Model configs
# =========================
@dataclass
class ExpConfig:
    name: str
    unfreeze_last_n: int = 2
    use_extra_trf: bool = True
    extra_trf_layers: int = 1
    use_hidden_specaug: bool = True
    use_sat: bool = False

    # baseline switches
    fusion: str = "gate"     # "gate" | "concat" | "deep_only"
    use_trad: bool = True    # deep_only will set False internally

# =========================
# E2E Model with switchable fusion
# =========================
class E2EModel(nn.Module):
    def __init__(self, num_emotions: int, num_speakers: int, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_DIR, local_files_only=True)
        self.wav2vec2.freeze_feature_encoder()

        # freeze all
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

        # unfreeze last N transformer layers (12 layers for wav2vec2-base)
        lastN = int(cfg.unfreeze_last_n)
        if lastN > 0:
            last_layers = list(range(12 - lastN, 12))
            for name, param in self.wav2vec2.named_parameters():
                if any(f"encoder.layers.{i}." in name for i in last_layers):
                    param.requires_grad = True

        self.specaug = HiddenSpecAug() if cfg.use_hidden_specaug else None
        self.extra_trf = ExtraTransformerEncoder(num_layers=cfg.extra_trf_layers) if cfg.use_extra_trf else None

        # Traditional stream
        self.trad_stream = nn.Sequential(
            nn.Linear(45, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=768, hidden_size=64,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        # Gated fusion params (used if fusion=="gate")
        self.gate_trad = nn.Linear(128, 128)
        self.gate_deep = nn.Linear(128, 128)

        # Heads input dim depends on fusion
        if cfg.fusion == "concat":
            head_in = 256
        else:
            head_in = 128  # gate or deep_only

        self.emotion_classifier = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions),
        )

        self.speaker_classifier = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_speakers),
        )

    def forward(self, input_values, attention_mask, x_trad, alpha: float):
        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hs = out.last_hidden_state  # (B, T, 768)

        if self.specaug is not None:
            hs = self.specaug(hs)

        if self.extra_trf is not None:
            # fixed-length -> no padding at hidden level
            B, T, _ = hs.shape
            key_padding_mask = torch.zeros((B, T), dtype=torch.bool, device=hs.device)
            hs = self.extra_trf(hs, key_padding_mask=key_padding_mask)

        lstm_out, _ = self.lstm(hs)  # (B, T, 128)
        attn = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (B, T, 1)
        h_deep = torch.sum(lstm_out * attn, dim=1)  # (B, 128)

        # build fusion representation
        if self.cfg.fusion == "deep_only":
            h_fusion = h_deep

        else:
            # we will still compute trad branch unless you explicitly disable
            if self.cfg.use_trad:
                h_trad = self.trad_stream(x_trad)  # (B, 128)
            else:
                h_trad = torch.zeros_like(h_deep)

            if self.cfg.fusion == "gate":
                z = torch.sigmoid(self.gate_trad(h_trad) + self.gate_deep(h_deep))
                h_fusion = z * h_trad + (1 - z) * h_deep
            elif self.cfg.fusion == "concat":
                h_fusion = torch.cat([h_deep, h_trad], dim=1)  # (B, 256)
            else:
                raise ValueError(f"Unknown fusion type: {self.cfg.fusion}")

        emo_logits = self.emotion_classifier(h_fusion)
        spk_logits = self.speaker_classifier(grad_reverse(h_fusion, alpha))
        return emo_logits, spk_logits


# =========================
# Trad-only baseline (MLP)
# =========================
class TradOnlyMLP(nn.Module):
    def __init__(self, num_emotions: int, num_speakers: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(45, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_emotions),
        )
        self.speaker_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_speakers),
        )

    def forward(self, x_trad, alpha: float):
        h = self.backbone(x_trad)
        emo_logits = self.emotion_head(h)
        spk_logits = self.speaker_head(grad_reverse(h, alpha))
        return emo_logits, spk_logits


# =========================
# Loss helpers
# =========================
def make_class_balanced_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    N = counts.sum()
    K = float(num_classes)
    w = N / (K * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# =========================
# Train/Eval
# =========================
@torch.no_grad()
def evaluate_e2e(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    spk_true, spk_pred = [], []

    for input_values, attn_mask, x_trad, y_emo, y_spk in loader:
        input_values = input_values.to(device)
        attn_mask = attn_mask.to(device)
        x_trad = x_trad.to(device)

        emo_logits, spk_logits = model(input_values, attn_mask, x_trad, alpha=0.0)

        y_hat = emo_logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y_emo.numpy()); y_pred.extend(y_hat)

        spk_hat = spk_logits.argmax(dim=1).cpu().numpy()
        spk_true.extend(y_spk.numpy()); spk_pred.extend(spk_hat)

    acc = accuracy_score(y_true, y_pred) * 100
    uar = recall_score(y_true, y_pred, average="macro") * 100
    mf1 = f1_score(y_true, y_pred, average="macro") * 100
    spk_acc = accuracy_score(spk_true, spk_pred) * 100
    spk_uar = recall_score(spk_true, spk_pred, average="macro") * 100
    return acc, uar, mf1, spk_acc, spk_uar

@torch.no_grad()
def evaluate_trad(model, loader_trad, device):
    model.eval()
    y_true, y_pred = [], []
    spk_true, spk_pred = [], []
    for x_trad, y_emo, y_spk in loader_trad:
        x_trad = x_trad.to(device)
        emo_logits, spk_logits = model(x_trad, alpha=0.0)

        y_hat = emo_logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y_emo.cpu().numpy()); y_pred.extend(y_hat)

        spk_hat = spk_logits.argmax(dim=1).cpu().numpy()
        spk_true.extend(y_spk.cpu().numpy()); spk_pred.extend(spk_hat)

    acc = accuracy_score(y_true, y_pred) * 100
    uar = recall_score(y_true, y_pred, average="macro") * 100
    mf1 = f1_score(y_true, y_pred, average="macro") * 100
    spk_acc = accuracy_score(spk_true, spk_pred) * 100
    spk_uar = recall_score(spk_true, spk_pred, average="macro") * 100
    return acc, uar, mf1, spk_acc, spk_uar

def train_one_epoch_e2e(model, loader, optimizer, device,
                        criterion_emo, criterion_spk,
                        alpha, lambda_spk,
                        accum_steps=1, max_grad_norm=1.0,
                        scaler: Optional[torch.cuda.amp.GradScaler] = None,
                        use_amp: bool = False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    emo_loss_sum, spk_loss_sum, n = 0.0, 0.0, 0

    for step, (input_values, attn_mask, x_trad, y_emo, y_spk) in enumerate(loader):
        input_values = input_values.to(device)
        attn_mask = attn_mask.to(device)
        x_trad = x_trad.to(device)
        y_emo = y_emo.to(device)
        y_spk = y_spk.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            emo_logits, spk_logits = model(input_values, attn_mask, x_trad, alpha=alpha)
            loss_emo = criterion_emo(emo_logits, y_emo)
            loss_spk = criterion_spk(spk_logits, y_spk)
            loss = (loss_emo + lambda_spk * loss_spk) / accum_steps

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

        emo_loss_sum += float(loss_emo.item())
        spk_loss_sum += float(loss_spk.item())
        n += 1

    return emo_loss_sum / max(n, 1), spk_loss_sum / max(n, 1)

def train_one_epoch_trad(model, loader_trad, optimizer, device,
                         criterion_emo, criterion_spk,
                         alpha, lambda_spk,
                         accum_steps=1, max_grad_norm=1.0,
                         scaler: Optional[torch.cuda.amp.GradScaler] = None,
                         use_amp: bool = False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    emo_loss_sum, spk_loss_sum, n = 0.0, 0.0, 0

    for step, (x_trad, y_emo, y_spk) in enumerate(loader_trad):
        x_trad = x_trad.to(device)
        y_emo = y_emo.to(device)
        y_spk = y_spk.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            emo_logits, spk_logits = model(x_trad, alpha=alpha)
            loss_emo = criterion_emo(emo_logits, y_emo)
            loss_spk = criterion_spk(spk_logits, y_spk)
            loss = (loss_emo + lambda_spk * loss_spk) / accum_steps

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

        emo_loss_sum += float(loss_emo.item())
        spk_loss_sum += float(loss_spk.item())
        n += 1

    return emo_loss_sum / max(n, 1), spk_loss_sum / max(n, 1)


# =========================
# Trad-only DataLoader
# =========================
class TradOnlyDataset(Dataset):
    def __init__(self, X_trad, y_emo, y_spk):
        self.X = torch.tensor(X_trad, dtype=torch.float32)
        self.y_emo = torch.tensor(y_emo, dtype=torch.long)
        self.y_spk = torch.tensor(y_spk, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_emo[idx], self.y_spk[idx]


# =========================
# Experiment runner
# =========================
def run_one_experiment_e2e(cfg: ExpConfig,
                           file_paths_train, file_paths_test,
                           X_trad_train, X_trad_test,
                           y_emo_train, y_emo_test,
                           y_spk_train, y_spk_test,
                           emo_enc, spk_enc, scaler_feat: StandardScaler,
                           device,
                           out_dir: str) -> Dict[str, Any]:

    print(f"\n====================\n▶ {cfg.name}\n====================")
    train_ds = AudioTradDataset(file_paths_train, X_trad_train, y_emo_train, y_spk_train,
                                target_sr=TARGET_SR, max_sec=MAX_SEC)
    test_ds  = AudioTradDataset(file_paths_test,  X_trad_test,  y_emo_test,  y_spk_test,
                                target_sr=TARGET_SR, max_sec=MAX_SEC)

    collate = CollatorSimple()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

    # force deep_only to not use trad
    if cfg.fusion == "deep_only":
        cfg.use_trad = False

    model = E2EModel(num_emotions=len(emo_enc.classes_),
                     num_speakers=len(spk_enc.classes_),
                     cfg=cfg).to(device)

    print(f"Trainable params: {count_trainable_params(model)/1e6:.2f}M | "
          f"fusion={cfg.fusion} | specaug={cfg.use_hidden_specaug} | extraTRF={cfg.use_extra_trf} | SAT={cfg.use_sat}")

    criterion_emo = nn.CrossEntropyLoss()
    criterion_spk = nn.CrossEntropyLoss()

    wav2vec2_params = [p for p in model.wav2vec2.parameters() if p.requires_grad]

    head_params: List[nn.Parameter] = []
    # add all non-wav2vec2 params
    for name, p in model.named_parameters():
        if "wav2vec2" not in name:
            head_params.append(p)

    param_groups = []
    if len(wav2vec2_params) > 0:
        param_groups.append({"params": wav2vec2_params, "lr": LR_W2V})
    param_groups.append({"params": head_params, "lr": LR_HEAD})
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    base_lr_w2v = LR_W2V
    base_lr_head = LR_HEAD

    use_amp = torch.cuda.is_available()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"uar": -1, "acc": -1, "mf1": -1, "spk_acc": -1, "spk_uar": -1, "epoch": -1}
    ckpt_path = os.path.join(out_dir, f"{cfg.name}_best.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        fac = cosine_lr_factor(epoch, NUM_EPOCHS, WARMUP_EPOCHS)
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]["lr"] = base_lr_w2v * fac
            optimizer.param_groups[1]["lr"] = base_lr_head * fac
            lr_show = optimizer.param_groups[0]["lr"]
        else:
            optimizer.param_groups[0]["lr"] = base_lr_head * fac
            lr_show = optimizer.param_groups[0]["lr"]

        # SAT schedule
        if (not cfg.use_sat) or (epoch <= SAT_WARMUP_EPOCHS):
            alpha = 0.0
            lam = 0.0
        else:
            t = (epoch - SAT_WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - SAT_WARMUP_EPOCHS))
            t = min(1.0, max(0.0, float(t)))
            alpha = SAT_ALPHA_MAX * t
            lam = SAT_LAMBDA_MAX * t

        tr_emo, tr_spk = train_one_epoch_e2e(
            model, train_loader, optimizer, device,
            criterion_emo, criterion_spk,
            alpha=alpha, lambda_spk=lam,
            accum_steps=ACCUM_STEPS, max_grad_norm=MAX_GRAD_NORM,
            scaler=amp_scaler, use_amp=use_amp
        )

        acc, uar, mf1, spk_acc, spk_uar = evaluate_e2e(model, test_loader, device)
        print(f"Epoch {epoch:02d} | lr={lr_show:.2e} | "
              f"Train Emo={tr_emo:.4f} Spk={tr_spk:.4f} | "
              f"Val Acc={acc:.2f}% UAR={uar:.2f}% MF1={mf1:.2f}% | "
              f"SpkAcc={spk_acc:.2f}% SpkUAR={spk_uar:.2f}% | "
              f"alpha={alpha:.3f} lam={lam:.4f}")

        if uar > best["uar"]:
            best = {"uar": uar, "acc": acc, "mf1": mf1, "spk_acc": spk_acc, "spk_uar": spk_uar, "epoch": epoch}
            torch.save({
                "model": model.state_dict(),
                "scaler_mean": scaler_feat.mean_,
                "scaler_scale": scaler_feat.scale_,
                "emo_classes": emo_enc.classes_,
                "spk_classes": spk_enc.classes_,
                "exp_cfg": cfg.__dict__,
                "best": best,
                "seed": int(os.environ.get("RUN_SEED", "0")),
                "split_random_state": SPLIT_RANDOM_STATE,
            }, ckpt_path)
            print(f"  ✅ Saved best: {ckpt_path} | best UAR={best['uar']:.2f}% (epoch {best['epoch']})")

    print(f"✅ Finished {cfg.name} | Best={best}")
    return {"exp": cfg.name, "best_epoch": best["epoch"], "acc": best["acc"], "uar": best["uar"], "macro_f1": best["mf1"],
            "spk_acc": best["spk_acc"], "spk_uar": best["spk_uar"], "ckpt": ckpt_path}

def run_one_experiment_trad_only(name: str,
                                X_trad_train, X_trad_test,
                                y_emo_train, y_emo_test,
                                y_spk_train, y_spk_test,
                                emo_enc, spk_enc,
                                device,
                                out_dir: str) -> Dict[str, Any]:

    print(f"\n====================\n▶ {name}\n====================")
    train_ds = TradOnlyDataset(X_trad_train, y_emo_train, y_spk_train)
    test_ds  = TradOnlyDataset(X_trad_test,  y_emo_test,  y_spk_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TradOnlyMLP(num_emotions=len(emo_enc.classes_), num_speakers=len(spk_enc.classes_)).to(device)
    print(f"Trainable params: {count_trainable_params(model)/1e6:.2f}M | Trad-only MLP")

    criterion_emo = nn.CrossEntropyLoss()
    criterion_spk = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = torch.cuda.is_available()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"uar": -1, "acc": -1, "mf1": -1, "spk_acc": -1, "spk_uar": -1, "epoch": -1}
    ckpt_path = os.path.join(out_dir, f"{name}_best.pth")

    for epoch in range(1, 61):  # small MLP can train longer quickly
        # no SAT for trad-only baseline
        alpha = 0.0
        lam = 0.0

        tr_emo, tr_spk = train_one_epoch_trad(
            model, train_loader, optimizer, device,
            criterion_emo, criterion_spk,
            alpha=alpha, lambda_spk=lam,
            accum_steps=1, max_grad_norm=5.0,
            scaler=amp_scaler, use_amp=use_amp
        )
        acc, uar, mf1, spk_acc, spk_uar = evaluate_trad(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train Emo={tr_emo:.4f} Spk={tr_spk:.4f} | "
              f"Val Acc={acc:.2f}% UAR={uar:.2f}% MF1={mf1:.2f}% | SpkAcc={spk_acc:.2f}% SpkUAR={spk_uar:.2f}%")

        if uar > best["uar"]:
            best = {"uar": uar, "acc": acc, "mf1": mf1, "spk_acc": spk_acc, "spk_uar": spk_uar, "epoch": epoch}
            torch.save({"model": model.state_dict(), "best": best}, ckpt_path)
            print(f"  ✅ Saved best: {ckpt_path} | best UAR={best['uar']:.2f}% (epoch {best['epoch']})")

    print(f"✅ Finished {name} | Best={best}")
    return {"exp": name, "best_epoch": best["epoch"], "acc": best["acc"], "uar": best["uar"], "macro_f1": best["mf1"],
            "spk_acc": best["spk_acc"], "spk_uar": best["spk_uar"], "ckpt": ckpt_path}


# =========================
# Main runner per seed
# =========================
def run_seed(seed: int, out_root: str, include_trad_only: bool):
    os.environ["RUN_SEED"] = str(seed)
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Seed={seed} | Device: {device}")

    seed_dir = os.path.join(out_root, f"seed{seed}")
    ensure_dir(seed_dir)

    # load CREMA-D data
    all_files = sorted([f for f in os.listdir(DATA_WAV_DIR) if f.endswith(".wav")])
    wav_paths = np.array([os.path.join(DATA_WAV_DIR, f) for f in all_files])

    X_trad = np.load(X_TRAD_PATH)
    y_emo_labels = np.load(Y_EMO_PATH)
    speaker_ids_raw  = np.load(SPK_ID_PATH)

    assert len(wav_paths) == len(X_trad) == len(y_emo_labels) == len(speaker_ids_raw), \
        f"Length mismatch: wav={len(wav_paths)}, X_trad={len(X_trad)}, y={len(y_emo_labels)}, spk={len(speaker_ids_raw)}"

    emo_enc = LabelEncoder()
    y_emo = emo_enc.fit_transform(y_emo_labels)

    spk_enc = LabelEncoder()
    y_spk = spk_enc.fit_transform(speaker_ids_raw)

    # fixed speaker-independent split (same across seeds)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_trad, y_emo, groups=speaker_ids_raw))

    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train speakers: {len(np.unique(speaker_ids_raw[train_idx]))} | Test speakers: {len(np.unique(speaker_ids_raw[test_idx]))}")

    # standardize trad features (fit on train)
    scaler_feat = StandardScaler()
    X_trad_train = scaler_feat.fit_transform(X_trad[train_idx])
    X_trad_test  = scaler_feat.transform(X_trad[test_idx])

    file_paths_train = wav_paths[train_idx]
    file_paths_test  = wav_paths[test_idx]

    y_emo_train = y_emo[train_idx]
    y_emo_test  = y_emo[test_idx]
    y_spk_train = y_spk[train_idx]
    y_spk_test  = y_spk[test_idx]

    # -----------------------------
    # Define experiments (B, C, Ours)
    # -----------------------------
    EXPS: List[ExpConfig] = [
        # (B) Wav2Vec2-only baseline (C-like training)
        ExpConfig(
            name="B_deep_only_C",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_sat=False,
            fusion="deep_only",
            use_trad=False
        ),
        ExpConfig(
            name="B_deep_only_C_SAT",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_sat=True,
            fusion="deep_only",
            use_trad=False
        ),

        # (C) Simple concat fusion baseline
        ExpConfig(
            name="C_concat_C",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_sat=False,
            fusion="concat",
            use_trad=True
        ),
        ExpConfig(
            name="C_concat_C_SAT",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_sat=True,
            fusion="concat",
            use_trad=True
        ),

        # (Ours) gated fusion
        ExpConfig(
            name="O_gate_C",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_sat=False,
            fusion="gate",
            use_trad=True
        ),
        ExpConfig(
            name="O_gate_C_SAT",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_sat=True,
            fusion="gate",
            use_trad=True
        ),
    ]

    results: List[Dict[str, Any]] = []

    # optional Trad-only baseline (A)
    if include_trad_only:
        r = run_one_experiment_trad_only(
            name="A_trad_only_MLP",
            X_trad_train=X_trad_train, X_trad_test=X_trad_test,
            y_emo_train=y_emo_train, y_emo_test=y_emo_test,
            y_spk_train=y_spk_train, y_spk_test=y_spk_test,
            emo_enc=emo_enc, spk_enc=spk_enc,
            device=device,
            out_dir=seed_dir
        )
        results.append(r)

    # run e2e experiments
    for cfg in EXPS:
        r = run_one_experiment_e2e(
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
            scaler_feat=scaler_feat,
            device=device,
            out_dir=seed_dir
        )
        results.append(r)

    # write seed CSV
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
    ap.add_argument("--out_root", required=True, help="Output root directory, e.g. /root/autodl-tmp/cremad_baselines_BC")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026], help="Seeds to run (default: 13 42 2026)")
    ap.add_argument("--include_trad_only", action="store_true", help="Also run A_trad_only_MLP baseline (optional)")
    ap.add_argument("--run_ttest", action="store_true", help="Run aggregation t-test script after all seeds finish")
    ap.add_argument("--ttest_script", default="/root/autodl-tmp/ttest_emotion_and_speaker_by_seed.py", help="Path to t-test script")
    ap.add_argument("--digits", type=int, default=3)
    args = ap.parse_args()

    ensure_dir(args.out_root)

    csvs = []
    for s in args.seeds:
        csvs.append(run_seed(s, args.out_root, include_trad_only=args.include_trad_only))

    # optional aggregation using your t-test script
    if args.run_ttest:
        if not os.path.exists(args.ttest_script):
            print(f"⚠️ ttest_script not found: {args.ttest_script} (skip)")
            return
        cmd = ["python", args.ttest_script] + csvs
        # default comparisons in ttest script are A_base/C_plus_specaug/C_plus_SAT_noCB,
        # but here our exp names are different; so we only run summary (mean±std) and
        # you can pass custom --exp-a/--exp-c/--exp-sat later if you want.
        print("\n🧪 Running t-test aggregator (note: exp names differ; you may want custom flags)...")
        print(" ".join(cmd))
        subprocess.run(cmd, check=False)

    print("\n✅ All done.")
    print("Per-seed CSVs:")
    for c in csvs:
        print(" -", c)
    print("\nNext: aggregate across seeds with your t-test script using the exp names in these CSVs.")
    print("Example exp names include: B_deep_only_C, C_concat_C, O_gate_C, and their *_SAT variants.")


if __name__ == "__main__":
    main()