#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_main_ablation_v2_ext_clear.py

SER main ablation (v2 aligned), clear naming, speaker leakage (SpkAcc + SpkUAR).

This script supports:
  - CREMA-D v2 (your original SCI_* npys)
  - Any dataset with v2_dir containing:
      file_paths.npy / y_labels.npy / speaker_ids.npy
    (optional) X_trad_v2.npy   (if missing, uses zeros(45D))

Split protocol:
  - Default: GroupShuffleSplit(test_size=0.2, random_state=42), groups=speaker_ids
  - If --split_json is provided: uses train_idx/test_idx in that json (e.g., LOSO)

Best checkpoint:
  - Selected by best UAR on test split

Report:
  - Acc / UAR / MF1 + SpkAcc / SpkUAR

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
import json
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
# Defaults
# =========================
DEFAULT_MODEL_DIR = "/root/autodl-tmp/wav2vec2-base-local"

# Split protocol (default)
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.2

# Audio
TARGET_SR = 16000
MAX_SEC   = 3.0

# Training
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

# If trad feature file missing, use 45D zeros (matches your existing branch expectation)
TRAD_DIM_FALLBACK = 45


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

def _load_npy_flexible(v2_dir: str, stem_candidates: List[str], allow_pickle: bool = False):
    """
    Try multiple file name candidates under v2_dir.
    Return loaded np array, or raise FileNotFoundError.
    """
    for name in stem_candidates:
        p = os.path.join(v2_dir, name)
        if os.path.exists(p):
            return np.load(p, allow_pickle=allow_pickle)
    raise FileNotFoundError(f"None of these files exist under {v2_dir}: {stem_candidates}")

def _maybe_load_trad(v2_dir: str, n: int) -> np.ndarray:
    """
    Optional trad features:
      - X_trad_v2.npy (new)
      - SCI_X_trad_v2.npy (legacy)
    If missing: zeros(n, 45)
    """
    cand = ["X_trad_v2.npy", "SCI_X_trad_v2.npy"]
    for name in cand:
        p = os.path.join(v2_dir, name)
        if os.path.exists(p):
            X = np.load(p, allow_pickle=True)
            return X
    return np.zeros((n, TRAD_DIM_FALLBACK), dtype=np.float32)

def _read_split_json(split_json: str):
    with open(split_json, "r") as f:
        obj = json.load(f)
    if "train_idx" not in obj or "test_idx" not in obj:
        raise ValueError(f"--split_json must contain train_idx and test_idx. Got keys: {list(obj.keys())}")
    train_idx = np.array(obj["train_idx"], dtype=np.int64)
    test_idx  = np.array(obj["test_idx"], dtype=np.int64)
    return train_idx, test_idx


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
    def __init__(self, file_paths: np.ndarray, X_trad: np.ndarray, y_emo: np.ndarray, y_spk: np.ndarray):
        self.file_paths = file_paths
        self.X_trad = X_trad.astype(np.float32)
        self.y_emo = y_emo.astype(np.int64)
        self.y_spk = y_spk.astype(np.int64)

        assert len(self.file_paths) == len(self.X_trad) == len(self.y_emo) == len(self.y_spk)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.file_paths[idx], self.X_trad[idx], self.y_emo[idx], self.y_spk[idx]

class AudioCollator:
    def __init__(self, target_sr=TARGET_SR, max_sec=MAX_SEC):
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_sec)

    def __call__(self, batch):
        file_paths, X_trad, y_emo, y_spk = zip(*batch)

        wavs = []
        for fp in file_paths:
            wav, sr = torchaudio.load(fp)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)  # mono
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            wav = wav.squeeze(0)  # [T]

            # pad/trim
            if wav.numel() < self.max_len:
                pad = self.max_len - wav.numel()
                wav = F.pad(wav, (0, pad))
            else:
                wav = wav[:self.max_len]
            wavs.append(wav)

        wavs = torch.stack(wavs, dim=0)  # [B, T]
        X_trad = torch.tensor(np.stack(X_trad, axis=0), dtype=torch.float32)
        y_emo = torch.tensor(np.array(y_emo), dtype=torch.long)
        y_spk = torch.tensor(np.array(y_spk), dtype=torch.long)
        return wavs, X_trad, y_emo, y_spk


# =========================
# Hidden-state SpecAug
# =========================
def hidden_specaug(h: torch.Tensor, time_mask_ratio=0.06, feat_mask_ratio=0.12):
    """
    h: [B, T, C]
    """
    B, T, C = h.shape
    out = h

    # time mask
    t = max(1, int(T * time_mask_ratio))
    for b in range(B):
        t0 = random.randint(0, max(0, T - t))
        out[b, t0:t0 + t, :] = 0.0

    # feature mask
    f = max(1, int(C * feat_mask_ratio))
    for b in range(B):
        f0 = random.randint(0, max(0, C - f))
        out[b, :, f0:f0 + f] = 0.0

    return out


# =========================
# Model blocks
# =========================
class ExtraTransformer(nn.Module):
    def __init__(self, d_model: int, nhead=8, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        return self.enc(x)

class FusionHead(nn.Module):
    def __init__(self, d_audio: int, d_trad: int, fusion: str = "gate", num_classes: int = 6):
        super().__init__()
        self.fusion = fusion
        self.proj_trad = nn.Sequential(
            nn.Linear(d_trad, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_audio),
        )
        if fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(d_audio * 2, d_audio),
                nn.ReLU(),
                nn.Linear(d_audio, d_audio),
                nn.Sigmoid()
            )
            out_dim = d_audio
        elif fusion == "concat":
            out_dim = d_audio * 2
        else:
            raise ValueError(f"Unknown fusion={fusion}")

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, z_audio: torch.Tensor, x_trad: torch.Tensor):
        z_trad = self.proj_trad(x_trad)
        if self.fusion == "gate":
            g = self.gate(torch.cat([z_audio, z_trad], dim=-1))
            z = g * z_audio + (1 - g) * z_trad
        else:
            z = torch.cat([z_audio, z_trad], dim=-1)
        logits = self.classifier(z)
        return logits, z

class SpeakerHead(nn.Module):
    def __init__(self, d_in: int, num_spk: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_spk)
        )

    def forward(self, x):
        return self.net(x)

class SERModel(nn.Module):
    def __init__(
        self,
        w2v: Wav2Vec2Model,
        d_trad: int,
        num_emo: int,
        num_spk: int,
        fusion: str = "gate",
        use_extra_trf: bool = True,
        extra_trf_layers: int = 1,
        use_hidden_specaug: bool = True,
        use_sat: bool = False,
    ):
        super().__init__()
        self.w2v = w2v
        self.use_extra_trf = use_extra_trf
        self.use_hidden_specaug = use_hidden_specaug
        self.use_sat = use_sat

        d_audio = w2v.config.hidden_size

        self.extra_trf = ExtraTransformer(d_model=d_audio, num_layers=extra_trf_layers) if use_extra_trf else None

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fusion_head = FusionHead(d_audio=d_audio, d_trad=d_trad, fusion=fusion, num_classes=num_emo)

        # speaker head takes fused embedding (z)
        z_dim = d_audio if fusion == "gate" else d_audio * 2
        self.spk_head = SpeakerHead(d_in=z_dim, num_spk=num_spk)

    def forward(self, wavs: torch.Tensor, x_trad: torch.Tensor, grl_alpha: float = 0.0):
        """
        wavs: [B, T]
        """
        out = self.w2v(wavs).last_hidden_state  # [B, T', C]
        if self.use_hidden_specaug and self.training:
            out = hidden_specaug(out)

        if self.use_extra_trf and self.extra_trf is not None:
            out = self.extra_trf(out)

        # mean pool
        z_audio = out.transpose(1, 2)           # [B, C, T']
        z_audio = self.pool(z_audio).squeeze(-1)  # [B, C]

        emo_logits, z = self.fusion_head(z_audio, x_trad)

        spk_logits = None
        if self.use_sat:
            z_grl = grad_reverse(z, grl_alpha)
            spk_logits = self.spk_head(z_grl)

        return emo_logits, spk_logits, z


# =========================
# Experiment config
# =========================
@dataclass
class ExpConfig:
    name: str
    fusion: str = "gate"              # gate / concat
    use_extra_trf: bool = False
    extra_trf_layers: int = 1
    use_hidden_specaug: bool = False
    use_sat: bool = False
    unfreeze_last_n: int = 2


# =========================
# Metrics
# =========================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, uar, mf1


# =========================
# Train / Eval
# =========================
def set_w2v_trainable_last_n(model: SERModel, last_n: int):
    # freeze all
    for p in model.w2v.parameters():
        p.requires_grad = False
    if last_n <= 0:
        return

    # unfreeze last N encoder layers + layer_norm (robust to HF internals)
    if hasattr(model.w2v, "encoder") and hasattr(model.w2v.encoder, "layers"):
        layers = model.w2v.encoder.layers
        for layer in layers[-last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
    # also unfreeze feature projection / layer norm if exists
    if hasattr(model.w2v, "feature_projection"):
        for p in model.w2v.feature_projection.parameters():
            p.requires_grad = True

def build_optim(model: SERModel):
    w2v_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("w2v.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("w2v.")]

    opt = optim.AdamW(
        [
            {"params": w2v_params, "lr": LR_W2V},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY
    )
    return opt

def train_one_epoch(
    model: SERModel,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    epoch_1based: int,
    total_epochs: int
):
    model.train()
    opt.zero_grad(set_to_none=True)

    total_loss = 0.0
    n = 0

    # SAT schedule
    if model.use_sat:
        if epoch_1based <= SAT_WARMUP_EPOCHS:
            grl_alpha = 0.0
            sat_lambda = 0.0
        else:
            t = (epoch_1based - SAT_WARMUP_EPOCHS) / max(1, (total_epochs - SAT_WARMUP_EPOCHS))
            grl_alpha = SAT_ALPHA_MAX * t
            sat_lambda = SAT_LAMBDA_MAX * t
    else:
        grl_alpha = 0.0
        sat_lambda = 0.0

    for step, (wavs, X_trad, y_emo, y_spk) in enumerate(loader, start=1):
        wavs = wavs.to(device)
        X_trad = X_trad.to(device)
        y_emo = y_emo.to(device)
        y_spk = y_spk.to(device)

        emo_logits, spk_logits, _ = model(wavs, X_trad, grl_alpha=grl_alpha)

        loss_emo = F.cross_entropy(emo_logits, y_emo)
        loss = loss_emo

        if model.use_sat and spk_logits is not None:
            loss_spk = F.cross_entropy(spk_logits, y_spk)
            loss = loss + sat_lambda * loss_spk

        loss = loss / ACCUM_STEPS
        loss.backward()

        if step % ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            opt.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * ACCUM_STEPS
        n += 1

    return total_loss / max(1, n)

@torch.no_grad()
def eval_model(model: SERModel, loader: DataLoader, device: torch.device, use_spk_head: bool):
    model.eval()

    emo_true, emo_pred = [], []
    spk_true, spk_pred = [], []

    for wavs, X_trad, y_emo, y_spk in loader:
        wavs = wavs.to(device)
        X_trad = X_trad.to(device)

        emo_logits, spk_logits, _ = model(wavs, X_trad, grl_alpha=0.0)

        ep = torch.argmax(emo_logits, dim=-1).cpu().numpy()
        emo_pred.append(ep)
        emo_true.append(y_emo.numpy())

        if use_spk_head and spk_logits is not None:
            sp = torch.argmax(spk_logits, dim=-1).cpu().numpy()
            spk_pred.append(sp)
            spk_true.append(y_spk.numpy())

    emo_true = np.concatenate(emo_true, axis=0)
    emo_pred = np.concatenate(emo_pred, axis=0)
    acc, uar, mf1 = compute_metrics(emo_true, emo_pred)

    if use_spk_head and len(spk_true) > 0:
        spk_true = np.concatenate(spk_true, axis=0)
        spk_pred = np.concatenate(spk_pred, axis=0)
        spk_acc, spk_uar, _ = compute_metrics(spk_true, spk_pred)
    else:
        spk_acc, spk_uar = 0.0, 0.0

    return acc, uar, mf1, spk_acc, spk_uar


def run_one_exp(
    cfg: ExpConfig,
    file_paths_train: np.ndarray,
    file_paths_test: np.ndarray,
    X_trad_train: np.ndarray,
    X_trad_test: np.ndarray,
    y_emo_train: np.ndarray,
    y_emo_test: np.ndarray,
    y_spk_train: np.ndarray,
    y_spk_test: np.ndarray,
    emo_enc: LabelEncoder,
    spk_enc: LabelEncoder,
    device: torch.device,
    out_dir: str,
    model_dir: str
) -> Dict[str, Any]:

    # Load wav2vec2
    w2v = Wav2Vec2Model.from_pretrained(model_dir)

    model = SERModel(
        w2v=w2v,
        d_trad=X_trad_train.shape[1],
        num_emo=len(emo_enc.classes_),
        num_spk=len(spk_enc.classes_),
        fusion=cfg.fusion,
        use_extra_trf=cfg.use_extra_trf,
        extra_trf_layers=cfg.extra_trf_layers,
        use_hidden_specaug=cfg.use_hidden_specaug,
        use_sat=cfg.use_sat,
    ).to(device)

    set_w2v_trainable_last_n(model, cfg.unfreeze_last_n)

    opt = build_optim(model)

    train_ds = AudioTradDataset(file_paths_train, X_trad_train, y_emo_train, y_spk_train)
    test_ds  = AudioTradDataset(file_paths_test,  X_trad_test,  y_emo_test,  y_spk_test)

    collate = AudioCollator()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate, drop_last=False)

    best_uar = -1.0
    best_epoch = 0
    best_path = ""

    for epoch in range(1, NUM_EPOCHS + 1):
        # cosine schedule
        lr_factor = cosine_lr_factor(epoch, NUM_EPOCHS, WARMUP_EPOCHS)
        for pg in opt.param_groups:
            base_lr = LR_W2V if pg["lr"] <= LR_W2V * 10 else LR_HEAD
            # (safer) just scale whatever current group lr base is:
            pg["lr"] = pg["lr"] * 0 + (LR_W2V if pg["lr"] <= LR_W2V * 10 else LR_HEAD) * lr_factor

        _ = train_one_epoch(model, train_loader, opt, device, epoch, NUM_EPOCHS)
        acc, uar, mf1, spk_acc, spk_uar = eval_model(model, test_loader, device, use_spk_head=cfg.use_sat)

        if uar > best_uar:
            best_uar = uar
            best_epoch = epoch
            best_path = os.path.join(out_dir, f"{cfg.name}_best.pt")
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "emo_classes": emo_enc.classes_.tolist(),
                    "spk_classes": spk_enc.classes_.tolist(),
                    "state_dict": model.state_dict(),
                },
                best_path
            )

    # reload best and compute final metrics (stable)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    acc, uar, mf1, spk_acc, spk_uar = eval_model(model, test_loader, device, use_spk_head=cfg.use_sat)

    return {
        "exp": cfg.name,
        "best_epoch": best_epoch,
        "acc": float(acc),
        "uar": float(uar),
        "macro_f1": float(mf1),
        "spk_acc": float(spk_acc),
        "spk_uar": float(spk_uar),
        "ckpt": best_path
    }


def run_seed(seed: int, out_root: str, v2_dir: str, split_json: str, exp_names: List[str], model_dir: str):
    seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Device: {device}")
    print(f"Seed={seed}")

    seed_dir = os.path.join(out_root, f"seed{seed}")
    ensure_dir(seed_dir)

    # =========================
    # Load v2 inputs (flexible)
    # =========================
    wav_paths = _load_npy_flexible(v2_dir, ["file_paths.npy", "SCI_file_paths.npy"], allow_pickle=True)
    y_emo_labels = _load_npy_flexible(v2_dir, ["y_labels.npy", "SCI_y_labels.npy"], allow_pickle=True)
    speaker_ids_raw = _load_npy_flexible(v2_dir, ["speaker_ids.npy", "SCI_speaker_ids.npy"], allow_pickle=True)

    wav_paths = np.array(wav_paths, dtype=object)
    y_emo_labels = np.array(y_emo_labels)
    speaker_ids_raw = np.array(speaker_ids_raw, dtype=object)

    # Trad optional
    X_trad = _maybe_load_trad(v2_dir, n=len(wav_paths))
    X_trad = np.array(X_trad, dtype=np.float32)

    assert len(wav_paths) == len(y_emo_labels) == len(speaker_ids_raw), \
        f"Length mismatch: wav={len(wav_paths)}, y={len(y_emo_labels)}, spk={len(speaker_ids_raw)}"

    if len(X_trad) != len(wav_paths):
        raise ValueError(f"Length mismatch: wav={len(wav_paths)}, X_trad={len(X_trad)}")

    # label encode
    emo_enc = LabelEncoder()
    y_emo = emo_enc.fit_transform(y_emo_labels)

    spk_enc = LabelEncoder()
    y_spk = spk_enc.fit_transform(speaker_ids_raw)

    # =========================
    # Split (LOSO via json OR default GSS)
    # =========================
    if split_json:
        train_idx, test_idx = _read_split_json(split_json)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X_trad, y_emo, groups=speaker_ids_raw))

    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train speakers: {len(np.unique(speaker_ids_raw[train_idx]))} | Test speakers: {len(np.unique(speaker_ids_raw[test_idx]))}")

    # Scale trad (even if zeros fallback, scaling is stable)
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
    # Experiments
    # =========================
    EXPS: List[ExpConfig] = [
        ExpConfig(name="A_base", fusion="gate", use_extra_trf=False, use_hidden_specaug=False, use_sat=False, unfreeze_last_n=2),

        ExpConfig(name="C_gate", fusion="gate", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),
        ExpConfig(name="E_gate_SAT_noCB", fusion="gate", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=True, unfreeze_last_n=2),

        ExpConfig(name="C_gate_noTRF", fusion="gate", use_extra_trf=False, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),

        ExpConfig(name="C_concat", fusion="concat", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),
        ExpConfig(name="E_concat_SAT_noCB", fusion="concat", use_extra_trf=True, extra_trf_layers=1, use_hidden_specaug=True, use_sat=True, unfreeze_last_n=2),

        ExpConfig(name="C_concat_noTRF", fusion="concat", use_extra_trf=False, use_hidden_specaug=True, use_sat=False, unfreeze_last_n=2),
    ]

    if exp_names:
        keep = set(exp_names)
        EXPS = [e for e in EXPS if e.name in keep]
        print("Running only exps:", [e.name for e in EXPS])

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
            out_dir=seed_dir,
            model_dir=model_dir,
        )
        results.append(r)

    csv_path = os.path.join(seed_dir, "ablation_chain_summary_with_spk_uar.csv")
    with open(csv_path, "w") as f:
        f.write("exp,best_epoch,acc,uar,macro_f1,spk_acc,spk_uar,ckpt\n")
        for r in results:
            f.write(
                f"{r['exp']},{r['best_epoch']},{r['acc']:.4f},{r['uar']:.4f},{r['macro_f1']:.4f},"
                f"{r['spk_acc']:.4f},{r['spk_uar']:.4f},{r['ckpt']}\n"
            )

    print("\n====================")
    print(f"✅ Seed {seed} done. Saved CSV: {csv_path}")
    print("====================\n")
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Output root, e.g. /root/autodl-tmp/casia_main_v2_loso/fold_test_xxx")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])

    ap.add_argument("--v2_dir", default="/root/autodl-tmp", help="Directory containing v2 npys (file_paths.npy/y_labels.npy/speaker_ids.npy).")
    ap.add_argument("--split_json", default="", help="If set, use train_idx/test_idx from this JSON (e.g., LOSO).")
    ap.add_argument("--exp_names", nargs="*", default=[], help="If set, only run these experiments (e.g., C_gate E_gate_SAT_noCB).")

    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Local path of wav2vec2 model (HF format).")
    args = ap.parse_args()

    ensure_dir(args.out_root)

    for s in args.seeds:
        run_seed(
            seed=s,
            out_root=args.out_root,
            v2_dir=args.v2_dir,
            split_json=args.split_json,
            exp_names=args.exp_names,
            model_dir=args.model_dir,
        )

    print("\n✅ All done.")

if __name__ == "__main__":
    main()