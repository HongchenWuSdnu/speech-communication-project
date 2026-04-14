# =========================
# train_main_A0.py
# Main baseline: A0 (Wav2Vec2 fully frozen, NO SAT)
# Offline HuggingFace local dir + soundfile wav loader
# =========================
import os
import random
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

import torchaudio  # only for Resample
import soundfile as sf
from transformers import Wav2Vec2Model

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# --------- EDIT IF NEEDED ----------
MODEL_DIR = "/root/autodl-tmp/wav2vec2-base-local"
DATA_WAV_DIR = "/root/autodl-tmp/AudioWAV"
X_TRAD_PATH  = "/root/autodl-tmp/SCI_X_trad.npy"
Y_EMO_PATH   = "/root/autodl-tmp/SCI_y_labels.npy"
SPK_ID_PATH  = "/root/autodl-tmp/SCI_speaker_ids.npy"
OUT_DIR      = "/root/autodl-tmp"
# ----------------------------------


# =========================
# Utils
# =========================
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# GRL (kept for compatibility, but alpha=0 in A0)
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
# Dataset + Collator
# =========================
class AudioTradDataset(Dataset):
    def __init__(self, file_paths, x_trad, y_emo, y_spk, target_sr=16000, max_sec=3.0):
        self.file_paths = list(file_paths)
        self.x_trad = torch.tensor(x_trad, dtype=torch.float32)
        self.y_emo = torch.tensor(y_emo, dtype=torch.long)
        self.y_spk = torch.tensor(y_spk, dtype=torch.long)
        self.target_sr = target_sr
        self.target_len = int(target_sr * max_sec)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav, sr = sf.read(self.file_paths[idx], dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)  # mono
        wav = torch.from_numpy(wav)

        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(wav)

        n = wav.shape[0]
        if n < self.target_len:
            wav = F.pad(wav, (0, self.target_len - n))
        else:
            wav = wav[:self.target_len]

        return {
            "waveform": wav,         # (L,)
            "x_trad": self.x_trad[idx],
            "y_emo": self.y_emo[idx],
            "y_spk": self.y_spk[idx],
        }


class CollatorFixedLen:
    """Dataset already pads/trims to fixed length, so collate is simple and robust."""
    def __call__(self, batch):
        waveforms = torch.stack([b["waveform"] for b in batch], dim=0)  # (B, L)
        attn_mask = torch.ones_like(waveforms, dtype=torch.long)         # (B, L)
        x_trad = torch.stack([b["x_trad"] for b in batch], dim=0)
        y_emo = torch.stack([b["y_emo"] for b in batch], dim=0)
        y_spk = torch.stack([b["y_spk"] for b in batch], dim=0)
        return waveforms, attn_mask, x_trad, y_emo, y_spk


# =========================
# Model (A0: W2V fully frozen)
# =========================
class E2E_Net_A0(nn.Module):
    def __init__(self, num_emotions: int, num_speakers: int, speaker_head_dim: int = 32):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_DIR, local_files_only=True)
        self.wav2vec2.freeze_feature_encoder()

        # ✅ A0: freeze ALL wav2vec2 params
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

        self.trad_stream = nn.Sequential(
            nn.Linear(45, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=768, hidden_size=64,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        self.gate_trad = nn.Linear(128, 128)
        self.gate_deep = nn.Linear(128, 128)

        self.emotion_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions),
        )

        # kept for reporting (will be trained but alpha=0, lambda=0 => effectively irrelevant)
        self.speaker_classifier = nn.Sequential(
            nn.Linear(128, speaker_head_dim),
            nn.BatchNorm1d(speaker_head_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(speaker_head_dim, num_speakers),
        )

    def forward(self, input_values, attention_mask, x_trad, alpha: float):
        hs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state  # (B,T,768)

        lstm_out, _ = self.lstm(hs)  # (B,T,128)
        attn = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (B,T,1)
        h_deep = torch.sum(lstm_out * attn, dim=1)  # (B,128)

        h_trad = self.trad_stream(x_trad)  # (B,128)

        z = torch.sigmoid(self.gate_trad(h_trad) + self.gate_deep(h_deep))
        h_fusion = z * h_trad + (1 - z) * h_deep

        emo_logits = self.emotion_classifier(h_fusion)
        spk_logits = self.speaker_classifier(grad_reverse(h_fusion, alpha))
        return emo_logits, spk_logits


# =========================
# Train/Eval
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for input_values, attn_mask, x_trad, y_emo, _y_spk in loader:
        input_values = input_values.to(device)
        attn_mask = attn_mask.to(device)
        x_trad = x_trad.to(device)

        emo_logits, _ = model(input_values, attn_mask, x_trad, alpha=0.0)
        y_hat = emo_logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y_emo.numpy())
        y_pred.extend(y_hat)

    acc = accuracy_score(y_true, y_pred) * 100
    uar = recall_score(y_true, y_pred, average="macro") * 100
    mf1 = f1_score(y_true, y_pred, average="macro") * 100
    return acc, uar, mf1


def train_one_epoch(model, loader, optimizer, device, criterion_emo,
                    accum_steps=1, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    emo_loss_sum, n = 0.0, 0
    for step, (input_values, attn_mask, x_trad, y_emo, _y_spk) in enumerate(loader):
        input_values = input_values.to(device)
        attn_mask = attn_mask.to(device)
        x_trad = x_trad.to(device)
        y_emo = y_emo.to(device)

        emo_logits, _ = model(input_values, attn_mask, x_trad, alpha=0.0)
        loss_emo = criterion_emo(emo_logits, y_emo)

        (loss_emo / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        emo_loss_sum += loss_emo.item()
        n += 1

    return emo_loss_sum / max(n, 1)


# =========================
# Main
# =========================
if __name__ == "__main__":
    seed_all(42)

    # Train config
    NUM_EPOCHS   = 30
    BATCH_SIZE   = 32
    ACCUM_STEPS  = 1
    NUM_WORKERS  = 4
    LR_HEAD      = 5e-4
    WEIGHT_DECAY = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}")

    # Load data
    all_files = sorted([f for f in os.listdir(DATA_WAV_DIR) if f.endswith(".wav")])
    wav_paths = np.array([os.path.join(DATA_WAV_DIR, f) for f in all_files])

    X_trad = np.load(X_TRAD_PATH)
    y_emo_labels = np.load(Y_EMO_PATH)
    speaker_ids  = np.load(SPK_ID_PATH)

    assert len(wav_paths) == len(X_trad) == len(y_emo_labels) == len(speaker_ids), \
        f"Length mismatch: wav={len(wav_paths)}, X_trad={len(X_trad)}, y={len(y_emo_labels)}, spk={len(speaker_ids)}"

    emo_enc = LabelEncoder()
    y_emo = emo_enc.fit_transform(y_emo_labels)

    spk_enc = LabelEncoder()
    y_spk = spk_enc.fit_transform(speaker_ids)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_trad, y_emo, groups=speaker_ids))
    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train speakers: {len(np.unique(speaker_ids[train_idx]))} | Test speakers: {len(np.unique(speaker_ids[test_idx]))}")

    scaler = StandardScaler()
    X_trad_train = scaler.fit_transform(X_trad[train_idx])
    X_trad_test  = scaler.transform(X_trad[test_idx])

    train_ds = AudioTradDataset(wav_paths[train_idx], X_trad_train, y_emo[train_idx], y_spk[train_idx])
    test_ds  = AudioTradDataset(wav_paths[test_idx],  X_trad_test,  y_emo[test_idx],  y_spk[test_idx])

    collate = CollatorFixedLen()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)

    model = E2E_Net_A0(num_emotions=len(emo_enc.classes_), num_speakers=len(spk_enc.classes_)).to(device)
    print(f"Trainable params: {count_trainable_params(model)/1e6:.2f}M")

    criterion_emo = nn.CrossEntropyLoss()

    # optimizer: only train heads (wav2vec2 frozen)
    head_params = []
    for m in [model.trad_stream, model.lstm, model.temporal_attn,
              model.gate_trad, model.gate_deep,
              model.emotion_classifier, model.speaker_classifier]:
        head_params += list(m.parameters())

    optimizer = optim.AdamW([{"params": head_params, "lr": LR_HEAD}],
                            weight_decay=WEIGHT_DECAY)

    best_uar = -1.0
    best_path = os.path.join(OUT_DIR, "E2E_A0_best.pth")

    print("\n⚔️ Start training (A0 baseline)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, criterion_emo,
                                  accum_steps=ACCUM_STEPS, max_grad_norm=1.0)
        acc, uar, mf1 = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | TrainLoss={tr_loss:.4f} | Val Acc={acc:.2f}% UAR={uar:.2f}% MF1={mf1:.2f}%")

        if uar > best_uar:
            best_uar = uar
            torch.save(
                {
                    "model": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "emo_classes": emo_enc.classes_,
                    "spk_classes": spk_enc.classes_,
                    "best_uar": best_uar,
                    "epoch": epoch,
                },
                best_path
            )
            print(f"  ✅ Saved best to {best_path} (best UAR={best_uar:.2f}%)")

    print(f"\n🔥 Done. Best UAR={best_uar:.2f}% | ckpt: {best_path}")