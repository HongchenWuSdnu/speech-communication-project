# run_ablation_chain_Csat_noCB.py
import os
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional

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
# Config
# =========================
MODEL_DIR = "/root/autodl-tmp/wav2vec2-base-local"

# v2-aligned file list (absolute paths, stable order)
FILE_PATHS_NPY = "/root/autodl-tmp/SCI_file_paths.npy"

# v2-aligned labels/speakers (already aligned with FILE_PATHS_NPY)
Y_EMO_PATH  = "/root/autodl-tmp/SCI_y_labels.npy"
SPK_ID_PATH = "/root/autodl-tmp/SCI_speaker_ids.npy"

# v2-aligned traditional features (IMPORTANT: must match FILE_PATHS_NPY order)
X_TRAD_PATH = "/root/autodl-tmp/SCI_X_trad_v2.npy"

# keep for legacy/reference (not used after switching to FILE_PATHS_NPY)
DATA_WAV_DIR = "/root/autodl-tmp/AudioWAV"

# reproducibility (set by sed when running multiple seeds)
SEED = 42
SPLIT_RANDOM_STATE = 42

OUT_ROOT = f"/root/autodl-tmp/ablation_chain_CsatNoCB_seed{SEED}"

# training
NUM_EPOCHS   = 30
BATCH_SIZE   = 16
ACCUM_STEPS  = 2
NUM_WORKERS  = 4

LR_W2V       = 1e-5
LR_HEAD      = 5e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

# SAT schedule (weak ramp)
SAT_WARMUP_EPOCHS = 8     # first N epochs: no SAT
SAT_ALPHA_MAX     = 0.02
SAT_LAMBDA_MAX    = 0.003

# audio
TARGET_SR = 16000
MAX_SEC   = 3.0

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

def cosine_lr_factor(epoch_idx_1based: int, total_epochs: int, warmup_epochs: int) -> float:
    e = epoch_idx_1based
    warmup_epochs = max(1, int(warmup_epochs))
    if e <= warmup_epochs:
        return e / warmup_epochs
    t = (e - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * t))

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
        attention_mask = torch.ones_like(waveforms, dtype=torch.long)   # (B, L) waveform-level

        x_trad = torch.stack([b["x_trad"] for b in batch], dim=0)       # (B, 45)
        y_emo  = torch.stack([b["y_emo"] for b in batch], dim=0)        # (B,)
        y_spk  = torch.stack([b["y_spk"] for b in batch], dim=0)        # (B,)

        return waveforms, attention_mask, x_trad, y_emo, y_spk

# =========================
# Components
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
# Model
# =========================
@dataclass
class ExpConfig:
    name: str
    unfreeze_last_n: int = 2
    use_extra_trf: bool = False
    extra_trf_layers: int = 1
    use_hidden_specaug: bool = False
    use_class_balanced: bool = False
    use_sat: bool = False

class E2EModel(nn.Module):
    def __init__(self, num_emotions: int, num_speakers: int, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_DIR, local_files_only=True)
        self.wav2vec2.freeze_feature_encoder()

        for p in self.wav2vec2.parameters():
            p.requires_grad = False

        lastN = int(cfg.unfreeze_last_n)
        if lastN > 0:
            last_layers = list(range(12 - lastN, 12))
            for name, param in self.wav2vec2.named_parameters():
                if any(f"encoder.layers.{i}." in name for i in last_layers):
                    param.requires_grad = True

        self.specaug = HiddenSpecAug() if cfg.use_hidden_specaug else None
        self.extra_trf = ExtraTransformerEncoder(num_layers=cfg.extra_trf_layers) if cfg.use_extra_trf else None

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

        self.speaker_classifier = nn.Sequential(
            nn.Linear(128, 32),
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
            B, T, _ = hs.shape
            key_padding_mask = torch.zeros((B, T), dtype=torch.bool, device=hs.device)
            hs = self.extra_trf(hs, key_padding_mask=key_padding_mask)

        lstm_out, _ = self.lstm(hs)  # (B, T, 128)
        attn = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (B, T, 1)
        h_deep = torch.sum(lstm_out * attn, dim=1)  # (B, 128)

        h_trad = self.trad_stream(x_trad)  # (B, 128)
        z = torch.sigmoid(self.gate_trad(h_trad) + self.gate_deep(h_deep))
        h_fusion = z * h_trad + (1 - z) * h_deep  # (B,128)

        emo_logits = self.emotion_classifier(h_fusion)
        spk_logits = self.speaker_classifier(grad_reverse(h_fusion, alpha))
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
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    spk_true, spk_pred = [], []

    for input_values, attn_mask, x_trad, y_emo, y_spk in loader:
        input_values = input_values.to(device)
        attn_mask = attn_mask.to(device)
        x_trad = x_trad.to(device)

        emo_logits, spk_logits = model(input_values, attn_mask, x_trad, alpha=0.0)

        y_hat = emo_logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y_emo.numpy())
        y_pred.extend(y_hat)

        spk_hat = spk_logits.argmax(dim=1).cpu().numpy()
        spk_true.extend(y_spk.numpy())
        spk_pred.extend(spk_hat)

    acc = accuracy_score(y_true, y_pred) * 100
    uar = recall_score(y_true, y_pred, average="macro") * 100
    mf1 = f1_score(y_true, y_pred, average="macro") * 100
    spk_acc = accuracy_score(spk_true, spk_pred) * 100
    return acc, uar, mf1, spk_acc

def train_one_epoch(model, loader, optimizer, device,
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
            loss = loss_emo + lambda_spk * loss_spk
            loss = loss / accum_steps

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
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        emo_loss_sum += float(loss_emo.item())
        spk_loss_sum += float(loss_spk.item())
        n += 1

    return emo_loss_sum / max(n, 1), spk_loss_sum / max(n, 1)

# =========================
# Experiment runner
# =========================
def run_experiment(cfg: ExpConfig,
                   file_paths_train, file_paths_test,
                   X_trad_train, X_trad_test,
                   y_emo_train, y_emo_test,
                   y_spk_train, y_spk_test,
                   emo_enc, spk_enc, scaler_feat: StandardScaler,
                   device,
                   out_dir: str):

    print(f"\n====================\n▶ {cfg.name}\n====================")

    train_ds = AudioTradDataset(file_paths_train, X_trad_train, y_emo_train, y_spk_train,
                                target_sr=TARGET_SR, max_sec=MAX_SEC)
    test_ds  = AudioTradDataset(file_paths_test,  X_trad_test,  y_emo_test,  y_spk_test,
                                target_sr=TARGET_SR, max_sec=MAX_SEC)

    collate = CollatorSimple()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate
    )

    model = E2EModel(num_emotions=len(emo_enc.classes_),
                     num_speakers=len(spk_enc.classes_),
                     cfg=cfg).to(device)

    trainable = count_trainable_params(model)
    print(f"Trainable params: {trainable/1e6:.2f}M | lastN={cfg.unfreeze_last_n} | "
          f"extraTRF={cfg.use_extra_trf}({cfg.extra_trf_layers if cfg.use_extra_trf else 0}) | "
          f"specaug={cfg.use_hidden_specaug} | CB={cfg.use_class_balanced} | SAT={cfg.use_sat}")

    # loss
    if cfg.use_class_balanced:
        w = make_class_balanced_weights(np.array(y_emo_train), num_classes=len(emo_enc.classes_)).to(device)
        criterion_emo = nn.CrossEntropyLoss(weight=w)
    else:
        criterion_emo = nn.CrossEntropyLoss()
    criterion_spk = nn.CrossEntropyLoss()

    wav2vec2_params = [p for p in model.wav2vec2.parameters() if p.requires_grad]
    head_params = []
    for m in [model.trad_stream, model.lstm, model.temporal_attn,
              model.gate_trad, model.gate_deep,
              model.emotion_classifier, model.speaker_classifier]:
        head_params += list(m.parameters())
    if model.specaug is not None:
        head_params += list(model.specaug.parameters())
    if model.extra_trf is not None:
        head_params += list(model.extra_trf.parameters())

    param_groups = []
    if len(wav2vec2_params) > 0:
        param_groups.append({"params": wav2vec2_params, "lr": LR_W2V})
    param_groups.append({"params": head_params, "lr": LR_HEAD})
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    base_lr_w2v = LR_W2V
    base_lr_head = LR_HEAD
    warmup_epochs = 5

    use_amp = torch.cuda.is_available()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = {"uar": -1, "acc": -1, "mf1": -1, "spk_acc": -1, "epoch": -1}
    ckpt_path = os.path.join(out_dir, f"{cfg.name}_best.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        fac = cosine_lr_factor(epoch, NUM_EPOCHS, warmup_epochs)
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

        tr_emo, tr_spk = train_one_epoch(
            model, train_loader, optimizer, device,
            criterion_emo, criterion_spk,
            alpha=alpha, lambda_spk=lam,
            accum_steps=ACCUM_STEPS, max_grad_norm=MAX_GRAD_NORM,
            scaler=amp_scaler, use_amp=use_amp
        )

        acc, uar, mf1, spk_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | lr={lr_show:.2e} | "
              f"Train Emo={tr_emo:.4f} Spk={tr_spk:.4f} | "
              f"Test Acc={acc:.2f}% UAR={uar:.2f}% MF1={mf1:.2f}% | SpkAcc={spk_acc:.2f}% | "
              f"alpha={alpha:.3f} lam={lam:.4f}")

        if uar > best["uar"]:
            best = {"uar": uar, "acc": acc, "mf1": mf1, "spk_acc": spk_acc, "epoch": epoch}
            torch.save({
                "model": model.state_dict(),
                "scaler_mean": scaler_feat.mean_,
                "scaler_scale": scaler_feat.scale_,
                "emo_classes": emo_enc.classes_,
                "spk_classes": spk_enc.classes_,
                "exp_cfg": cfg.__dict__,
                "best": best,
                "seed": SEED,
                "split_random_state": SPLIT_RANDOM_STATE,
            }, ckpt_path)
            print(f"  ✅ Saved best: {ckpt_path} | best UAR={best['uar']:.2f}% (epoch {best['epoch']})")

    print(f"✅ Finished {cfg.name} | Best={best}")
    return best, ckpt_path

# =========================
# Main
# =========================
def main():
    seed_all(SEED)
    os.makedirs(OUT_ROOT, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}")

    wav_paths = np.load(FILE_PATHS_NPY, allow_pickle=True)

    X_trad = np.load(X_TRAD_PATH)
    y_emo_labels = np.load(Y_EMO_PATH)
    speaker_ids  = np.load(SPK_ID_PATH, allow_pickle=True)

    assert len(wav_paths) == len(X_trad) == len(y_emo_labels) == len(speaker_ids), \
        f"Length mismatch: wav={len(wav_paths)}, X_trad={len(X_trad)}, y={len(y_emo_labels)}, spk={len(speaker_ids)}"

    emo_enc = LabelEncoder()
    y_emo = emo_enc.fit_transform(y_emo_labels)

    spk_enc = LabelEncoder()
    y_spk = spk_enc.fit_transform(speaker_ids)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_trad, y_emo, groups=speaker_ids))

    print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
    print(f"Train speakers: {len(np.unique(speaker_ids[train_idx]))} | Test speakers: {len(np.unique(speaker_ids[test_idx]))}")

    scaler_feat = StandardScaler()
    X_trad_train = scaler_feat.fit_transform(X_trad[train_idx])
    X_trad_test  = scaler_feat.transform(X_trad[test_idx])

    file_paths_train = wav_paths[train_idx]
    file_paths_test  = wav_paths[test_idx]

    y_emo_train = y_emo[train_idx]
    y_emo_test  = y_emo[test_idx]
    y_spk_train = y_spk[train_idx]
    y_spk_test  = y_spk[test_idx]

    # ===== A / C / C+SAT(noCB) =====
    exps = [
        ExpConfig(
            name="A_base",
            unfreeze_last_n=2,
            use_extra_trf=False,
            extra_trf_layers=0,
            use_hidden_specaug=False,
            use_class_balanced=False,
            use_sat=False
        ),
        ExpConfig(
            name="C_plus_specaug",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_class_balanced=False,
            use_sat=False
        ),
        ExpConfig(
            name="C_plus_SAT_noCB",
            unfreeze_last_n=2,
            use_extra_trf=True,
            extra_trf_layers=1,
            use_hidden_specaug=True,
            use_class_balanced=False,   # <-- key: NO class-balanced
            use_sat=True                # <-- key: SAT ON
        ),
    ]

    results = []
    csv_path = os.path.join(OUT_ROOT, "ablation_chain_summary.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    for cfg in exps:
        best, ckpt = run_experiment(
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
            out_dir=OUT_ROOT
        )
        results.append({
            "exp": cfg.name,
            "best_epoch": best["epoch"],
            "acc": best["acc"],
            "uar": best["uar"],
            "macro_f1": best["mf1"],
            "spk_acc": best["spk_acc"],
            "ckpt": ckpt,
        })

        header_needed = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if header_needed:
                f.write("exp,best_epoch,acc,uar,macro_f1,spk_acc,ckpt\n")
            r = results[-1]
            f.write(f"{r['exp']},{r['best_epoch']},{r['acc']:.4f},{r['uar']:.4f},{r['macro_f1']:.4f},{r['spk_acc']:.4f},{r['ckpt']}\n")

    print("\n====================\n✅ Summary (A / C / C+SAT_noCB)\n====================")
    for r in results:
        print(f"{r['exp']:18s} | epoch {int(r['best_epoch']):02d} | "
              f"Acc {r['acc']:.2f}% | UAR {r['uar']:.2f}% | MF1 {r['macro_f1']:.2f}% | SpkAcc {r['spk_acc']:.2f}%")
    print(f"\n📄 Saved CSV: {csv_path}")
    print(f"📦 All ckpts in: {OUT_ROOT}")

if __name__ == "__main__":
    main()