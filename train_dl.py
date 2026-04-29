"""
train_dl.py — Fine-tune SBERT dual-encoder for resume–JD matching.

Improvements over original dl_training.ipynb:
  - Deeper regression head (512 → 128 → 1 with BatchNorm + GELU)
  - MAX_LEN raised to 384 (from 256) to capture more resume content
  - Learning rate scheduler (OneCycleLR)
  - Gradient clipping (max_norm=1.0)
  - Early stopping with configurable patience
  - Comprehensive evaluation (MSE, RMSE, MAE, R²)
  - Prediction distribution plot

Usage:
    python train_dl.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "data/cleaned_data/merged_training_v4.csv"
SAVE_DIR = "models/dl_resume_match"
PLOTS_DIR = "models/plots"

EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 384
PATIENCE = 3          # early stopping patience
GRAD_CLIP = 1.0       # gradient clipping max norm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────
class ResumeJDDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        resume_text = str(self.df.iloc[idx]["resume_text"])
        jd_text = str(self.df.iloc[idx]["jd_text"])
        label = self.df.iloc[idx]["match_score"] / 100.0

        enc_resume = self.tokenizer(
            resume_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc_jd = self.tokenizer(
            jd_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "resume_input_ids": enc_resume["input_ids"].squeeze(),
            "resume_attention_mask": enc_resume["attention_mask"].squeeze(),
            "jd_input_ids": enc_jd["input_ids"].squeeze(),
            "jd_attention_mask": enc_jd["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.float),
        }


# ── Model ────────────────────────────────────────────────────────────────────
class ResumeMatchModel(nn.Module):
    """Dual-encoder SBERT with deeper regression head."""

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask):
        r_out = self.encoder(
            input_ids=resume_input_ids, attention_mask=resume_attention_mask
        ).pooler_output
        j_out = self.encoder(
            input_ids=jd_input_ids, attention_mask=jd_attention_mask
        ).pooler_output
        combined = torch.cat((r_out, j_out), dim=1)
        return self.regressor(combined).squeeze(-1)


# ── Training / Evaluation ────────────────────────────────────────────────────
def train_fn(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Training", leave=False):
        optimizer.zero_grad()
        outputs = model(
            batch["resume_input_ids"].to(DEVICE),
            batch["resume_attention_mask"].to(DEVICE),
            batch["jd_input_ids"].to(DEVICE),
            batch["jd_attention_mask"].to(DEVICE),
        )
        loss = criterion(outputs, batch["label"].to(DEVICE))
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_fn(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, truths = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            outputs = model(
                batch["resume_input_ids"].to(DEVICE),
                batch["resume_attention_mask"].to(DEVICE),
                batch["jd_input_ids"].to(DEVICE),
                batch["jd_attention_mask"].to(DEVICE),
            )
            loss = criterion(outputs, batch["label"].to(DEVICE))
            total_loss += loss.item()
            preds.extend(outputs.cpu().numpy())
            truths.extend(batch["label"].cpu().numpy())
    return total_loss / len(loader), np.array(preds), np.array(truths)


def compute_metrics(preds, truths):
    """Metrics on the 0–100 scale."""
    p = preds * 100
    t = truths * 100
    return {
        "mse": float(mean_squared_error(t, p)),
        "rmse": float(np.sqrt(mean_squared_error(t, p))),
        "mae": float(mean_absolute_error(t, p)),
        "r2": float(r2_score(t, p)),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_ds = ResumeJDDataset(train_df, tokenizer, MAX_LEN)
    val_ds = ResumeJDDataset(val_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = ResumeMatchModel(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.MSELoss()

    # OneCycleLR scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 10, total_steps=total_steps
    )

    # ── Training loop with early stopping ────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\nStarting training — {EPOCHS} epochs, patience={PATIENCE}")
    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, criterion)
        val_loss, preds, truths = eval_fn(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        metrics = compute_metrics(preds, truths)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"Train: {train_loss:.5f}  Val: {val_loss:.5f}  "
            f"RMSE: {metrics['rmse']:.2f}  R²: {metrics['r2']:.4f}  "
            f"LR: {lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "dl_resume_match.pt"),
            )
            best_metrics = metrics
            best_preds = preds
            best_truths = truths
            print("  ✅ Model saved!")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"  🛑 Early stopping at epoch {epoch + 1}")
                break

    # ── Save metrics ─────────────────────────────────────────────────────
    best_metrics["best_val_loss"] = float(best_val_loss)
    best_metrics["epochs_trained"] = epoch + 1
    with open(os.path.join(SAVE_DIR, "training_metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\n📊  Metrics saved → {SAVE_DIR}/training_metrics.json")

    # ── Plots ────────────────────────────────────────────────────────────
    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("DL Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dl_loss_curve.png"), dpi=150)
    plt.close()

    # Predicted vs Actual
    plt.figure(figsize=(7, 7))
    plt.scatter(best_truths * 100, best_preds * 100, alpha=0.4, s=10)
    plt.plot([0, 100], [0, 100], "r--", lw=1.5, label="Perfect")
    plt.xlabel("Actual Match Score")
    plt.ylabel("Predicted Match Score")
    plt.title("DL Model — Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dl_pred_vs_actual.png"), dpi=150)
    plt.close()

    print(f"📈  Plots saved → {PLOTS_DIR}/")
    print("\n✅ DL training complete!")


if __name__ == "__main__":
    main()
