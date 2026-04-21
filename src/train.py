"""
train.py
========
End-to-end training script for the Ekstraliga pitch outcome prediction model.

Usage
-----
  python src/train.py --data data/ekstraliga/2025_Ekstraliga_Stats.xlsx
  python src/train.py --data data/ekstraliga/2025_Ekstraliga_Stats.xlsx --model lstm
  python src/train.py --data data/ekstraliga/2025_Ekstraliga_Stats.xlsx --model markov

Outputs (saved to runs/<timestamp>/)
  - model checkpoint (.pt for LSTM, .pkl for Markov)
  - training metrics (metrics.json)
  - confusion matrix figure (confusion_matrix.png)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import build_pa_dataframe
from features import (
    Vocabulary, VOCAB, OUTCOME_CATEGORIES,
    compute_batter_tendencies, compute_pitcher_tendencies,
    build_feature_matrix, PASequenceDataset,
    train_val_test_split,
)
from models.markov import OutcomeMarkov
from models.lstm_model import (
    PitchLSTM, FocalLoss, compute_class_weights,
    train_one_epoch, evaluate,
)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Ekstraliga pitch model")
    p.add_argument("--data",    default="data/ekstraliga/2025_Ekstraliga_Stats.xlsx")
    p.add_argument("--model",   choices=["markov", "lstm", "both"], default="both")
    p.add_argument("--epochs",  type=int, default=30)
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--hidden",  type=int, default=128)
    p.add_argument("--embed",   type=int, default=32)
    p.add_argument("--layers",  type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--outdir",  default=None)
    return p.parse_args()


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_run(args) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.outdir) if args.outdir else Path("runs") / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data pipeline ─────────────────────────────────────────────────────────────

def prepare_data(args):
    print("=" * 60)
    print("Step 1/4 — Loading & preprocessing PA logs")
    print("=" * 60)
    pa_df = build_pa_dataframe(args.data)

    print("\nStep 2/4 — Computing player tendency features")
    batter_tend  = compute_batter_tendencies(pa_df)
    pitcher_tend = compute_pitcher_tendencies(pa_df)
    feat_df      = build_feature_matrix(pa_df, batter_tend, pitcher_tend)

    print("\nStep 3/4 — Chronological train / val / test split")
    train_df, val_df, test_df = train_val_test_split(feat_df)

    return pa_df, feat_df, train_df, val_df, test_df, batter_tend, pitcher_tend


# ── Markov training ──────────────────────────────────────────────────────────

def run_markov(train_df, val_df, test_df, outdir: Path) -> dict:
    print("\n" + "=" * 60)
    print("Markov Baseline")
    print("=" * 60)

    model = OutcomeMarkov(smoothing=0.5)
    model.fit(train_df)

    val_metrics  = model.evaluate(val_df)
    test_metrics = model.evaluate(test_df)

    print(f"  Val  accuracy : {val_metrics['accuracy']:.4f}  (n={val_metrics['n']})")
    print(f"  Test accuracy : {test_metrics['accuracy']:.4f}  (n={test_metrics['n']})")

    import pickle
    with open(outdir / "markov_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return {"markov_val": val_metrics, "markov_test": test_metrics}


# ── LSTM training ─────────────────────────────────────────────────────────────

def run_lstm(train_df, val_df, test_df, args, outdir: Path) -> dict:
    print("\n" + "=" * 60)
    print("LSTM Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Datasets
    train_ds = PASequenceDataset(train_df)
    val_ds   = PASequenceDataset(val_df)
    test_ds  = PASequenceDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    # Model
    model = PitchLSTM(
        vocab_size    = len(VOCAB),
        embed_dim     = args.embed,
        hidden_dim    = args.hidden,
        n_layers      = args.layers,
        n_tend_feats  = train_ds.n_tendency_features,
        tend_proj_dim = 32,
        n_classes     = len(OUTCOME_CATEGORIES),
        dropout       = args.dropout,
    ).to(device)

    params = model.count_params()
    print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Loss — class-weighted cross entropy
    class_weights = compute_class_weights(
        train_df["outcome_label"].tolist(), len(OUTCOME_CATEGORIES)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    best_val_acc = 0.0
    best_epoch   = 0
    history      = []

    print(f"\n  Training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch":      epoch,
            "train_loss": train_m["loss"],
            "train_acc":  train_m["accuracy"],
            "val_loss":   val_m["loss"],
            "val_acc":    val_m["accuracy"],
        }
        history.append(row)

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch   = epoch
            torch.save(model.state_dict(), outdir / "lstm_best.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  [{epoch:3d}/{args.epochs}] "
                  f"train_loss={train_m['loss']:.4f}  train_acc={train_m['accuracy']:.4f} | "
                  f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}")

    print(f"\n  Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(outdir / "lstm_best.pt", map_location=device))
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"  Test accuracy: {test_m['accuracy']:.4f}")

    # Save training history
    history_df_path = outdir / "lstm_history.json"
    with open(history_df_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "lstm_best_val_acc":  best_val_acc,
        "lstm_best_epoch":    best_epoch,
        "lstm_test_acc":      test_m["accuracy"],
        "lstm_test_loss":     test_m["loss"],
    }


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion(preds, labels, class_names, outpath: Path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("LSTM — Normalised Confusion Matrix (Test Set)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved → {outpath}")
    except Exception as e:
        print(f"  [plot_confusion] skipped: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = setup_run(args)
    print(f"\nOutput directory: {outdir}\n")

    _, feat_df, train_df, val_df, test_df, _, _ = prepare_data(args)

    all_metrics = {}

    if args.model in ("markov", "both"):
        m_metrics = run_markov(train_df, val_df, test_df, outdir)
        all_metrics.update(m_metrics)

    if args.model in ("lstm", "both"):
        l_metrics = run_lstm(train_df, val_df, test_df, args, outdir)
        all_metrics.update(l_metrics)

    # Save all metrics
    with open(outdir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("All metrics saved →", outdir / "metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
