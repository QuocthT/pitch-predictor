"""
transfer_train.py
=================
Three-phase transfer learning pipeline: MLB → Ekstraliga.

Phase 1  MLB Pretraining
  Train PitchLSTM from scratch on large MLB dataset.
  The shared pitch-token vocabulary and tendency feature schema mean the
  same model architecture applies to both datasets unchanged.

Phase 2  Ekstraliga Fine-tuning (frozen encoder)
  Load MLB checkpoint. Freeze embedding + LSTM encoder.
  Only the tendency MLP and classifier head are updated.
  Any MLB-only physical features (velocity, spin rate) are zero-masked.

Phase 3  Full Fine-tuning (optional, --full_finetune)
  Unfreeze encoder. Train all parameters with a small learning rate.
  This is the final Ekstraliga-optimised model used by decision_support.py.

Baseline
  For comparison, a separate model is trained from scratch on Ekstraliga only.
  The delta between baseline and Phase 2/3 quantifies the transfer learning gain.

Usage
-----
  # First pull MLB data:
  python src/mlb_api.py --season 2024 --n_games 500 --outdir data/mlb/

  # Then run transfer learning:
  python src/transfer_train.py \\
      --ekstra_data   data/ekstraliga/2025_Ekstraliga_Stats.xlsx \\
      --mlb_dir       data/mlb/ \\
      --phase1_epochs 20 \\
      --phase2_epochs 30 \\
      --full_finetune

  # Skip Phase 1 if you already have an MLB checkpoint:
  python src/transfer_train.py \\
      --ekstra_data    data/ekstraliga/2025_Ekstraliga_Stats.xlsx \\
      --mlb_checkpoint runs/transfer_XXXX/mlb_pretrained.pt \\
      --phase2_epochs  30 \\
      --full_finetune
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import build_pa_dataframe
from mlb_preprocessing import build_mlb_pa_dataframe
from features import (
    VOCAB, OUTCOME_CATEGORIES,
    compute_batter_tendencies, compute_pitcher_tendencies,
    build_feature_matrix, PASequenceDataset,
    train_val_test_split,
)
from models.lstm_model import (
    PitchLSTM, compute_class_weights,
    train_one_epoch, evaluate,
)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Transfer learning: MLB → Ekstraliga")
    # Data
    p.add_argument("--ekstra_data",    default="data/ekstraliga/2025_Ekstraliga_Stats.xlsx")
    p.add_argument("--mlb_dir",        default="data/mlb/")
    p.add_argument("--mlb_pkl",        default=None,
                   help="Pre-cached MLB pkl path — skips parquet loading")
    p.add_argument("--mlb_checkpoint", default=None,
                   help="Existing MLB .pt checkpoint — skips Phase 1 entirely")
    # Architecture (must match across all phases)
    p.add_argument("--embed",   type=int,   default=32)
    p.add_argument("--hidden",  type=int,   default=128)
    p.add_argument("--layers",  type=int,   default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    # Phase 1 — MLB pretraining
    p.add_argument("--phase1_epochs", type=int,   default=20)
    p.add_argument("--phase1_lr",     type=float, default=1e-3)
    p.add_argument("--phase1_batch",  type=int,   default=128)
    # Phase 2 — Ekstraliga fine-tuning (frozen encoder)
    p.add_argument("--phase2_epochs", type=int,   default=30)
    p.add_argument("--phase2_lr",     type=float, default=5e-4)
    p.add_argument("--phase2_batch",  type=int,   default=64)
    # Phase 3 — full fine-tuning
    p.add_argument("--full_finetune",  action="store_true")
    p.add_argument("--phase3_epochs",  type=int,   default=15)
    p.add_argument("--phase3_lr",      type=float, default=1e-4)
    # Misc
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_run(args) -> Path:
    ts  = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.outdir) if args.outdir else Path("runs") / f"transfer_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_datasets(pa_df: pd.DataFrame, xlsx_path: str | None = None):
    batter_tend  = compute_batter_tendencies(pa_df)
    pitcher_tend = compute_pitcher_tendencies(pa_df)
    feat_df      = build_feature_matrix(pa_df, batter_tend, pitcher_tend,
                                        xlsx_path=xlsx_path)
    train_df, val_df, test_df = train_val_test_split(feat_df)
    return (
        PASequenceDataset(train_df),
        PASequenceDataset(val_df),
        PASequenceDataset(test_df),
        batter_tend,
        pitcher_tend,
    )


def make_loaders(train_ds, val_ds, test_ds, batch_size: int):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )


def build_model(args, n_tend_feats: int, device: torch.device) -> PitchLSTM:
    return PitchLSTM(
        vocab_size    = len(VOCAB),
        embed_dim     = args.embed,
        hidden_dim    = args.hidden,
        n_layers      = args.layers,
        n_tend_feats  = n_tend_feats,
        tend_proj_dim = 32,
        n_classes     = len(OUTCOME_CATEGORIES),
        dropout       = args.dropout,
    ).to(device)


# ── Generic training phase ────────────────────────────────────────────────────

def run_phase(
    label:            str,
    model:            PitchLSTM,
    train_loader,
    val_loader,
    test_loader,
    n_epochs:         int,
    lr:               float,
    device:           torch.device,
    class_weights:    torch.Tensor,
    outdir:           Path,
    checkpoint_name:  str,
    mask_indices:     list[int] | None = None,
) -> dict:
    """
    Single training loop used by all phases.
    Only parameters with requires_grad=True are passed to the optimizer,
    so freeze_encoder() / unfreeze_encoder() control what is updated.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    params = model.count_params()
    print(f"  Trainable: {params['trainable']:,} / {params['total']:,} parameters")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0.0
    best_epoch   = 0
    history      = []

    for epoch in range(1, n_epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            mask_indices=mask_indices,
        )
        val_m = evaluate(
            model, val_loader, criterion, device,
            mask_indices=mask_indices,
        )
        scheduler.step()

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_m["loss"], 5),
            "train_acc":  round(train_m["accuracy"], 5),
            "val_loss":   round(val_m["loss"], 5),
            "val_acc":    round(val_m["accuracy"], 5),
        })

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch   = epoch
            torch.save(model.state_dict(), outdir / checkpoint_name)

        if epoch % 5 == 0 or epoch == n_epochs:
            print(f"  [{epoch:3d}/{n_epochs}] "
                  f"train_loss={train_m['loss']:.4f}  train_acc={train_m['accuracy']:.4f} | "
                  f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}")

    print(f"\n  Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Reload best checkpoint and score on test set
    model.load_state_dict(torch.load(outdir / checkpoint_name, map_location=device))
    test_m = evaluate(model, test_loader, criterion, device, mask_indices=mask_indices)
    print(f"  Test acc:     {test_m['accuracy']:.4f}")

    hist_path = outdir / checkpoint_name.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    tag = label.lower().replace(" ", "_")
    return {
        f"{tag}_best_val_acc": round(best_val_acc, 5),
        f"{tag}_best_epoch":   best_epoch,
        f"{tag}_test_acc":     round(test_m["accuracy"], 5),
        f"{tag}_test_loss":    round(test_m["loss"], 5),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)
    outdir = setup_run(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nOutput dir : {outdir}")
    print(f"Device     : {device}\n")

    all_metrics: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — MLB Pretraining
    # ─────────────────────────────────────────────────────────────────────────
    if args.mlb_checkpoint:
        print(f"[Phase 1] Skipping — using checkpoint: {args.mlb_checkpoint}")
        mlb_ckpt_path = Path(args.mlb_checkpoint)
    else:
        print("=" * 60)
        print("PHASE 1 — MLB Pretraining")
        print("=" * 60)

        # Load MLB PA data (from cache or parquet)
        if args.mlb_pkl and Path(args.mlb_pkl).exists():
            print(f"  Loading cached MLB pkl: {args.mlb_pkl}")
            mlb_pa_df = pd.read_pickle(args.mlb_pkl)
        else:
            mlb_pa_df = build_mlb_pa_dataframe(args.mlb_dir)
            cache_path = Path(args.mlb_dir) / "mlb_pa_clean.pkl"
            mlb_pa_df.to_pickle(cache_path)
            print(f"  MLB PA DataFrame cached → {cache_path}")

        print(f"\n  MLB: {len(mlb_pa_df):,} plate appearances")

        mlb_train_ds, mlb_val_ds, mlb_test_ds, _, _ = build_datasets(mlb_pa_df)
        mlb_loaders = make_loaders(mlb_train_ds, mlb_val_ds, mlb_test_ds, args.phase1_batch)

        mlb_weights = compute_class_weights(
            mlb_train_ds.df["outcome_label"].tolist(), len(OUTCOME_CATEGORIES)
        )

        model = build_model(args, mlb_train_ds.n_tendency_features, device)
        print(f"  Model parameters: {model.count_params()['total']:,}")

        p1_metrics = run_phase(
            label           = "Phase 1 — MLB Pretraining",
            model           = model,
            train_loader    = mlb_loaders[0],
            val_loader      = mlb_loaders[1],
            test_loader     = mlb_loaders[2],
            n_epochs        = args.phase1_epochs,
            lr              = args.phase1_lr,
            device          = device,
            class_weights   = mlb_weights,
            outdir          = outdir,
            checkpoint_name = "mlb_pretrained.pt",
        )
        all_metrics.update(p1_metrics)
        mlb_ckpt_path = outdir / "mlb_pretrained.pt"

    # ─────────────────────────────────────────────────────────────────────────
    # Load Ekstraliga data (shared across Phase 2, 3, and baseline)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading Ekstraliga data")
    print("=" * 60)

    ek_pa_df = build_pa_dataframe(args.ekstra_data)
    print(f"\n  Ekstraliga: {len(ek_pa_df):,} plate appearances")

    ek_train_ds, ek_val_ds, ek_test_ds, ek_batter_tend, ek_pitcher_tend = \
        build_datasets(ek_pa_df, xlsx_path=args.ekstra_data)
    ek_loaders = make_loaders(ek_train_ds, ek_val_ds, ek_test_ds, args.phase2_batch)
    ek_weights  = compute_class_weights(
        ek_train_ds.df["outcome_label"].tolist(), len(OUTCOME_CATEGORIES)
    )

    # MLB-only physical feature indices to zero-mask during Ekstraliga training.
    # In the current schema both datasets share identical tendency columns, so
    # this list is empty. If you extend the MLB feature set with velocity /
    # spin-rate columns, add their column indices here.
    MLB_ONLY_INDICES: list[int] = []

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Ekstraliga Fine-tuning (frozen encoder)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Ekstraliga Fine-tuning (frozen encoder)")
    print("=" * 60)

    model = build_model(args, ek_train_ds.n_tendency_features, device)
    state = torch.load(mlb_ckpt_path, map_location=device)
    model.load_state_dict(state)

    model.freeze_encoder()
    print(f"\n  Encoder frozen. "
          f"Trainable: {model.count_params()['trainable']:,} / "
          f"{model.count_params()['total']:,} params")

    p2_metrics = run_phase(
        label           = "Phase 2 — Ekstraliga (frozen encoder)",
        model           = model,
        train_loader    = ek_loaders[0],
        val_loader      = ek_loaders[1],
        test_loader     = ek_loaders[2],
        n_epochs        = args.phase2_epochs,
        lr              = args.phase2_lr,
        device          = device,
        class_weights   = ek_weights,
        outdir          = outdir,
        checkpoint_name = "ekstra_frozen.pt",
        mask_indices    = MLB_ONLY_INDICES,
    )
    all_metrics.update(p2_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Full Fine-tuning (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if args.full_finetune:
        print("\n" + "=" * 60)
        print("PHASE 3 — Full Fine-tuning (unfrozen encoder)")
        print("=" * 60)

        model.load_state_dict(
            torch.load(outdir / "ekstra_frozen.pt", map_location=device)
        )
        model.unfreeze_encoder()
        print(f"\n  Encoder unfrozen. "
              f"Trainable: {model.count_params()['trainable']:,} params")

        p3_metrics = run_phase(
            label           = "Phase 3 — Full Fine-tuning",
            model           = model,
            train_loader    = ek_loaders[0],
            val_loader      = ek_loaders[1],
            test_loader     = ek_loaders[2],
            n_epochs        = args.phase3_epochs,
            lr              = args.phase3_lr,
            device          = device,
            class_weights   = ek_weights,
            outdir          = outdir,
            checkpoint_name = "ekstra_full.pt",
            mask_indices    = MLB_ONLY_INDICES,
        )
        all_metrics.update(p3_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # BASELINE — Ekstraliga only (no transfer learning)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE — Ekstraliga from scratch (no transfer)")
    print("=" * 60)

    scratch_model = build_model(args, ek_train_ds.n_tendency_features, device)
    print(f"\n  Parameters: {scratch_model.count_params()['total']:,}")

    base_metrics = run_phase(
        label           = "Baseline — scratch",
        model           = scratch_model,
        train_loader    = ek_loaders[0],
        val_loader      = ek_loaders[1],
        test_loader     = ek_loaders[2],
        n_epochs        = args.phase2_epochs,
        lr              = args.phase2_lr,
        device          = device,
        class_weights   = ek_weights,
        outdir          = outdir,
        checkpoint_name = "scratch_model.pt",
    )
    all_metrics.update(base_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    scratch_acc = all_metrics.get("baseline_—_scratch_test_acc", 0.0)
    frozen_acc  = all_metrics.get("phase_2_—_ekstraliga_(frozen_encoder)_test_acc", 0.0)

    print("\n" + "=" * 60)
    print("TRANSFER LEARNING RESULTS")
    print("=" * 60)
    print(f"  Baseline (scratch)         test acc : {scratch_acc:.4f}")
    print(f"  Phase 2  (frozen encoder)  test acc : {frozen_acc:.4f}")
    print(f"  Transfer gain (frozen)              : {frozen_acc - scratch_acc:+.4f}")

    if args.full_finetune:
        full_acc = all_metrics.get("phase_3_—_full_fine-tuning_test_acc", 0.0)
        print(f"  Phase 3  (full fine-tune)  test acc : {full_acc:.4f}")
        print(f"  Transfer gain (full FT)             : {full_acc - scratch_acc:+.4f}")

    # Save metrics and player tendency tables
    with open(outdir / "transfer_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    from preprocessing import (
        load_cumulative_batter_features,
        load_cumulative_pitcher_features,
    )
    cum_bat = load_cumulative_batter_features(args.ekstra_data)
    cum_pit = load_cumulative_pitcher_features(args.ekstra_data)

    # Save cumulative stats as the tendency tables for decision support
    # Fall back to PA-log stats for any player not in the cumulative sheet
    final_bat = cum_bat.combine_first(ek_batter_tend)
    final_pit = cum_pit.combine_first(ek_pitcher_tend)

    final_bat.to_pickle(outdir / "batter_tendencies.pkl")
    final_pit.to_pickle(outdir / "pitcher_tendencies.pkl")
    print("Saved cumulative-based tendency tables for decision support.")

    print(f"\nAll outputs saved to: {outdir}\n")


if __name__ == "__main__":
    main()