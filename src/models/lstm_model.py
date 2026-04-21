"""
models/lstm_model.py
====================
LSTM-based pitch-sequence model designed for transfer learning:

  MLB stage  → train on large MLB dataset with full feature set
  Ekstraliga → fine-tune on Ekstraliga PA logs with physical features MASKED

Architecture
------------
  1. Token Embedding layer  (vocab_size → embed_dim)
  2. LSTM encoder           (embed_dim → hidden_dim, n_layers)
  3. Tendency MLP bridge    (n_tend_feats → tend_proj_dim)
  4. Classifier head        ([hidden_dim + tend_proj_dim] → n_classes)

Feature masking
---------------
During fine-tuning, set `mask_mlb_features=True` to zero-out any MLB-only
features (velocity, spin rate, etc.) while keeping Ekstraliga-available
features (handedness, count, swing rate, K%, BB%, …).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PitchLSTM(nn.Module):
    """
    Parameters
    ----------
    vocab_size       : int — number of pitch tokens (from Vocabulary)
    embed_dim        : int — pitch token embedding dimension
    hidden_dim       : int — LSTM hidden state dimension
    n_layers         : int — number of LSTM layers
    n_tend_feats     : int — number of tendency / context features
    tend_proj_dim    : int — projection size for tendency features MLP
    n_classes        : int — number of outcome categories (default 9)
    dropout          : float — dropout rate (applied between LSTM layers)
    pad_token_id     : int — index of PAD token (default 0)
    """

    def __init__(
        self,
        vocab_size:    int   = 28,
        embed_dim:     int   = 32,
        hidden_dim:    int   = 128,
        n_layers:      int   = 2,
        n_tend_feats:  int   = 15,
        tend_proj_dim: int   = 32,
        n_classes:     int   = 9,
        dropout:       float = 0.3,
        pad_token_id:  int   = 0,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers
        self.pad_token_id = pad_token_id

        # ── Sequence encoder ─────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_token_id
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.seq_dropout = nn.Dropout(dropout)

        # ── Tendency MLP ──────────────────────────────────────────────────────
        self.tend_proj = nn.Sequential(
            nn.Linear(n_tend_feats, tend_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + tend_proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, seq_len)
        tendencies: torch.Tensor,          # (B, n_tend_feats)
        lengths: torch.Tensor | None = None,  # (B,) actual sequence lengths
    ) -> torch.Tensor:
        """
        Returns logits of shape (B, n_classes).

        Parameters
        ----------
        input_ids  : padded token ID tensor
        tendencies : batter / pitcher tendency features
        lengths    : actual (unpadded) sequence lengths; if provided, the last
                     valid hidden state is used (proper handling of variable
                     length). If None, the last timestep is used.
        """
        # Embed
        emb = self.embedding(input_ids)             # (B, L, embed_dim)

        # LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(emb)

        # Take the top layer's hidden state
        seq_repr = self.seq_dropout(h_n[-1])        # (B, hidden_dim)

        # Tendency projection
        tend_repr = self.tend_proj(tendencies)       # (B, tend_proj_dim)

        # Concatenate and classify
        combined = torch.cat([seq_repr, tend_repr], dim=-1)
        logits   = self.classifier(combined)         # (B, n_classes)
        return logits

    # ── Utility methods ───────────────────────────────────────────────────────

    def freeze_encoder(self):
        """Freeze embedding + LSTM (call before Ekstraliga fine-tuning)."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.embedding.parameters():
            param.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        """Freeze classifier head (useful for feature extraction mode)."""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def count_params(self) -> dict[str, int]:
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Training utilities ────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced outcomes.
    Downweights easy examples (common outcomes) and focuses on hard ones.
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p     = torch.exp(log_p)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal   = -((1 - p_t) ** self.gamma) * log_p_t
        return focal.mean()


def compute_class_weights(outcome_labels: list[int], n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss to handle imbalance.
    """
    counts = torch.zeros(n_classes)
    for lbl in outcome_labels:
        if 0 <= lbl < n_classes:
            counts[lbl] += 1
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights


def mask_mlb_features(
    tendencies: torch.Tensor,
    mlb_feature_indices: list[int],
) -> torch.Tensor:
    """
    Zero-out MLB-only physical features during Ekstraliga fine-tuning.

    Parameters
    ----------
    tendencies          : (B, n_tend_feats) tensor
    mlb_feature_indices : list of column indices to mask (velocity, spin, etc.)
                          These are 0-indexed into the tendency feature vector.

    In our current Ekstraliga-only setup mlb_feature_indices will be empty []
    because we have no physical features to start with.  When pre-training on
    MLB, you would add indices corresponding to velo / spin / release columns.
    """
    if not mlb_feature_indices:
        return tendencies
    masked = tendencies.clone()
    masked[:, mlb_feature_indices] = 0.0
    return masked


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: PitchLSTM,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mask_indices: list[int] | None = None,
    clip_grad: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for input_ids, tendencies, labels in loader:
        input_ids  = input_ids.to(device)
        tendencies = tendencies.to(device)
        labels     = labels.to(device)

        if mask_indices:
            tendencies = mask_mlb_features(tendencies, mask_indices)

        # Compute actual lengths for pack_padded_sequence
        lengths = (input_ids != 0).sum(dim=1).clamp(min=1)

        optimizer.zero_grad()
        logits = model(input_ids, tendencies, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)

    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: PitchLSTM,
    loader,
    criterion: nn.Module,
    device: torch.device,
    mask_indices: list[int] | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    all_preds  = []
    all_labels = []

    for input_ids, tendencies, labels in loader:
        input_ids  = input_ids.to(device)
        tendencies = tendencies.to(device)
        labels     = labels.to(device)

        if mask_indices:
            tendencies = mask_mlb_features(tendencies, mask_indices)

        lengths = (input_ids != 0).sum(dim=1).clamp(min=1)
        logits  = model(input_ids, tendencies, lengths)
        loss    = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "loss":      total_loss / total,
        "accuracy":  correct / total,
        "preds":     all_preds,
        "labels":    all_labels,
    }
