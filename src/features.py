"""
features.py
===========
Transform the clean PA DataFrame produced by preprocessing.py into
numeric features consumable by model training.

Two main outputs:
  1. Vocabulary + token encoder for pitch sequences (for LSTM / Transformer)
  2. Player-tendency features (K%, BB%, swing rate, handedness) joinable
     onto each PA row

Design note
-----------
We use integer token IDs for sequence models and one-hot / scalar encodings
for tree / logistic baselines. A shared `PAFeatureEncoder` handles both.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Vocabulary ────────────────────────────────────────────────────────────────

# Special tokens
PAD   = "<PAD>"    # 0  — sequence padding
SOS   = "<SOS>"    # 1  — start-of-sequence (prepended during teaching forcing)
EOS   = "<EOS>"    # 2  — end-of-sequence / terminal
UNK   = "<UNK>"    # 3  — unseen token at inference time

BASE_VOCAB = [PAD, SOS, EOS, UNK]

# All known pitch tokens (mid-PA + terminal)
PITCH_VOCABULARY = [
    "B", "F", "Sw", "Sc",             # mid-PA
    "Ks", "Kc", "BB", "IBB",          # K / BB
    "HBP",                             # HBP
    "GO", "FO", "LO",                  # outs-in-play
    "ROE", "FC", "OOB",               # other reach
    "1B", "2B", "3B", "HR",           # hits
    "SAC B", "SAC FO",                 # sac
]

FULL_VOCAB = BASE_VOCAB + PITCH_VOCABULARY

# Outcome categories (prediction targets)
OUTCOME_CATEGORIES = ["K", "BB", "HBP", "OUT", "REACH", "HIT", "XBH", "HR", "OTHER"]


class Vocabulary:
    """Bi-directional token ↔ integer mapping."""

    def __init__(self, tokens: list[str] = FULL_VOCAB):
        self.token2id: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.id2token: dict[int, str] = {i: t for i, t in enumerate(tokens)}
        self.pad_id   = self.token2id[PAD]
        self.sos_id   = self.token2id[SOS]
        self.eos_id   = self.token2id[EOS]
        self.unk_id   = self.token2id[UNK]

    def encode(self, token: str) -> int:
        return self.token2id.get(token, self.unk_id)

    def decode(self, idx: int) -> str:
        return self.id2token.get(idx, UNK)

    def encode_sequence(self, seq: list[str]) -> list[int]:
        return [self.encode(t) for t in seq]

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


VOCAB = Vocabulary()


# ── Player tendency features ──────────────────────────────────────────────────

def compute_batter_tendencies(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-batter aggregate tendencies from the PA DataFrame.

    Features returned (indexed by batter name):
      k_rate, bb_rate, hit_rate, xbh_rate, hr_rate, swing_contact_prox,
      avg_pa_len, n_pa (sample size)
    """
    rows = []
    for batter, grp in pa_df.groupby("batter"):
        n = len(grp)
        k_rate   = (grp["result_category"] == "K").mean()
        bb_rate  = (grp["result_category"] == "BB").mean()
        hit_rate = grp["result_category"].isin(["HIT", "XBH", "HR"]).mean()
        xbh_rate = grp["result_category"].isin(["XBH", "HR"]).mean()
        hr_rate  = (grp["result_category"] == "HR").mean()
        avg_len  = grp["seq_len"].mean()

        # Swing proxy: fraction of pitches that were Sw or F
        total_pitches = sum(len(s) for s in grp["sequence"])
        swing_pitches = sum(
            sum(1 for tok in s if tok in {"Sw", "F"}) for s in grp["sequence"]
        )
        swing_rate = swing_pitches / total_pitches if total_pitches > 0 else 0.0

        rows.append({
            "batter":    batter,
            "k_rate":    k_rate,
            "bb_rate":   bb_rate,
            "hit_rate":  hit_rate,
            "xbh_rate":  xbh_rate,
            "hr_rate":   hr_rate,
            "avg_pa_len": avg_len,
            "swing_rate": swing_rate,
            "batter_n_pa": n,
        })
    return pd.DataFrame(rows).set_index("batter")


def compute_pitcher_tendencies(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-pitcher aggregate tendencies.

    Features: k_rate, bb_rate, hits_allowed_rate, avg_pa_len,
              first_pitch_strike_rate, n_pa
    """
    rows = []
    for pitcher, grp in pa_df.groupby("pitcher"):
        n = len(grp)
        k_rate     = (grp["result_category"] == "K").mean()
        bb_rate    = (grp["result_category"] == "BB").mean()
        hit_allow  = grp["result_category"].isin(["HIT", "XBH", "HR"]).mean()
        avg_len    = grp["seq_len"].mean()

        fps = sum(
            1 for s in grp["sequence"]
            if len(s) > 0 and s[0] in {"Sc", "Ks", "Sw", "F"}
        ) / n

        rows.append({
            "pitcher":               pitcher,
            "pitcher_k_rate":        k_rate,
            "pitcher_bb_rate":       bb_rate,
            "pitcher_hit_allow":     hit_allow,
            "pitcher_avg_pa_len":    avg_len,
            "first_pitch_strike_rt": fps,
            "pitcher_n_pa":          n,
        })
    return pd.DataFrame(rows).set_index("pitcher")


def build_feature_matrix(
    pa_df: pd.DataFrame,
    batter_tend: pd.DataFrame,
    pitcher_tend: pd.DataFrame,
    xlsx_path: str | None = None,
) -> pd.DataFrame:
    df = pa_df.copy()

    # Override with cumulative sheet stats if path provided
    if xlsx_path:
        from preprocessing import (
            load_cumulative_batter_features,
            load_cumulative_pitcher_features,
        )
        cum_bat = load_cumulative_batter_features(xlsx_path)
        cum_pit = load_cumulative_pitcher_features(xlsx_path)
        batter_tend  = cum_bat.combine_first(batter_tend)
        pitcher_tend = cum_pit.combine_first(pitcher_tend)

    # Join batter tendencies
    df = df.join(batter_tend, on="batter", how="left")

    # Join pitcher tendencies
    df = df.join(pitcher_tend, on="pitcher", how="left")

    # Handedness one-hot
    df["batter_rhb"] = (df["batter_hand"] == "RHB").astype(int)
    df["pitcher_rhp"] = (df["pitcher_hand"] == "RHP").astype(int)
    df["platoon_adv"] = (
        ((df["batter_hand"] == "RHB") & (df["pitcher_hand"] == "LHP")) |
        ((df["batter_hand"] == "LHB") & (df["pitcher_hand"] == "RHP"))
    ).astype(int)

    # Outcome label (integer)
    df["outcome_label"] = df["result_category"].map(
        {cat: i for i, cat in enumerate(OUTCOME_CATEGORIES)}
    ).fillna(len(OUTCOME_CATEGORIES) - 1).astype(int)

    return df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PASequenceDataset(Dataset):
    """
    Returns (input_ids, tendency_features, outcome_label) for each PA.

    input_ids       : LongTensor [seq_len]  — token IDs for the sequence
    tendency_feats  : FloatTensor [n_feats] — batter/pitcher scalar features
    outcome_label   : LongTensor []         — class index for PA result category
    """

    TENDENCY_COLS = [
        "k_rate", "bb_rate", "hit_rate", "xbh_rate", "hr_rate",
        "avg_pa_len", "swing_rate",
        "pitcher_k_rate", "pitcher_bb_rate", "pitcher_hit_allow",
        "pitcher_avg_pa_len", "first_pitch_strike_rt",
        "batter_rhb", "pitcher_rhp", "platoon_adv",
    ]

    def __init__(
        self,
        feature_df: pd.DataFrame,
        vocab: Vocabulary = VOCAB,
        max_len: int = 11,
        use_prefix_only: bool = False,
        prefix_len: int | None = None,
    ):
        """
        Parameters
        ----------
        feature_df      : output of build_feature_matrix()
        vocab           : Vocabulary instance
        max_len         : maximum sequence length (pad / truncate to this)
        use_prefix_only : if True, feed only the first `prefix_len` pitches
                          to simulate in-game prediction mid-PA
        prefix_len      : number of pitches to use (default: random 1…seq_len)
        """
        self.df           = feature_df.reset_index(drop=True)
        self.vocab        = vocab
        self.max_len      = max_len
        self.use_prefix   = use_prefix_only
        self.prefix_len   = prefix_len

        # Fill NaN tendency features with column medians
        for col in self.TENDENCY_COLS:
            if col not in self.df.columns:
                self.df[col] = 0.0
            self.df[col] = self.df[col].fillna(self.df[col].median()).fillna(0.0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq: list[str] = row["sequence"]

        if self.use_prefix:
            cut = self.prefix_len if self.prefix_len else np.random.randint(1, len(seq) + 1)
            seq = seq[:cut]

        ids = self.vocab.encode_sequence(seq)

        # Pad / truncate
        ids = ids[: self.max_len]
        pad_len = self.max_len - len(ids)
        ids = ids + [self.vocab.pad_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)

        tend_vals = [float(row[c]) for c in self.TENDENCY_COLS]
        tendency  = torch.tensor(tend_vals, dtype=torch.float)

        label = torch.tensor(int(row["outcome_label"]), dtype=torch.long)

        return input_ids, tendency, label

    @property
    def n_tendency_features(self) -> int:
        return len(self.TENDENCY_COLS)


def train_val_test_split(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: train on earliest games, validate + test on later ones.
    This avoids data leakage (future game stats bleeding into training).
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    val_start  = int(n * (1 - val_frac - test_frac))
    test_start = int(n * (1 - test_frac))

    train = df_sorted.iloc[:val_start]
    val   = df_sorted.iloc[val_start:test_start]
    test  = df_sorted.iloc[test_start:]

    print(f"[features] Split → train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
