"""
mlb_preprocessing.py
====================
Convert raw MLB parquet files (produced by mlb_api.py) into the same
DataFrame schema as the Ekstraliga PA logs, so the shared feature-engineering
pipeline (features.py) can ingest both datasets identically.

Usage (standalone)
------------------
  python src/mlb_preprocessing.py --mlb_dir data/mlb/ --out data/mlb/mlb_pa_clean.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

# Mirror the same OUTCOME_MAP from preprocessing.py
OUTCOME_MAP = {
    "Ks": "K",  "Kc": "K",
    "BB": "BB", "IBB": "BB",
    "HBP": "HBP",
    "GO": "OUT", "FO": "OUT", "LO": "OUT",
    "SAC B": "OUT", "SAC FO": "OUT",
    "ROE": "REACH", "FC": "REACH", "OOB": "REACH",
    "1B": "HIT",
    "2B": "XBH", "3B": "XBH",
    "HR": "HR",
}

def _coerce_to_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(val)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    return []

def load_mlb_parquets(mlb_dir: str) -> pd.DataFrame:
    """Load all parquet files from the MLB data directory and concatenate."""
    p = Path(mlb_dir)
    files = sorted(p.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {mlb_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[mlb_preprocessing] Loaded {len(df):,} raw PA records from {len(files)} file(s)")
    return df


def clean_mlb_pa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise raw MLB parquet rows into the Ekstraliga-compatible schema.

    Input columns (from mlb_api.py parse_pitch_sequence):
      batter, batter_hand, pitcher, pitcher_hand, sequence (list[str]),
      seq_len, result, result_category, balls_final, strikes_final,
      source, game_pk, date

    Output schema matches build_pa_dataframe() in preprocessing.py exactly.
    """
    df = df.copy()

    # Drop rows with missing or empty sequences
    df["sequence"] = df["sequence"].apply(_coerce_to_list)
    mask = df["sequence"].apply(lambda s: len(s) > 0)
    df = df[mask].reset_index(drop=True)

    # Re-derive result_category from result token via OUTCOME_MAP
    # (mlb_api.py sometimes leaves this as None for unknown events)
    df["result_category"] = df["result"].map(OUTCOME_MAP).fillna("OTHER")

    # Recompute seq_len to be safe
    df["seq_len"] = df["sequence"].apply(len)

    # Normalise date column — mlb_api.py attaches date from the schedule dict
    if "date" not in df.columns:
        df["date"] = "2024-01-01"
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["date"] = df["date"].fillna("2024-01-01")

    # Build game_id from game_pk
    if "game_pk" in df.columns:
        df["game_id"] = df["game_pk"].astype(str)
    elif "game_id" not in df.columns:
        df["game_id"] = ""

    # Add placeholder columns that Ekstraliga has but MLB does not
    for col in ["team", "opponent"]:
        if col not in df.columns:
            df[col] = ""

    # Ensure source tag
    df["source"] = "mlb"

    # Add pa_id
    df = df.reset_index(drop=True)
    df["pa_id"] = df.index

    # Drop rows with null batter/pitcher/result
    df = df.dropna(subset=["batter", "pitcher", "result_category"])

    # Select and order columns to exactly match Ekstraliga schema
    cols = [
        "pa_id", "date", "team", "opponent", "game_id",
        "batter", "batter_hand", "pitcher", "pitcher_hand",
        "sequence", "seq_len", "result", "result_category",
        "balls_final", "strikes_final", "source",
    ]
    df = df[[c for c in cols if c in df.columns]]

    print(f"[mlb_preprocessing] Cleaned → {len(df):,} plate appearances")
    print(f"  Outcome distribution: {df['result_category'].value_counts().to_dict()}")
    return df


def build_mlb_pa_dataframe(mlb_dir: str) -> pd.DataFrame:
    """Main entry point: load all MLB parquets and return clean PA DataFrame."""
    raw = load_mlb_parquets(mlb_dir)
    return clean_mlb_pa(raw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mlb_dir", default="data/mlb/")
    p.add_argument("--out",     default="data/mlb/mlb_pa_clean.pkl")
    args = p.parse_args()

    df = build_mlb_pa_dataframe(args.mlb_dir)
    df.to_pickle(args.out)
    print(f"Saved → {args.out}")