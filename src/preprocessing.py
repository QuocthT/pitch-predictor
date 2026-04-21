"""
preprocessing.py
================
Load and clean the Ekstraliga PA Logs spreadsheet into structured sequences
suitable for sequence modelling.

Output contract
---------------
Each row of the returned DataFrame represents one plate appearance with:
  - pa_id            : unique integer PA identifier
  - date, team, opponent, game_id
  - batter, batter_hand, pitcher, pitcher_hand
  - sequence         : list of pitch tokens, e.g. ['B','Sw','F','Ks']
  - seq_len          : number of pitches in the PA
  - result           : final PA outcome token
  - result_category  : coarse outcome bucket (K, BB, HBP, OUT, HIT, XBH, HR, OTHER)
  - balls_final, strikes_final
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Token vocabulary ─────────────────────────────────────────────────────────
# Pitch-event tokens that are mid-PA (not terminal)
MID_PITCH_TOKENS = {"B", "F", "Sw", "Sc"}

# Terminal tokens (end the PA) — they appear as the *last* pitch token
TERMINAL_TOKENS = {
    "Ks", "Kc",          # strikeouts
    "BB", "IBB",         # walks
    "HBP",               # hit by pitch
    "GO", "FO", "LO",    # outs in play
    "ROE", "FC",         # reached base (error / fielder's choice)
    "1B", "2B", "3B", "HR",  # hits
    "SAC B", "SAC FO",   # sacrifices
    "OOB",               # other
}

ALL_PITCH_TOKENS = MID_PITCH_TOKENS | TERMINAL_TOKENS

# Coarse outcome categories
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

PITCH_COLS = [f"{n}th Pitch" if n >= 10 else f"{n}{'st' if n==1 else 'nd' if n==2 else 'rd' if n==3 else 'th'} Pitch"
              for n in range(1, 12)]
# Fix: 1st, 2nd, 3rd, then 4th–11th
PITCH_COLS = (
    ["1st Pitch", "2nd Pitch", "3rd Pitch"] +
    [f"{n}th Pitch" for n in range(4, 12)]
)


def load_raw(xlsx_path: str) -> pd.DataFrame:
    """Read PA Logs sheet and return with proper column headers."""
    raw = pd.read_excel(xlsx_path, sheet_name="PA Logs", header=0)
    # Row 0 contains the true headers
    true_header = raw.iloc[0].tolist()
    raw.columns = true_header
    raw = raw.iloc[1:].reset_index(drop=True)
    return raw


def _clean_token(val) -> str | None:
    """Normalise a raw cell value to a pitch token, or None if empty."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("", "nan", "NaN"):
        return None
    return s


def extract_sequence(row: pd.Series) -> list[str]:
    """Pull the ordered pitch sequence from a PA row, stopping at first None."""
    seq = []
    for col in PITCH_COLS:
        if col not in row.index:
            break
        tok = _clean_token(row[col])
        if tok is None:
            break
        # Deduplicate duplicated column names – pandas may have appended .1 etc.
        seq.append(tok)
    return seq


def _resolve_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The PA Logs sheet has multiple columns with the same header (e.g. many
    '1st Pitch' columns for swing indicator, count state, etc.).
    We keep only the FIRST occurrence of each pitch-column name since that
    is the raw token column; the others are derived metrics.
    """
    seen: dict[str, int] = {}
    new_cols = []
    for i, col in enumerate(df.columns):
        if col not in seen:
            seen[col] = i
            new_cols.append(col)
        else:
            new_cols.append(f"_dup_{col}_{i}")
    df.columns = new_cols
    return df


def build_pa_dataframe(xlsx_path: str) -> pd.DataFrame:
    """
    Main entry point. Returns a clean PA-level DataFrame.

    Parameters
    ----------
    xlsx_path : str
        Path to 2025_Ekstraliga_Stats.xlsx

    Returns
    -------
    pd.DataFrame with one row per plate appearance.
    """
    raw = load_raw(xlsx_path)
    raw = _resolve_duplicate_columns(raw)

    records = []
    for pa_id, row in raw.iterrows():
        batter  = str(row.get("Batter",  "")).strip()
        pitcher = str(row.get("Pitcher", "")).strip()
        result  = _clean_token(row.get("Result"))
        date    = str(row.get("Date", "")).strip()
        team    = str(row.get("Team", "")).strip()
        opp     = str(row.get("Opponent", "")).strip()

        # Skip header rows that leaked through, empty rows, or rows without result
        if batter in ("", "Batter", "nan") or result is None:
            continue

        seq = extract_sequence(row)
        if len(seq) == 0:
            continue

        # The last token in sequence is the terminal event; mid-pitch tokens precede it
        # Validate: at least the terminal token must be recognisable
        terminal = seq[-1]
        if terminal not in ALL_PITCH_TOKENS:
            terminal = result  # fall back to Result column

        try:
            balls   = int(row.get("B", 0) or 0)
            strikes = int(row.get("S", 0) or 0)
        except (ValueError, TypeError):
            balls, strikes = 0, 0

        records.append({
            "pa_id":          pa_id,
            "date":           date,
            "team":           team,
            "opponent":       opp,
            "game_id":        str(row.get("Game #", "")).strip(),
            "batter":         batter,
            "batter_hand":    str(row.get("Batter Hand", "")).strip(),
            "pitcher":        pitcher,
            "pitcher_hand":   str(row.get("Pitcher Hand", "")).strip(),
            "sequence":       seq,
            "seq_len":        len(seq),
            "result":         result,
            "result_category": OUTCOME_MAP.get(result, "OTHER"),
            "balls_final":    balls,
            "strikes_final":  strikes,
        })

    df = pd.DataFrame(records)
    print(f"[preprocessing] Loaded {len(df)} plate appearances "
          f"({df['result_category'].value_counts().to_dict()})")
    return df


def load_cumulative_stats(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Hitting Cumulative and Pitching Cumulative sheets for player tendencies.
    Returns (hitting_df, pitching_df).
    """
    hit = pd.read_excel(xlsx_path, sheet_name="Hitting Cumulative", header=0)
    pit = pd.read_excel(xlsx_path, sheet_name="Pitching Cumulative", header=0)
    return hit, pit


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ekstraliga/2025_Ekstraliga_Stats.xlsx"
    df = build_pa_dataframe(path)
    print(df[["batter", "pitcher", "sequence", "result", "result_category"]].head(10).to_string())
    df.to_pickle("data/ekstraliga/pa_clean.pkl")
    print("\nSaved → data/ekstraliga/pa_clean.pkl")
