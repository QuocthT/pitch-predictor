"""
mlb_api.py
==========
Pull pitch-by-pitch data from the MLB Stats API for model pretraining.

The MLB dataset is used in Phase 1 of the transfer learning pipeline:
  MLB (large, rich features) → pre-train LSTM encoder
  Ekstraliga (small, masked features) → fine-tune classifier head

Usage
-----
  # Pull 500 game sample from 2024 regular season
  python src/mlb_api.py --season 2024 --n_games 500 --outdir data/mlb/

  # Pull from a specific date range
  python src/mlb_api.py --start 2024-06-01 --end 2024-08-31 --outdir data/mlb/

MLB Stats API is free and requires no API key.
Base URL: https://statsapi.mlb.com/api/v1
"""

from __future__ import annotations

import json
import time
import random
import logging
from pathlib import Path
from typing import Generator

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"


def get_schedule(
    season: int,
    start_date: str | None = None,
    end_date:   str | None = None,
    sport_id:   int = 1,          # 1 = MLB
) -> list[dict]:
    """Return list of game metadata dicts for the given season / date range."""
    params = {
        "sportId":   sport_id,
        "season":    season,
        "gameType":  "R",          # Regular season; use "P" for playoffs
        "hydrate":   "team",
    }
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    url  = f"{BASE_URL}/schedule"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            games.append({
                "game_pk":   game["gamePk"],
                "date":      date_block["date"],
                "home_team": game["teams"]["home"]["team"]["name"],
                "away_team": game["teams"]["away"]["team"]["name"],
            })
    return games


def get_live_feed(game_pk: int) -> dict:
    """Fetch the full live feed for a game (all play-by-play data)."""
    url  = f"{BASE_URL.replace('v1', 'v1.1')}/game/{game_pk}/feed/live"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_pitch_sequence(play: dict) -> dict | None:
    """
    Extract a single plate appearance from a live-feed 'allPlays' entry.

    Returns a dict that mirrors the Ekstraliga PA Logs schema so the same
    preprocessing pipeline can ingest both datasets.
    """
    try:
        result_type = play["result"]["type"]
        if result_type != "atBat":
            return None

        batter_name   = play["matchup"]["batter"]["fullName"]
        pitcher_name  = play["matchup"]["pitcher"]["fullName"]
        batter_side   = play["matchup"]["batSide"]["code"]    # L / R
        pitcher_side  = play["matchup"]["pitchHand"]["code"]  # L / R

        # Map to Ekstraliga convention
        batter_hand  = "LHB" if batter_side  == "L" else "RHB"
        pitcher_hand = "LHP" if pitcher_side == "L" else "RHP"

        event        = play["result"].get("event", "")
        description  = play["result"].get("description", "")

        # Translate MLB event to Ekstraliga result token
        result_token = _mlb_event_to_token(event)

        pitches_raw = play.get("pitchIndex", [])
        all_events  = play.get("playEvents", [])

        pitch_tokens = []
        for idx in pitches_raw:
            pe = all_events[idx]
            tok = _mlb_pitch_to_token(
                pe.get("details", {}).get("code", ""),
                pe.get("details", {}).get("description", ""),
            )
            if tok:
                pitch_tokens.append(tok)

        if not pitch_tokens:
            return None

        # Append terminal token (from result)
        if result_token and pitch_tokens[-1] not in _TERMINAL_MLB_TOKENS:
            pitch_tokens.append(result_token)

        # Count the final balls / strikes from the last pitch count state
        count = play.get("count", {})
        balls   = count.get("balls",   0)
        strikes = count.get("strikes", 0)

        return {
            "batter":         batter_name,
            "batter_hand":    batter_hand,
            "pitcher":        pitcher_name,
            "pitcher_hand":   pitcher_hand,
            "sequence":       pitch_tokens,
            "seq_len":        len(pitch_tokens),
            "result":         result_token,
            "result_category": _mlb_event_to_category(event),
            "balls_final":    balls,
            "strikes_final":  strikes,
            "source":         "mlb",
        }

    except (KeyError, IndexError, TypeError):
        return None


# ── Token translation tables ──────────────────────────────────────────────────

_TERMINAL_MLB_TOKENS = {
    "Ks", "Kc", "BB", "IBB", "HBP",
    "GO", "FO", "LO", "ROE", "FC",
    "1B", "2B", "3B", "HR",
    "SAC B", "SAC FO",
}

_MLB_PITCH_CODE_MAP = {
    # code → Ekstraliga token
    "B":  "B",    # ball
    "C":  "Sc",   # called strike
    "S":  "Sw",   # swinging strike
    "F":  "F",    # foul
    "X":  None,   # ball in play — handled via result
    "T":  "F",    # foul tip (treated as foul)
    "L":  "F",    # foul bunt
    "O":  "Sw",   # swinging strike bunt
    "M":  "Sw",   # missed bunt
    "Q":  "Sc",   # pitchout called strike
    "R":  "F",    # foul pitchout
    "P":  "B",    # pitchout
    "I":  "B",    # intentional ball
    "H":  "B",    # hit by pitch (treated as ball; actual HBP via result)
    "N":  None,   # no pitch
    "V":  "B",    # automatic ball
    "Y":  "Sc",   # automatic strike
    "*B": "B",
    "PO": None,   # pickoff
}

def _mlb_pitch_to_token(code: str, desc: str) -> str | None:
    return _MLB_PITCH_CODE_MAP.get(code.strip())


def _mlb_event_to_token(event: str) -> str | None:
    e = event.strip().lower()
    MAP = {
        "strikeout":            "Ks",
        "strikeout double play": "Ks",
        "walk":                 "BB",
        "intent walk":          "IBB",
        "hit by pitch":         "HBP",
        "single":               "1B",
        "double":               "2B",
        "triple":               "3B",
        "home run":             "HR",
        "grounded into double play": "GO",
        "ground out":           "GO",
        "groundout":            "GO",
        "fly out":              "FO",
        "flyout":               "FO",
        "line out":             "LO",
        "lineout":              "LO",
        "field error":          "ROE",
        "fielder's choice out": "FC",
        "fielder's choice":     "FC",
        "sac bunt":             "SAC B",
        "sac fly":              "SAC FO",
        "force out":            "GO",
        "pop out":              "FO",
        "bunt groundout":       "GO",
        "bunt pop out":         "FO",
    }
    return MAP.get(e)


def _mlb_event_to_category(event: str) -> str:
    token = _mlb_event_to_token(event) or "OTHER"
    from features import OUTCOME_MAP
    return OUTCOME_MAP.get(token, "OTHER")


# ── Batch fetcher ─────────────────────────────────────────────────────────────

def fetch_games(
    game_pks: list[int],
    delay_sec: float = 0.3,
    max_retries: int = 3,
) -> Generator[dict, None, None]:
    """
    Generator that yields parsed PA dicts for each game.
    Adds polite delay between requests to avoid hammering the API.
    """
    for pk in game_pks:
        for attempt in range(max_retries):
            try:
                feed = get_live_feed(pk)
                plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
                for play in plays:
                    pa = parse_pitch_sequence(play)
                    if pa:
                        pa["game_pk"] = pk
                        yield pa
                break
            except requests.RequestException as e:
                log.warning(f"Game {pk} attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        time.sleep(delay_sec)


def pull_and_save(
    season: int = 2024,
    n_games: int = 500,
    start_date: str | None = None,
    end_date:   str | None = None,
    outdir: str = "data/mlb/",
    seed: int = 42,
):
    """
    Main pipeline: pull n_games randomly sampled from schedule, parse,
    and save to Parquet.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Fetching schedule for {season} season …")
    schedule = get_schedule(season, start_date, end_date)
    log.info(f"Found {len(schedule)} games in schedule")

    random.seed(seed)
    sample = random.sample(schedule, min(n_games, len(schedule)))
    game_pks = [g["game_pk"] for g in sample]

    log.info(f"Pulling {len(game_pks)} games …")
    records = list(fetch_games(game_pks))
    log.info(f"Parsed {len(records)} plate appearances")

    df = pd.DataFrame(records)
    outpath = out / f"mlb_{season}_{len(game_pks)}games.parquet"
    df.to_parquet(outpath, index=False)
    log.info(f"Saved → {outpath}")

    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--season",  type=int, default=2024)
    p.add_argument("--n_games", type=int, default=500)
    p.add_argument("--start",   default=None, help="YYYY-MM-DD")
    p.add_argument("--end",     default=None, help="YYYY-MM-DD")
    p.add_argument("--outdir",  default="data/mlb/")
    args = p.parse_args()

    pull_and_save(
        season     = args.season,
        n_games    = args.n_games,
        start_date = args.start,
        end_date   = args.end,
        outdir     = args.outdir,
    )
