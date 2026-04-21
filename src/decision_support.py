"""
decision_support.py
===================
In-game substitution decision support tool.

Given the current game state (inning, outs, runners on base, current PA context)
and a roster of available substitutes, this module ranks:
  - Pinch-hitter candidates  (probability of reaching base / slugging)
  - Relief pitcher candidates (probability of getting outs / avoiding damage)

It uses:
  1. The trained LSTM model (or Markov fallback) to predict PA outcome
     distributions for each candidate matchup.
  2. Leverage Index-weighted expected run value to rank decisions.

Usage
-----
  from decision_support import DecisionEngine
  engine = DecisionEngine.from_checkpoint("runs/latest/lstm_best.pt")
  recs = engine.recommend(
      situation=GameSituation(inning=7, outs=1, runners=[True, False, True],
                              score_diff=-1),
      available_batters=["Player A", "Player B"],
      current_pitcher="Jones, Tom",
  )
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import (
    VOCAB, Vocabulary, OUTCOME_CATEGORIES,
    PASequenceDataset, compute_batter_tendencies, compute_pitcher_tendencies,
)
from models.lstm_model import PitchLSTM


# ── Run-value table (simplified 24-state model) ───────────────────────────────
# Expected runs from each base-out state (outs × runners).
# Values are approximate averages from MLB; Ekstraliga will be re-estimated
# once enough inning-level game-log data is available.
# Format: RV[outs][runner_state] where runner_state = int(1B)<<2 | int(2B)<<1 | int(3B)

RUN_EXPECTANCY = {
    # outs=0
    0: {0: 0.481, 1: 0.254, 2: 0.783, 3: 0.478, 4: 1.068, 5: 0.908, 6: 1.373, 7: 1.920},
    # outs=1
    1: {0: 0.254, 1: 0.161, 2: 0.478, 3: 0.304, 4: 0.700, 5: 0.543, 6: 0.908, 7: 1.243},
    # outs=2
    2: {0: 0.098, 1: 0.072, 2: 0.208, 3: 0.113, 4: 0.343, 5: 0.263, 6: 0.463, 7: 0.785},
}

# Change in run expectancy per outcome (from/to transitions are simplified here)
OUTCOME_RUN_VALUES = {
    "K":     -0.27,
    "BB":    +0.30,
    "HBP":   +0.30,
    "OUT":   -0.24,
    "REACH": +0.50,
    "HIT":   +0.47,
    "XBH":   +0.77,
    "HR":    +1.40,
    "OTHER": +0.10,
}


# ── Game situation ─────────────────────────────────────────────────────────────

@dataclass
class GameSituation:
    inning:     int           # 1–9+
    outs:       int           # 0, 1, 2
    runners:    list[bool]    # [on_1B, on_2B, on_3B]
    score_diff: int           # batting_team_score - fielding_team_score (neg = losing)
    balls:      int = 0       # current count
    strikes:    int = 0

    @property
    def runner_state(self) -> int:
        return (int(self.runners[0]) << 2 |
                int(self.runners[1]) << 1 |
                int(self.runners[2]))

    @property
    def run_expectancy(self) -> float:
        return RUN_EXPECTANCY.get(self.outs, {}).get(self.runner_state, 0.25)

    @property
    def leverage_index(self) -> float:
        """Crude leverage estimate: late innings, close games, runners on."""
        inning_factor = max(1.0, (self.inning - 5) * 0.3 + 1.0)
        score_factor  = max(0.5, 1.5 - abs(self.score_diff) * 0.2)
        runner_factor = 1.0 + 0.2 * sum(self.runners)
        out_factor    = 1.0 + 0.15 * (2 - self.outs)
        return inning_factor * score_factor * runner_factor * out_factor


# ── Prediction engine ─────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Wraps a trained LSTM (or Markov fallback) and batter/pitcher tendency
    DataFrames to produce ranked substitution recommendations.
    """

    def __init__(
        self,
        model: PitchLSTM | None,
        batter_tend: pd.DataFrame,
        pitcher_tend: pd.DataFrame,
        vocab: Vocabulary = VOCAB,
        device: torch.device | None = None,
    ):
        self.model        = model
        self.batter_tend  = batter_tend
        self.pitcher_tend = pitcher_tend
        self.vocab        = vocab
        self.device       = device or torch.device("cpu")

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        lstm_ckpt: str,
        tendency_pkl: str,
        model_kwargs: dict | None = None,
    ) -> "DecisionEngine":
        """Load from saved checkpoint + tendency DataFrames."""
        with open(tendency_pkl, "rb") as f:
            saved = pickle.load(f)
        batter_tend  = saved["batter_tend"]
        pitcher_tend = saved["pitcher_tend"]

        kwargs = model_kwargs or {}
        model  = PitchLSTM(**kwargs)
        model.load_state_dict(
            torch.load(lstm_ckpt, map_location="cpu")
        )
        return cls(model, batter_tend, pitcher_tend)

    @classmethod
    def from_markov(
        cls,
        markov_pkl: str,
        tendency_pkl: str,
    ) -> "DecisionEngine":
        """Use a fitted OutcomeMarkov instead of LSTM."""
        with open(tendency_pkl, "rb") as f:
            saved = pickle.load(f)
        with open(markov_pkl, "rb") as f:
            markov = pickle.load(f)
        return cls(None, saved["batter_tend"], saved["pitcher_tend"],
                   _markov=markov)

    # ── Core prediction ────────────────────────────────────────────────────────

    def predict_outcome_proba(
        self,
        batter: str,
        pitcher: str,
        current_sequence: list[str],   # pitches seen so far in current PA
        situation: GameSituation,
    ) -> dict[str, float]:
        """
        Return a probability distribution over OUTCOME_CATEGORIES for the
        given batter–pitcher matchup and current PA sequence.
        """
        if self.model is not None:
            return self._lstm_predict(batter, pitcher, current_sequence, situation)
        else:
            return self._tendency_predict(batter, pitcher, situation)

    def _tendency_predict(
        self,
        batter: str,
        pitcher: str,
        situation: GameSituation,
    ) -> dict[str, float]:
        """
        Fallback: blend batter K%/BB% tendency with pitcher K%/BB% tendency
        to produce a rough probability distribution.
        """
        bt = self.batter_tend.loc[batter]  if batter  in self.batter_tend.index  else None
        pt = self.pitcher_tend.loc[pitcher] if pitcher in self.pitcher_tend.index else None

        # Default league-average rates
        k_rate   = 0.20
        bb_rate  = 0.15
        hit_rate = 0.23
        xbh_rate = 0.08
        hr_rate  = 0.04
        out_rate = 1.0 - k_rate - bb_rate - hit_rate - xbh_rate - hr_rate

        if bt is not None:
            k_rate   = (k_rate   + float(bt["k_rate"]))   / 2
            bb_rate  = (bb_rate  + float(bt["bb_rate"]))  / 2
            hit_rate = (hit_rate + float(bt["hit_rate"])) / 2
            xbh_rate = (xbh_rate + float(bt["xbh_rate"])) / 2
            hr_rate  = (hr_rate  + float(bt["hr_rate"]))  / 2

        if pt is not None:
            k_rate   = (k_rate   + float(pt["pitcher_k_rate"]))   / 2
            bb_rate  = (bb_rate  + float(pt["pitcher_bb_rate"]))  / 2
            hit_rate = (hit_rate + float(pt.get("pitcher_hit_allow", hit_rate))) / 2

        probs = {
            "K":     max(k_rate, 0.01),
            "BB":    max(bb_rate, 0.01),
            "HBP":   0.01,
            "OUT":   max(out_rate, 0.01),
            "REACH": 0.03,
            "HIT":   max(hit_rate, 0.01),
            "XBH":   max(xbh_rate, 0.01),
            "HR":    max(hr_rate, 0.001),
            "OTHER": 0.01,
        }
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def _lstm_predict(
        self,
        batter: str,
        pitcher: str,
        current_sequence: list[str],
        situation: GameSituation,
    ) -> dict[str, float]:
        """Run the LSTM and return softmax probability dict."""
        # Encode sequence
        ids = self.vocab.encode_sequence(current_sequence or ["B"])
        ids = ids[:11] + [self.vocab.pad_id] * max(0, 11 - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)

        # Tendency features (use tendency_predict as feature vector)
        bt = self.batter_tend.loc[batter]  if batter  in self.batter_tend.index  else None
        pt = self.pitcher_tend.loc[pitcher] if pitcher in self.pitcher_tend.index else None

        def _get(df, key, default=0.0):
            try:
                return float(df[key]) if df is not None else default
            except Exception:
                return default

        tend_vals = [
            _get(bt, "k_rate",    0.20),
            _get(bt, "bb_rate",   0.15),
            _get(bt, "hit_rate",  0.23),
            _get(bt, "xbh_rate",  0.08),
            _get(bt, "hr_rate",   0.04),
            _get(bt, "avg_pa_len", 3.5),
            _get(bt, "swing_rate", 0.45),
            _get(pt, "pitcher_k_rate",        0.20),
            _get(pt, "pitcher_bb_rate",       0.15),
            _get(pt, "pitcher_hit_allow",     0.25),
            _get(pt, "pitcher_avg_pa_len",    3.5),
            _get(pt, "first_pitch_strike_rt", 0.55),
            float(1),  # batter_rhb (assume R for now)
            float(1),  # pitcher_rhp
            float(0),  # platoon_adv
        ]
        tendencies = torch.tensor([tend_vals], dtype=torch.float).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, tendencies)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        return {cat: p for cat, p in zip(OUTCOME_CATEGORIES, probs)}

    # ── Expected value ─────────────────────────────────────────────────────────

    def expected_run_value(
        self,
        outcome_proba: dict[str, float],
        situation: GameSituation,
    ) -> float:
        """
        Expected change in run expectancy for the batting team, weighted by
        outcome probabilities.
        """
        ev = sum(
            p * OUTCOME_RUN_VALUES.get(cat, 0.0)
            for cat, p in outcome_proba.items()
        )
        return ev * situation.leverage_index

    # ── Main recommendation API ────────────────────────────────────────────────

    def recommend_pinch_hitter(
        self,
        situation: GameSituation,
        available_batters: list[str],
        current_pitcher: str,
        current_sequence: list[str] | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Rank available pinch hitters by expected run value.

        Returns a list of dicts sorted by expected_rv (highest first):
          {batter, outcome_proba, expected_rv, recommendation}
        """
        seq = current_sequence or []
        ranked = []
        for batter in available_batters:
            proba = self.predict_outcome_proba(batter, current_pitcher, seq, situation)
            erv   = self.expected_run_value(proba, situation)
            ranked.append({
                "batter":        batter,
                "outcome_proba": proba,
                "expected_rv":   erv,
            })

        ranked.sort(key=lambda x: x["expected_rv"], reverse=True)

        for i, r in enumerate(ranked):
            p = r["outcome_proba"]
            best_outcome = max(p, key=p.get)
            r["recommendation"] = (
                f"{'★ RECOMMEND' if i == 0 else '  Option ' + str(i+1):15s} | "
                f"ERV={r['expected_rv']:+.3f} | "
                f"Most likely: {best_outcome} ({p[best_outcome]:.0%}) | "
                f"K%={p.get('K',0):.0%}  BB%={p.get('BB',0):.0%}  "
                f"H%={p.get('HIT',0)+p.get('XBH',0)+p.get('HR',0):.0%}"
            )

        return ranked[:top_k]

    def recommend_relief_pitcher(
        self,
        situation: GameSituation,
        upcoming_batters: list[str],
        available_pitchers: list[str],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Rank available relief pitchers by *minimising* expected run value
        allowed across the upcoming lineup.

        Returns a list of dicts sorted by total_erv (lowest first = best pitcher):
          {pitcher, per_batter_proba, total_erv, recommendation}
        """
        ranked = []
        for pitcher in available_pitchers:
            total_erv = 0.0
            per_batter = {}
            for batter in upcoming_batters:
                proba = self.predict_outcome_proba(batter, pitcher, [], situation)
                erv   = self.expected_run_value(proba, situation)
                per_batter[batter] = {"proba": proba, "erv": erv}
                total_erv += erv

            ranked.append({
                "pitcher":         pitcher,
                "per_batter_erv":  per_batter,
                "total_erv":       total_erv,
            })

        ranked.sort(key=lambda x: x["total_erv"])  # lowest ERV allowed = best

        for i, r in enumerate(ranked):
            r["recommendation"] = (
                f"{'★ RECOMMEND' if i == 0 else '  Option ' + str(i+1):15s} | "
                f"Total ERV allowed={r['total_erv']:+.3f} vs {len(upcoming_batters)} batters"
            )

        return ranked[:top_k]

    def print_report(
        self,
        situation: GameSituation,
        ph_recs: list[dict] | None = None,
        rp_recs: list[dict] | None = None,
    ):
        print("\n" + "═" * 70)
        print(f"  DECISION SUPPORT REPORT")
        print(f"  Inning {situation.inning}, {situation.outs} out(s) | "
              f"Score diff: {situation.score_diff:+d} | "
              f"Leverage: {situation.leverage_index:.2f}")
        runners = ["1B" if situation.runners[0] else "",
                   "2B" if situation.runners[1] else "",
                   "3B" if situation.runners[2] else ""]
        print(f"  Runners: {', '.join(r for r in runners if r) or 'bases empty'}")
        print(f"  Run Expectancy: {situation.run_expectancy:.3f}")
        print("═" * 70)

        if ph_recs:
            print("\n  PINCH HITTER CANDIDATES")
            print("  " + "-" * 68)
            for r in ph_recs:
                print(f"  {r['recommendation']}")

        if rp_recs:
            print("\n  RELIEF PITCHER CANDIDATES")
            print("  " + "-" * 68)
            for r in rp_recs:
                print(f"  {r['recommendation']}")

        print("\n" + "═" * 70 + "\n")


# ── Quick demo (no trained model needed) ─────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from preprocessing import build_pa_dataframe

    xlsx = sys.argv[1] if len(sys.argv) > 1 else "data/ekstraliga/2025_Ekstraliga_Stats.xlsx"
    pa_df = build_pa_dataframe(xlsx)
    bt = compute_batter_tendencies(pa_df)
    pt = compute_pitcher_tendencies(pa_df)

    engine = DecisionEngine(
        model=None,
        batter_tend=bt,
        pitcher_tend=pt,
    )

    # Example game situation: bottom of the 7th, 1 out, runner on 2nd, losing by 1
    situation = GameSituation(
        inning=7, outs=1,
        runners=[False, True, False],
        score_diff=-1,
    )

    # Grab real player names from the data
    top_batters  = bt.nlargest(5, "hit_rate").index.tolist()
    top_pitchers = pt.nlargest(5, "pitcher_k_rate").index.tolist()
    opp_batters  = bt.nsmallest(5, "k_rate").index.tolist()

    ph_recs = engine.recommend_pinch_hitter(
        situation=situation,
        available_batters=top_batters,
        current_pitcher=top_pitchers[0] if top_pitchers else "Unknown",
    )

    rp_recs = engine.recommend_relief_pitcher(
        situation=situation,
        upcoming_batters=opp_batters[:3],
        available_pitchers=top_pitchers,
    )

    engine.print_report(situation, ph_recs, rp_recs)
