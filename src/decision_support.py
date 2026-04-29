"""
decision_support.py
===================
In-game substitution recommendation engine for the Ekstraliga decision tool.

Given the current game state (batter, pitcher, partial pitch sequence,
inning, outs, runners), this module:

  1. Predicts outcome probability distributions for any batter vs. pitcher
     matchup using the trained PitchLSTM.
  2. Ranks available pinch-hitter candidates by expected value (EV).
  3. Ranks available relief-pitcher candidates by EV.
  4. Generates a full matchup report with an ASCII probability breakdown.

Expected value uses run-expectancy-aligned weights so recommendations are
grounded in real baseball economics, not raw accuracy.

Usage (CLI)
-----------
  # Pinch-hitter recommendation
  python src/decision_support.py \\
      --checkpoint   runs/transfer_XXXX/ekstra_full.pt \\
      --batter_tend  runs/transfer_XXXX/batter_tendencies.pkl \\
      --pitcher_tend runs/transfer_XXXX/pitcher_tendencies.pkl \\
      --mode         pinch_hitter \\
      --current_pitcher "Jakub Kowalski" \\
      --candidates   "Adam Nowak,Piotr Wróblewski,Tomasz Kowalczyk" \\
      --sequence     B Sw

  # Relief pitcher recommendation
  python src/decision_support.py \\
      --checkpoint   runs/transfer_XXXX/ekstra_full.pt \\
      --batter_tend  runs/transfer_XXXX/batter_tendencies.pkl \\
      --pitcher_tend runs/transfer_XXXX/pitcher_tendencies.pkl \\
      --mode         relief_pitcher \\
      --current_batter "Jan Wiśniewski" \\
      --batter_hand  LHB \\
      --candidates   "Marek Zieliński,Krzysztof Dąbrowski" \\
      --inning 7 --outs 1 --runners 1--

  # Full matchup report
  python src/decision_support.py \\
      --checkpoint   runs/transfer_XXXX/ekstra_full.pt \\
      --batter_tend  runs/transfer_XXXX/batter_tendencies.pkl \\
      --pitcher_tend runs/transfer_XXXX/pitcher_tendencies.pkl \\
      --mode         matchup \\
      --current_batter  "Jan Wiśniewski" \\
      --current_pitcher "Jakub Kowalski" \\
      --sequence B Sc Sw

Usage (Python API)
------------------
  from decision_support import DecisionSupportEngine

  engine = DecisionSupportEngine.from_checkpoint(
      checkpoint_path   = "runs/.../ekstra_full.pt",
      batter_tend_path  = "runs/.../batter_tendencies.pkl",
      pitcher_tend_path = "runs/.../pitcher_tendencies.pkl",
  )

  # Evaluate a single matchup mid-PA
  report = engine.game_state_report(
      batter           = "Jan Wiśniewski",
      pitcher          = "Jakub Kowalski",
      partial_sequence = ["B", "Sc", "Sw"],
      inning=7, outs=1, runners="1--",
  )
  print(report["summary"])

  # Rank pinch hitters
  ranking = engine.recommend_pinch_hitter(
      current_pitcher  = "Jakub Kowalski",
      candidates       = ["Adam Nowak", "Piotr Wróblewski"],
      partial_sequence = ["B"],
  )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from features import VOCAB, OUTCOME_CATEGORIES, PASequenceDataset
from models.lstm_model import PitchLSTM


# ── Outcome weights ───────────────────────────────────────────────────────────
# Run-expectancy-aligned values per outcome event (approximate MLB linear weights).
# Positive = beneficial for the batter; negative = harmful.
BATTER_OUTCOME_WEIGHTS: dict[str, float] = {
    "K":     -0.30,
    "BB":    +0.30,
    "HBP":   +0.33,
    "OUT":   -0.27,
    "REACH": +0.20,
    "HIT":   +0.47,
    "XBH":   +0.76,
    "HR":    +1.40,
    "OTHER": -0.10,
}

# Pitcher perspective: simply negate (pitcher wants batter EV to be low)
PITCHER_OUTCOME_WEIGHTS: dict[str, float] = {
    k: -v for k, v in BATTER_OUTCOME_WEIGHTS.items()
}


class DecisionSupportEngine:
    """
    Wraps a trained PitchLSTM with player tendency tables to give
    in-game substitution recommendations.

    Parameters
    ----------
    model         : trained PitchLSTM (eval mode)
    batter_tend   : DataFrame indexed by batter name
                    (output of compute_batter_tendencies)
    pitcher_tend  : DataFrame indexed by pitcher name
                    (output of compute_pitcher_tendencies)
    vocab         : Vocabulary instance (default: global VOCAB)
    device        : torch.device for inference
    """

    # Must exactly match PASequenceDataset.TENDENCY_COLS
    TENDENCY_COLS = PASequenceDataset.TENDENCY_COLS

    def __init__(
        self,
        model:        PitchLSTM,
        batter_tend:  pd.DataFrame,
        pitcher_tend: pd.DataFrame,
        vocab=VOCAB,
        device: torch.device | None = None,
    ):
        self.model        = model
        self.batter_tend  = batter_tend
        self.pitcher_tend = pitcher_tend
        self.vocab        = vocab
        self.device       = device or torch.device("cpu")

        self.model.eval()
        self.model.to(self.device)

        # Pre-compute median fallback rows for unknown players
        self._batter_median  = batter_tend.median()
        self._pitcher_median = pitcher_tend.median()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path:   str,
        batter_tend_path:  str,
        pitcher_tend_path: str,
        model_kwargs:      dict | None = None,
        device:            str = "cpu",
    ) -> "DecisionSupportEngine":
        """
        Convenience constructor — loads everything from file paths.

        model_kwargs overrides default PitchLSTM constructor arguments.
        Only needed if you trained with non-default hyperparameters.
        """
        dev = torch.device(device)

        batter_tend  = pd.read_pickle(batter_tend_path)
        pitcher_tend = pd.read_pickle(pitcher_tend_path)

        defaults = {
            "vocab_size":    len(VOCAB),
            "embed_dim":     32,
            "hidden_dim":    128,
            "n_layers":      2,
            "n_tend_feats":  len(cls.TENDENCY_COLS),
            "tend_proj_dim": 32,
            "n_classes":     len(OUTCOME_CATEGORIES),
            "dropout":       0.0,   # no dropout at inference time
        }
        if model_kwargs:
            defaults.update(model_kwargs)

        model = PitchLSTM(**defaults)
        state = torch.load(checkpoint_path, map_location=dev)
        model.load_state_dict(state)
        model.eval()

        return cls(model, batter_tend, pitcher_tend, device=dev)

    # ── Tendency vector ───────────────────────────────────────────────────────

    def _tendency_vector(
        self,
        batter:       str,
        pitcher:      str,
        batter_hand:  str = "RHB",
        pitcher_hand: str = "RHP",
    ) -> torch.Tensor:
        """
        Build the 15-dimensional tendency feature vector for a matchup.
        Unknown players fall back to dataset-wide median values.
        """
        b = (
            self.batter_tend.loc[batter]
            if batter in self.batter_tend.index
            else self._batter_median
        )
        p = (
            self.pitcher_tend.loc[pitcher]
            if pitcher in self.pitcher_tend.index
            else self._pitcher_median
        )

        batter_rhb  = 1.0 if batter_hand  == "RHB" else 0.0
        pitcher_rhp = 1.0 if pitcher_hand == "RHP" else 0.0
        platoon_adv = 1.0 if (
            (batter_hand == "RHB" and pitcher_hand == "LHP") or
            (batter_hand == "LHB" and pitcher_hand == "RHP")
        ) else 0.0

        feats = [
            b.get("k_rate",              self._batter_median.get("k_rate", 0.0)),
            b.get("bb_rate",             self._batter_median.get("bb_rate", 0.0)),
            b.get("hit_rate",            self._batter_median.get("hit_rate", 0.0)),
            b.get("xbh_rate",            self._batter_median.get("xbh_rate", 0.0)),
            b.get("hr_rate",             self._batter_median.get("hr_rate", 0.0)),
            b.get("avg_pa_len",          self._batter_median.get("avg_pa_len", 4.0)),
            b.get("swing_rate",          self._batter_median.get("swing_rate", 0.0)),
            p.get("pitcher_k_rate",      self._pitcher_median.get("pitcher_k_rate", 0.0)),
            p.get("pitcher_bb_rate",     self._pitcher_median.get("pitcher_bb_rate", 0.0)),
            p.get("pitcher_hit_allow",   self._pitcher_median.get("pitcher_hit_allow", 0.0)),
            p.get("pitcher_avg_pa_len",  self._pitcher_median.get("pitcher_avg_pa_len", 4.0)),
            p.get("first_pitch_strike_rt",
                  self._pitcher_median.get("first_pitch_strike_rt", 0.0)),
            batter_rhb,
            pitcher_rhp,
            platoon_adv,
        ]
        return torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(self.device)

    # ── Core prediction ───────────────────────────────────────────────────────

    def predict_outcome_proba(
        self,
        batter:           str,
        pitcher:          str,
        partial_sequence: list[str] | None = None,
        batter_hand:      str = "RHB",
        pitcher_hand:     str = "RHP",
        max_len:          int = 11,
    ) -> dict[str, float]:
        """
        Predict outcome probability distribution for a matchup.

        Parameters
        ----------
        batter, pitcher     : player names; unknown names use median tendencies
        partial_sequence    : pitch tokens seen so far (e.g. ['B', 'Sw'])
                              — pass None or [] for start-of-PA prediction
        batter_hand         : 'RHB' or 'LHB'
        pitcher_hand        : 'RHP' or 'LHP'
        max_len             : sequence padding length (match training max_len)

        Returns
        -------
        dict[outcome_category → probability]
        """
        seq = partial_sequence or []

        # Encode and pad sequence
        ids = self.vocab.encode_sequence(seq)[:max_len]
        ids = ids + [self.vocab.pad_id] * (max_len - len(ids))
        input_ids  = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        tendencies = self._tendency_vector(batter, pitcher, batter_hand, pitcher_hand)

        with torch.no_grad():
            logits = self.model(input_ids, tendencies)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return {cat: float(p) for cat, p in zip(OUTCOME_CATEGORIES, probs)}

    def expected_value(
        self,
        proba:       dict[str, float],
        perspective: str = "batter",
    ) -> float:
        """
        Compute the run-expectancy expected value from a probability dict.

        perspective : 'batter'  → weight outcomes by batter benefit
                      'pitcher' → weight outcomes by pitcher benefit
        """
        weights = (
            BATTER_OUTCOME_WEIGHTS if perspective == "batter"
            else PITCHER_OUTCOME_WEIGHTS
        )
        return sum(proba.get(cat, 0.0) * w for cat, w in weights.items())

    # ── Recommendations ───────────────────────────────────────────────────────

    def recommend_pinch_hitter(
        self,
        current_pitcher:  str,
        candidates:       list[str],
        partial_sequence: list[str] | None = None,
        pitcher_hand:     str = "RHP",
        top_k:            int | None = None,
    ) -> list[dict]:
        """
        Rank pinch-hitter candidates against the current pitcher by batter EV.

        Returns a list of dicts (sorted descending by EV):
          rank, batter, hand, ev, hit_prob, k_prob, bb_prob, proba, recommendation
        """
        results = []
        for batter in candidates:
            batter_hand = self._infer_batter_hand(batter)
            proba = self.predict_outcome_proba(
                batter, current_pitcher, partial_sequence,
                batter_hand=batter_hand, pitcher_hand=pitcher_hand,
            )
            ev = self.expected_value(proba, perspective="batter")
            results.append({
                "batter":   batter,
                "hand":     batter_hand,
                "ev":       round(ev, 4),
                "hit_prob": round(
                    proba.get("HIT", 0) + proba.get("XBH", 0) + proba.get("HR", 0), 3
                ),
                "k_prob":   round(proba.get("K", 0), 3),
                "bb_prob":  round(proba.get("BB", 0), 3),
                "proba":    {k: round(v, 4) for k, v in proba.items()},
            })

        results.sort(key=lambda x: x["ev"], reverse=True)
        if top_k:
            results = results[:top_k]
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
            r["recommendation"] = "✓ RECOMMENDED" if rank == 1 else ""
        return results

    def recommend_relief_pitcher(
        self,
        current_batter:   str,
        candidates:       list[str],
        partial_sequence: list[str] | None = None,
        batter_hand:      str = "RHB",
        top_k:            int | None = None,
    ) -> list[dict]:
        """
        Rank relief pitcher candidates against the current batter by pitcher EV.

        Returns a list of dicts (sorted descending by pitcher EV):
          rank, pitcher, hand, ev, k_prob, hit_allow, bb_prob, proba, recommendation
        """
        results = []
        for pitcher in candidates:
            pitcher_hand = self._infer_pitcher_hand(pitcher)
            proba = self.predict_outcome_proba(
                current_batter, pitcher, partial_sequence,
                batter_hand=batter_hand, pitcher_hand=pitcher_hand,
            )
            ev = self.expected_value(proba, perspective="pitcher")
            results.append({
                "pitcher":    pitcher,
                "hand":       pitcher_hand,
                "ev":         round(ev, 4),
                "k_prob":     round(proba.get("K", 0), 3),
                "hit_allow":  round(
                    proba.get("HIT", 0) + proba.get("XBH", 0) + proba.get("HR", 0), 3
                ),
                "bb_prob":    round(proba.get("BB", 0), 3),
                "proba":      {k: round(v, 4) for k, v in proba.items()},
            })

        results.sort(key=lambda x: x["ev"], reverse=True)
        if top_k:
            results = results[:top_k]
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
            r["recommendation"] = "✓ RECOMMENDED" if rank == 1 else ""
        return results

    def game_state_report(
        self,
        batter:           str,
        pitcher:          str,
        partial_sequence: list[str] | None = None,
        batter_hand:      str = "RHB",
        pitcher_hand:     str = "RHP",
        inning:           int = 1,
        outs:             int = 0,
        runners:          str = "---",
    ) -> dict:
        """
        Full game-state prediction report for a single batter vs. pitcher matchup.

        Returns dict with keys:
          batter, pitcher, sequence, inning, outs, runners,
          proba (dict), batter_ev, pitcher_ev, summary (str)
        """
        proba      = self.predict_outcome_proba(
            batter, pitcher, partial_sequence, batter_hand, pitcher_hand
        )
        batter_ev  = self.expected_value(proba, "batter")
        pitcher_ev = self.expected_value(proba, "pitcher")

        seq_str = " → ".join(partial_sequence) if partial_sequence else "(start of PA)"

        lines = [
            f"{'─'*52}",
            f"  MATCHUP : {batter} ({batter_hand}) vs {pitcher} ({pitcher_hand})",
            f"  Context : Inning {inning} | {outs} out(s) | Runners: {runners}",
            f"  Sequence: {seq_str}",
            f"{'─'*52}",
            f"  {'Outcome':<10} {'Probability':>12}  {'Distribution'}",
            f"  {'───────':<10} {'───────────':>12}  {'────────────'}",
        ]
        for cat in OUTCOME_CATEGORIES:
            prob = proba.get(cat, 0.0)
            bar  = "█" * max(1, int(prob * 25)) if prob > 0.01 else ""
            lines.append(f"  {cat:<10} {prob:>11.1%}  {bar}")
        lines += [
            f"{'─'*52}",
            f"  Batter EV  : {batter_ev:+.4f}  "
            f"({'favours batter' if batter_ev > 0 else 'favours pitcher'})",
            f"  Pitcher EV : {pitcher_ev:+.4f}",
            f"{'─'*52}",
        ]

        return {
            "batter":     batter,
            "pitcher":    pitcher,
            "sequence":   partial_sequence or [],
            "inning":     inning,
            "outs":       outs,
            "runners":    runners,
            "proba":      proba,
            "batter_ev":  round(batter_ev, 4),
            "pitcher_ev": round(pitcher_ev, 4),
            "summary":    "\n".join(lines),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_batter_hand(self, batter: str) -> str:
        """
        Attempt to infer batter handedness from the tendency table.
        Falls back to 'RHB' if not stored (tendency table doesn't persist hand).
        Override this method if you add a player-hand lookup table.
        """
        return "RHB"

    def _infer_pitcher_hand(self, pitcher: str) -> str:
        return "RHP"

    def list_known_players(self) -> dict[str, list[str]]:
        """Return all players in the tendency tables."""
        return {
            "batters":  sorted(self.batter_tend.index.tolist()),
            "pitchers": sorted(self.pitcher_tend.index.tolist()),
        }


# ── Pretty printing ───────────────────────────────────────────────────────────

def _print_ranking(ranking: list[dict], title: str):
    key = "batter" if "batter" in ranking[0] else "pitcher"
    hit_key = "hit_prob" if key == "batter" else "hit_allow"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'#':<4} {key.capitalize():<26} {'Hand':<5} "
          f"{'EV':>7}  {'H%':>5}  {'K%':>5}  {'BB%':>5}")
    print(f"  {'─'*4} {'─'*26} {'─'*5} {'─'*7}  {'─'*5}  {'─'*5}  {'─'*5}")
    for r in ranking:
        rec = "  ← RECOMMENDED" if r["rank"] == 1 else ""
        print(
            f"  {r['rank']:<4} {r[key]:<26} {r.get('hand','?'):<5} "
            f"{r['ev']:>+7.4f}  "
            f"{r.get(hit_key, 0)*100:>4.1f}%  "
            f"{r['k_prob']*100:>4.1f}%  "
            f"{r['bb_prob']*100:>4.1f}%"
            f"{rec}"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ekstraliga in-game decision support")
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--batter_tend",      required=True)
    p.add_argument("--pitcher_tend",     required=True)
    p.add_argument("--mode",             required=True,
                   choices=["pinch_hitter", "relief_pitcher", "matchup", "list_players"])
    p.add_argument("--current_pitcher",  default=None)
    p.add_argument("--current_batter",   default=None)
    p.add_argument("--candidates",       default=None,
                   help="Comma-separated player names")
    p.add_argument("--sequence",         nargs="*", default=None,
                   help="Partial sequence tokens e.g. B Sw F")
    p.add_argument("--batter_hand",      default="RHB", choices=["RHB", "LHB"])
    p.add_argument("--pitcher_hand",     default="RHP", choices=["RHP", "LHP"])
    p.add_argument("--inning",           type=int, default=1)
    p.add_argument("--outs",             type=int, default=0)
    p.add_argument("--runners",          default="---")
    p.add_argument("--top_k",            type=int, default=None)
    p.add_argument("--device",           default="cpu")
    args = p.parse_args()

    engine = DecisionSupportEngine.from_checkpoint(
        checkpoint_path   = args.checkpoint,
        batter_tend_path  = args.batter_tend,
        pitcher_tend_path = args.pitcher_tend,
        device            = args.device,
    )

    if args.mode == "list_players":
        known = engine.list_known_players()
        print(f"\nKnown batters  ({len(known['batters'])}):")
        for b in known["batters"]:
            print(f"  {b}")
        print(f"\nKnown pitchers ({len(known['pitchers'])}):")
        for p_ in known["pitchers"]:
            print(f"  {p_}")
        return

    candidates = (
        [c.strip() for c in args.candidates.split("|")]
        if args.candidates else []
    )

    if args.mode == "pinch_hitter":
        if not args.current_pitcher or not candidates:
            sys.exit("ERROR: --current_pitcher and --candidates required")
        ranking = engine.recommend_pinch_hitter(
            current_pitcher  = args.current_pitcher,
            candidates       = candidates,
            partial_sequence = args.sequence,
            pitcher_hand     = args.pitcher_hand,
            top_k            = args.top_k,
        )
        _print_ranking(ranking, f"Pinch Hitter Rankings vs {args.current_pitcher}")

    elif args.mode == "relief_pitcher":
        if not args.current_batter or not candidates:
            sys.exit("ERROR: --current_batter and --candidates required")
        ranking = engine.recommend_relief_pitcher(
            current_batter   = args.current_batter,
            candidates       = candidates,
            partial_sequence = args.sequence,
            batter_hand      = args.batter_hand,
            top_k            = args.top_k,
        )
        _print_ranking(ranking, f"Relief Pitcher Rankings vs {args.current_batter}")

    elif args.mode == "matchup":
        if not args.current_batter or not args.current_pitcher:
            sys.exit("ERROR: --current_batter and --current_pitcher required")
        report = engine.game_state_report(
            batter           = args.current_batter,
            pitcher          = args.current_pitcher,
            partial_sequence = args.sequence,
            batter_hand      = args.batter_hand,
            pitcher_hand     = args.pitcher_hand,
            inning           = args.inning,
            outs             = args.outs,
            runners          = args.runners,
        )
        print(report["summary"])


if __name__ == "__main__":
    main()