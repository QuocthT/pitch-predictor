"""
models/markov.py
================
Markov-chain baseline for pitch-sequence prediction.

Two models are implemented:
  1. PitchTransitionMarkov — predicts the *next pitch token* given the
     current count state (balls, strikes).  This is a 2nd-order Markov
     chain: state = (current_token, count).
  2. OutcomeMarkov — predicts the PA *outcome category* given the final
     count state (balls, strikes).  Purely count-state based; no sequential
     memory.

Both expose a scikit-learn-compatible fit() / predict_proba() / predict() API.
"""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import pandas as pd


def _nested_float_dict():
    return defaultdict(float)

def _nested_dict_factory():
    return defaultdict(_nested_float_dict)


# ── helper ────────────────────────────────────────────────────────────────────

def _count_after(token: str) -> tuple[int, int]:
    """
    Return the (balls_delta, strikes_delta) a single pitch token adds.
    Used to simulate count progression.
    """
    balls_delta   = 1 if token == "B" else 0
    strikes_delta = 1 if token in {"Sw", "Sc", "F", "Ks", "Kc"} else 0
    return balls_delta, strikes_delta


def sequence_to_count_states(seq: list[str]) -> list[tuple[int, int]]:
    """
    Walk through a pitch sequence and return the count *before* each pitch.
    E.g.  ['B', 'Sc', 'Sw', 'Ks']  →  [(0,0), (1,0), (1,1), (1,2)]
    The last state is before the terminal pitch (which ends the PA).
    """
    b, s = 0, 0
    states = []
    for tok in seq:
        states.append((b, s))
        db, ds = _count_after(tok)
        b = min(b + db, 3)
        s = min(s + ds, 2)   # strikes cap at 2 (3rd = K or continuation)
    return states


# ── Model 1: Count-state outcome Markov ──────────────────────────────────────

class OutcomeMarkov:
    """
    P(outcome | final_count) estimated from training data.

    State: (balls, strikes) at time of last pitch (0–3, 0–2 → 12 states)
    Output: probability distribution over outcome categories
    """

    OUTCOME_CATEGORIES = [
        "K", "BB", "HBP", "OUT", "REACH", "HIT", "XBH", "HR", "OTHER"
    ]

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing
        # counts[(b,s)][outcome] = count
        self.counts: dict[tuple, dict[str, float]] = defaultdict(
            _nested_float_dict   # already defined at top of the file
        )
        self.fitted = False

    def fit(self, pa_df: pd.DataFrame) -> "OutcomeMarkov":
        """
        Parameters
        ----------
        pa_df : DataFrame with columns ['balls_final', 'strikes_final', 'result_category']
        """
        for _, row in pa_df.iterrows():
            state  = (int(row["balls_final"]), int(row["strikes_final"]))
            outcome = str(row["result_category"])
            self.counts[state][outcome] += 1.0
        self.fitted = True
        return self

    def predict_proba(
        self,
        balls: int,
        strikes: int,
    ) -> dict[str, float]:
        """Return probability dict over outcome categories."""
        assert self.fitted, "Call fit() first."
        state = (balls, strikes)
        cnts = dict(self.counts.get(state, {}))

        # Laplace smoothing
        total = sum(cnts.values()) + self.smoothing * len(self.OUTCOME_CATEGORIES)
        probs = {}
        for cat in self.OUTCOME_CATEGORIES:
            probs[cat] = (cnts.get(cat, 0.0) + self.smoothing) / total
        return probs

    def predict(self, balls: int, strikes: int) -> str:
        probs = self.predict_proba(balls, strikes)
        return max(probs, key=probs.get)

    def evaluate(self, pa_df: pd.DataFrame) -> dict:
        """Compute accuracy and top-1 metrics on a held-out PA DataFrame."""
        correct = 0
        total   = 0
        log_likelihood = 0.0
        for _, row in pa_df.iterrows():
            b, s   = int(row["balls_final"]), int(row["strikes_final"])
            actual = str(row["result_category"])
            probs  = self.predict_proba(b, s)
            pred   = max(probs, key=probs.get)
            correct += int(pred == actual)
            log_likelihood += np.log(probs.get(actual, 1e-8))
            total += 1
        return {
            "accuracy":       correct / total if total > 0 else 0.0,
            "log_likelihood": log_likelihood,
            "n":              total,
        }


# ── Model 2: Transition Markov (pitch-level prediction) ──────────────────────

class PitchTransitionMarkov:
    """
    P(next_token | current_token, count) — 2nd-order Markov chain.

    State: (last_token, balls, strikes)
    Output: probability distribution over next-pitch tokens
    """

    ALL_NEXT_TOKENS = [
        "B", "F", "Sw", "Sc",
        "Ks", "Kc", "BB", "HBP",
        "GO", "FO", "LO", "ROE", "FC",
        "1B", "2B", "3B", "HR",
    ]

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self.counts: dict[tuple, dict[str, float]] = defaultdict(
            _nested_float_dict
        )
        self.fitted = False

    def fit(self, pa_df: pd.DataFrame) -> "PitchTransitionMarkov":
        """
        Parameters
        ----------
        pa_df : DataFrame with column 'sequence' (list of pitch token strings)
        """
        for _, row in pa_df.iterrows():
            seq    = row["sequence"]
            states = sequence_to_count_states(seq)
            for i in range(len(seq) - 1):
                current_tok = seq[i]
                b, s        = states[i]
                next_tok    = seq[i + 1]
                state_key   = (current_tok, b, s)
                self.counts[state_key][next_tok] += 1.0
        self.fitted = True
        return self

    def predict_proba(
        self,
        last_token: str,
        balls: int,
        strikes: int,
    ) -> dict[str, float]:
        assert self.fitted, "Call fit() first."
        state_key = (last_token, balls, strikes)
        cnts = dict(self.counts.get(state_key, {}))
        total = sum(cnts.values()) + self.smoothing * len(self.ALL_NEXT_TOKENS)
        return {
            tok: (cnts.get(tok, 0.0) + self.smoothing) / total
            for tok in self.ALL_NEXT_TOKENS
        }

    def predict(self, last_token: str, balls: int, strikes: int) -> str:
        probs = self.predict_proba(last_token, balls, strikes)
        return max(probs, key=probs.get)

    def evaluate_next_pitch(self, pa_df: pd.DataFrame) -> dict:
        """
        For each non-terminal pitch in validation PAs, predict next token
        and compute accuracy.
        """
        correct = 0
        total   = 0
        for _, row in pa_df.iterrows():
            seq    = row["sequence"]
            states = sequence_to_count_states(seq)
            for i in range(len(seq) - 1):
                b, s      = states[i]
                pred      = self.predict(seq[i], b, s)
                correct  += int(pred == seq[i + 1])
                total    += 1
        return {
            "next_pitch_accuracy": correct / total if total > 0 else 0.0,
            "n_transitions":       total,
        }

    def transition_matrix_df(self) -> pd.DataFrame:
        """
        Marginalise over count states and return a token → token transition
        probability matrix as a DataFrame (useful for visualisation).
        """
        all_tokens = self.ALL_NEXT_TOKENS
        marginal: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for (tok, b, s), nxt_counts in self.counts.items():
            for nxt, cnt in nxt_counts.items():
                marginal[tok][nxt] += cnt

        rows = {}
        for tok in all_tokens:
            cnts  = marginal.get(tok, {})
            total = sum(cnts.values()) + self.smoothing * len(all_tokens)
            rows[tok] = {
                nxt: (cnts.get(nxt, 0.0) + self.smoothing) / total
                for nxt in all_tokens
            }
        return pd.DataFrame(rows).T  # rows=from_token, cols=to_token
