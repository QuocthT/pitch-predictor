# Pitch Sequence Prediction Model: MLB -> Polish Ekstraliga Transfer Learning

Quoc Dat Nguyen  
MSDS Student 
BUID: U55798231  

## Motivation

During the 2023 World Baseball Classic Final between Team USA and Venezuela, a controversial managerial decision sparked an idea about whether in-game substitution choices — specifically selecting a relief pitcher or pinch-hitter in a high-leverage situation — should be informed by data-driven models. This project aims to build it: a predictive tool that, given a specific game situation (current pitch count, inning, outs, batter/pitcher tendencies), outputs a probability distribution over possible plate appearance outcomes to mathematically assist a manager's decision.

The core challenge is the difference in data itself. Major League Baseball provides huge amount of very detailed data (velocity, spin rate, coordinates, pitch type, etc.), while the Polish Ekstraliga — a growing but rather small amateur league — only tracks basic pitch sequences (Ball, Strike, Foul, Swinging Strike, etc.) with no physical data tracked by sensors. The solution proposed here is to train a base sequence model on large MLB datasets, then use feature masking and transfer learning to adapt the model to be more accurace on the Ekstraliga data.

## Project Objectives

1. Build a pitch-sequence model trained on MLB data,
2. Adapt this model to the Ekstraliga context using transfer learning and feature masking, so it can operate on less informed data (pitch results, counts, and inferred player tendencies only).
3. Produce an in-game decision support tool that can recommend optimal pinch-hitter or relief pitcher substitutions given the current game state.

## Datasets

### Dataset 1: MLB (Primary Training Data)

- MLB Stats API - `https://statsapi.mlb.com`
- Baseball Savant - `https://baseballsavant.mlb.com`
- Millions of records
- Key features
  - Pitch type (fastball, slider, curveball, changeup, etc.)
  - Pitch velocity, spin rate, release point (x, y, z)
  - Plate location coordinates
  - Pitch result (ball, called strike, swinging strike, foul, in-play)
  - Count state (balls, strikes) at time of pitch
  - Batter/pitcher handedness
  - Game context (inning, outs, runners on base)
  - Plate appearance outcome (single, double, strikeout, walk, home run, etc.)

### Dataset 2: Polish Ekstraliga 2025

- Local dataset of year 2025 (`2025_Ekstraliga_Stats.xlsx`)
- The file contains 21 sheets organized into the following categories:

| Sheet Category | Description |

- | Hitting Cumulative / Regular Season / Playoffs | Per-player cumulative batting stats including PA, AVG, OBP, SLG, OPS, K%, BB%, contact rates, pitch count splits (0-0, 0-1, ..., 3-2), and performance vs LHP/RHP |
- | Pitching Cumulative / Regular Season / Playoffs | Per-pitcher stats including ERA, WHIP, FIP, K/BB, strike/ball distribution, CSW%, WHF%, first-pitch strike rate, early count tendencies |
- | Fielding Cumulative / Regular Season / Playoffs | Errors, putouts, assists, fielding %, positional starts |
- | U18 Hitting / Pitching / Fielding | Same structure as above for the under-18 division |
- | Hitting / Pitching / Fielding Game Logs | Game-by-game breakdowns per player |
- | PA Logs | pitch-by-pitch plate appearance logs with pitch sequence (B, F, Sw, Sc, GO, FO, 1B, 2B, 3B, HR, etc.), batter/pitcher handedness, count progression, and final PA result |
- | Glossary | Data dictionary for all abbreviations |

- Plate appearances: around 7000 individual PA records with full pitch sequences
- Pitch types tracked: Ball (B), Foul (F), Swinging Strike (Sw), Called Strike (Sc), Ground Out (GO), Fly Out (FO), Single (1B), Double (2B), Triple (3B), Home Run (HR), Reached on Error (ROE), and other

## Model Survey 

Several model are going to be evaluated for this task!

| Model Type | Rationale | Limitations |

- **Markov Chain** | Simple, interpretable; pitch sequences are naturally Markovian | No memory beyond current count; misses pitcher tendencies|
- **Logistic Regression / XGBoost on count states** | Easy baseline; count-state stats already precomputed | No sequential structure; treats each count independently. |
- **LSTM / GRU (Recurrent Networks)** | Captures sequential pitch dependencies; handles variable-length at-bats| Needs lots of data; may struggle with short Ekstraliga sequences |
- **Transformer (Attention-based)** | Best-in-class sequence modeling; attends to any prior pitch| Expensive to train; risk of overfitting on small dataset |
- **BERT-style pretraining on pitch tokens** | Pretrain on MLB, fine-tune on Ekstraliga — direct NLP analogy | Novel framing with little prior work to reference |

The most suitable right now seems LSTM or Transformer with feature masking, pretrained on MLB and fine-tuned on Ekstraliga PA logs with physical features dropped


## Repository Structure

```
pitch_model/
├── data/
│   ├── ekstraliga/          ← drop 2025_Ekstraliga_Stats.xlsx here
│   └── mlb/                 ← MLB parquet files (generated by mlb_api.py)
├── src/
│   ├── preprocessing.py     ← PA log parsing → clean DataFrame
│   ├── features.py          ← Vocabulary, player tendencies, PyTorch Dataset
│   ├── train.py             ← End-to-end training script (Markov + LSTM)
│   ├── decision_support.py  ← In-game substitution recommendation engine
│   ├── mlb_api.py           ← MLB Stats API data pulling script
│   └── models/
│       ├── markov.py        ← Markov chain baselines (pitch & outcome)
│       └── lstm_model.py    ← LSTM encoder + tendency MLP + classifier
├── notebooks/
│   └── 01_eda.ipynb         ← Exploratory data analysis
├── runs/                    ← Training outputs (auto-created)
└── requirements.txt
```