# C-MAPSS FD001 — Remaining Useful Life Prediction

End-to-end approach to predicting **Remaining Useful Life (RUL)** of turbofan
engines on the NASA C-MAPSS FD001 dataset, using tree-based ensembles with a
piecewise-linear degradation target.

## Problem

Each engine in the dataset runs until failure. Given a snapshot of 21 sensor
measurements and 3 operational settings at a given cycle, predict how many
cycles of useful life remain. Evaluation follows the official C-MAPSS
protocol: predict RUL at each test unit's **last recorded cycle** and compare
against the ground truth `RUL_FD001`.

## Approach

1. **Load & explore** — 26-column whitespace-separated text files with named
   columns (`unit`, `time`, settings, sensors). EDA lives in the scratch
   notebook (`notebook/`); it motivated the preprocessing below.
2. **Drop constant sensors** — several sensors (e.g. `T2`, `P2`, `epr`) never
   vary in FD001 and carry zero predictive signal.
3. **Piecewise-linear RUL target** — RUL = `t_max − t`, clipped at **125
   cycles**. Engines degrade negligibly early in life, so a linear target
   forces the model to overestimate healthy engines. The cap is a
   literature-backed standard for this dataset.
4. **Baseline comparison** — Random Forest, LightGBM, and XGBoost on
   identical snapshot features, no feature scaling (trees are invariant to
   monotonic transformations):

   Model | RMSE | R² |
 |---|---|---|
 | Random Forest | 17.23 | 0.815 |
 | **LightGBM** | **16.99** | **0.820** |
 | XGBoost | 17.05 | 0.819 |

   LightGBM was selected for hyperparameter tuning.

6. **Leakage-free tuning** — hyperparameter search with
   `RandomizedSearchCV` + **`GroupKFold`** (grouped by engine unit). Plain
   row-random CV would place cycles from the *same engine* in both train and
   validation folds, inflating scores; grouping keeps every engine entirely
   within one fold.
7. **Final evaluation** — the tuned model is evaluated once on the official
   held-out test set, with per-unit trajectory plots for diagnostics.

## Results

 | Model | MAE | RMSE | R² |
 |---|---|---|---|
 | LightGBM (tuned, GroupKFold CV) |  11.91 | 16.97 | 0.821 |

## Repository structure

```text
├── notebook/   # scratch notebook: EDA, baselines, tuning, diagnostics
├── src/
│   ├── config.py     # paths, column names, constants
│   ├── data.py       # loading, cleaning, RUL target, train/test prep
│   ├── model.py      # LightGBM + grouped hyperparameter search
│   ├── evaluate.py   # metrics and plots
│   └── main.py       # entry point
├── requirements.txt
└── README.md
```

## Data

The dataset is **not included**. Download the NASA C-MAPSS data
([Kaggle mirror](https://www.kaggle.com/datasets/behrad3d/nasa-c-maps)) and
place the files in `data/`:

```text
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

> Saxena, A., Goebel, D., Simon, D., & Eklund, N. (2008). Damage Propagation
> Modeling for Aircraft Engine Run-to-Failure Simulation. PHM.

## Setup & run

```bash
pip install -r requirements.txt
python src/main.py
```

## Notes & limitations

- CV scores use engine-grouped folds; row-random CV on this data reports
  optimistically low RMSE.
- Snapshot models ignore temporal trends; a windowed sequence model
  (CNN/LSTM) is the planned next step for cross-condition generalization.
