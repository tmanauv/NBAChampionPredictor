# NBA Champion Predictor

Predict NBA Champions using machine learning on 40+ years of historical regular-season data scraped from [Basketball Reference](https://www.basketball-reference.com/).

The model uses a **regression approach** — engineering features that resemble the characteristic traits of NBA Champions, then ranking teams by predicted Championship Share Score. Achieves an **NDCG@5 score of ~70%** on held-out seasons.

---

## Project Architecture

```
NBAChampionPredictor/
├── src/
│   ├── scraping.py        # Web scraping from Basketball Reference
│   ├── cache.py           # Parquet-based data caching layer
│   ├── http_client.py     # Robust HTTP client (retries, rate limiting)
│   ├── features.py        # Feature engineering & selection (MI, RFE, correlation)
│   ├── model.py           # Model training, cross-validation, grid search
│   ├── tuning.py          # Optuna-based hyperparameter optimization
│   ├── ensemble.py        # Voting & Stacking ensemble models
│   ├── evaluation.py      # Metrics, reporting, model serialization
│   └── utils.py           # Plotting utilities
├── tests/                 # Unit tests (pytest)
├── data/                  # Cached scraped data (Parquet, gitignored)
├── models/                # Serialized trained models (gitignored)
├── NBA Champ Predictor.ipynb  # Original exploratory notebook
├── predict.py             # CLI prediction script
├── requirements.txt       # Python dependencies
└── .github/workflows/     # CI (linting + tests)
```

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/tmanauv/NBAChampionPredictor.git
cd NBAChampionPredictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook "NBA Champ Predictor.ipynb"
```

### 3. Use the CLI

```bash
# Predict champions for the current season
python predict.py --season 2024

# Retrain with all data and predict
python predict.py --season 2024 --retrain

# Force re-scrape (bypass cache)
python predict.py --season 2024 --refresh
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Data Pipeline

1. **Scraping** (`src/scraping.py`): Fetches team records, conference standings, playoff records, and roster accolades from Basketball Reference
2. **Caching** (`src/cache.py`): Stores scraped data as Parquet files in `data/raw/` to avoid re-scraping on every run
3. **HTTP Robustness** (`src/http_client.py`): Retries with exponential backoff, rate limiting (3s between requests), proper User-Agent

## Feature Engineering

**Original features** from Basketball Reference:
- Team records (W, L, SRS, MOV, ORtg, DRtg, Pace, etc.)
- Conference standings and seeding
- Roster accolades (All-Stars, All-NBA, DPOY)
- Playoff records

**Engineered features** (`src/features.py`):
- `Win_Pct`, `Net_Rtg`, `Pace_Adj_Net_Rtg`
- `SRS_Rank`, `Shooting_Composite`, `Defensive_Index`
- `MOV_Zscore`, `FTr_Advantage`
- Optional: `Playoff_Experience` (cumulative roster playoff games)

**Feature selection** uses three methods and takes their union:
- Pearson correlation with target
- Mutual information regression
- Recursive Feature Elimination (RFE) with GBR

## Models

| Model | Description |
|-------|-------------|
| GradientBoostingRegressor | Primary model, tuned via Optuna |
| RandomForestRegressor | Ensemble tree model |
| XGBRegressor | Gradient boosting (XGBoost) |
| LGBMRegressor | Gradient boosting (LightGBM) |
| VotingRegressor | Average of all 4 models |
| StackingRegressor | Meta-learner (RidgeCV) on base model predictions |

Hyperparameter tuning uses **Optuna** (Bayesian optimization with TPE sampler) and **TimeSeriesSplit** cross-validation to prevent data leakage.

## Evaluation

Models are evaluated on:
- **R²**, **MAE**, **MSE**, **RMSE** — standard regression metrics
- **NDCG@5** — measures ranking quality (can the model correctly rank the top 5 championship contenders?)
- **Bootstrap 95% CI** — confidence intervals via 1000 bootstrap iterations

## CI/CD

GitHub Actions runs on every push and PR to `main`:
- **Lint**: `ruff check src/ tests/`
- **Smoke test**: Verify all module imports + `pytest tests/`

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes and ensure `ruff check src/ tests/` and `pytest tests/` pass
4. Open a PR against `main`

## License

This project is for educational purposes.
