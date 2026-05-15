# NBA Champion Predictor

Predict future NBA Champions (playoff performance) from historical regular-season data using ML regression. Scrapes 40+ years of team records from [Basketball Reference](https://www.basketball-reference.com) and ranks probable champions with an NDCG score of ~70%.

## Project Structure

```
src/nba_predictor/
├── config.py                # Constants (random seed, season ranges, etc.)
├── scraping/
│   ├── http.py              # Cached HTTP fetcher with retry/backoff
│   ├── utils.py             # Shared helpers (team-name mapping)
│   ├── team_records.py      # Team advanced stats
│   ├── playoff_records.py   # Playoff win/loss → Champion Share Score
│   ├── conf_standings.py    # Conference standings
│   ├── roster_accolades.py  # MVP, All-NBA, DPOY, All-Defense shares
│   ├── season_details.py    # Orchestrator (merges all sources for a season)
│   └── collect.py           # Multi-season collection loop
├── features/
│   ├── selection.py         # Correlation-based feature selection (train-only)
│   └── preprocessing.py     # ColumnTransformer pipeline builder
└── models/
    ├── bakeoff.py           # Cross-validated model comparison (GroupKFold)
    ├── grid_search.py       # Hyperparameter grid search
    ├── evaluate.py          # Per-season NDCG evaluation & prediction
    └── predict.py           # Model save/load (joblib)

notebooks/
├── exploration.ipynb        # Data collection & EDA
└── modeling.ipynb           # Model training, grid search, predictions
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/tmanauv/NBAChampionPredictor.git
cd NBAChampionPredictor
pip install -e ".[notebook]"

# Run the exploration notebook (scrapes data, ~6 min first time)
jupyter notebook notebooks/exploration.ipynb

# Or use the package directly
python -c "
from nba_predictor.scraping import collect_all_seasons
df = collect_all_seasons(start=2020, end=2023)
print(df.shape)
"
```

## Features

- **Web scraping** of team records, playoff results, conference standings, and player accolades from Basketball Reference
- **HTML caching** — scraped pages are cached locally (`data/cache/`), so subsequent runs skip network calls entirely
- **Retry with exponential backoff** for transient HTTP failures
- **Engineered features**: Strength of Year (SOY), Net Four Factors Rating, Updated Four Factors Rating, MVP/All-NBA/DPOY/All-Defense shares
- **Train-only feature selection** to prevent data leakage from holdout seasons
- **Model bake-off** with GroupKFold cross-validation (SVR, Linear Regression, Random Forest, XGBoost, Gradient Boosting)
- **Grid search** with per-season NDCG evaluation
- **Model persistence** via joblib

## Models

The pipeline evaluates five regressors and performs grid search on the top three:

| Model | Cross-Val MAE |
|-------|--------------|
| Random Forest | Best overall |
| Gradient Boosting | Strong |
| XGBoost | Strong |
| SVR | Moderate |
| Linear Regression | Baseline |

Best model achieves **~70% NDCG@5** on holdout seasons (2021-2023).

## Development

```bash
# Install dev tools
pip install ruff mypy pytest pre-commit

# Set up pre-commit hooks
pre-commit install

# Lint
ruff check src/

# Type check
mypy src/nba_predictor/ --ignore-missing-imports

# Run tests
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
