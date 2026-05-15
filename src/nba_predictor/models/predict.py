from __future__ import annotations

from pathlib import Path

import joblib


def save_model(model, path: str | Path) -> None:
    """Persist a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str | Path):
    """Load a previously saved model."""
    return joblib.load(path)
