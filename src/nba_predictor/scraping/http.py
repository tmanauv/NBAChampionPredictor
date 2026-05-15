from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from nba_predictor.config import BASE_URL

_DEFAULT_CACHE_DIR = Path("data/cache")

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def fetch_html(
    url: str,
    *,
    cache_dir: Path | None = _DEFAULT_CACHE_DIR,
    max_retries: int = _MAX_RETRIES,
    backoff_base: float = _BACKOFF_BASE,
) -> str:
    """Fetch HTML from *url* with caching and retry-with-backoff.

    Parameters
    ----------
    url : str
        Full URL to fetch.
    cache_dir : Path | None
        Directory to store cached HTML files.  Set to ``None`` to disable
        caching.
    max_retries : int
        Number of retry attempts on transient failures.
    backoff_base : float
        Base for exponential backoff (seconds).

    Returns
    -------
    str
        Raw HTML content.
    """
    if cache_dir is not None:
        cached = _read_cache(url, cache_dir)
        if cached is not None:
            return cached

    html = _fetch_with_retry(url, max_retries, backoff_base)

    if cache_dir is not None:
        _write_cache(url, html, cache_dir)

    return html


def _cache_path(url: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    safe_name = url.replace(BASE_URL, "").strip("/").replace("/", "_").replace(".", "_")
    filename = f"{safe_name}_{key}.html"
    return cache_dir / filename


def _read_cache(url: str, cache_dir: Path) -> str | None:
    path = _cache_path(url, cache_dir)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _write_cache(url: str, html: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url, cache_dir)
    path.write_text(html, encoding="utf-8")


def _fetch_with_retry(url: str, max_retries: int, backoff_base: float) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(url) as resp:
                return resp.read().decode("utf-8")
        except (URLError, OSError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait = backoff_base**attempt
                print(f"Retry {attempt + 1}/{max_retries} for {url} (waiting {wait:.1f}s): {exc}")
                time.sleep(wait)
    raise ConnectionError(f"Failed to fetch {url} after {max_retries + 1} attempts") from last_error
