"""Robust HTTP client with retries, rate limiting, and validation."""

import logging
import time

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Rate limiting: minimum seconds between requests to the same domain
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 3.0  # seconds


def _rate_limit():
    """Enforce minimum interval between HTTP requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - elapsed
        logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
        time.sleep(sleep_time)
    _last_request_time = time.time()


@retry(
    retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
def fetch_page(url):
    """Fetch a web page with retries and rate limiting.

    Args:
        url: The URL to fetch.

    Returns:
        str: The HTML content of the page.

    Raises:
        requests.HTTPError: If the response status code indicates an error
            after all retries are exhausted.
        requests.RequestException: If a network error persists after retries.
    """
    _rate_limit()
    logger.info("Fetching: %s", url)

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; NBAChampionPredictor/1.0; "
                "educational project)"
            )
        },
    )
    response.raise_for_status()
    return response.text


def validate_dataframe(df, expected_columns=None, min_rows=1, name="DataFrame"):
    """Validate a scraped DataFrame meets basic quality checks.

    Args:
        df: DataFrame to validate.
        expected_columns: Optional list of columns that must be present.
        min_rows: Minimum number of rows expected.
        name: Descriptive name for error messages.

    Raises:
        ValueError: If validation fails.
    """
    if df is None or df.empty:
        raise ValueError(f"{name}: DataFrame is empty")

    if len(df) < min_rows:
        raise ValueError(
            f"{name}: Expected at least {min_rows} rows, got {len(df)}"
        )

    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"{name}: Missing expected columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    # Check for excessive NaN values (>50% of any column is suspicious)
    nan_ratios = df.isnull().mean()
    high_nan_cols = nan_ratios[nan_ratios > 0.5]
    if not high_nan_cols.empty:
        logger.warning(
            "%s: High NaN ratio in columns: %s",
            name,
            dict(high_nan_cols),
        )
