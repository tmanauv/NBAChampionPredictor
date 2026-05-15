from __future__ import annotations

import pytest

from nba_predictor.scraping.http import _cache_path, _write_cache, fetch_html


class TestFetchHtml:
    def test_cache_write_and_read(self, tmp_path):
        cache_dir = tmp_path / "cache"
        url = "https://example.com/test.html"
        expected = "<html>hello</html>"

        _write_cache(url, expected, cache_dir)

        result = fetch_html(url, cache_dir=cache_dir)
        assert result == expected

    def test_no_cache_raises_when_network_blocked(self, tmp_path):
        url = "https://example.com/nocache.html"
        with pytest.raises((RuntimeError, ConnectionError)):
            fetch_html(url, cache_dir=None)

    def test_cache_path_is_deterministic(self, tmp_path):
        url = "https://www.basketball-reference.com/leagues/NBA_2023.html"
        p1 = _cache_path(url, tmp_path)
        p2 = _cache_path(url, tmp_path)
        assert p1 == p2
        assert p1.suffix == ".html"

    def test_cache_miss_then_hit(self, tmp_path):
        cache_dir = tmp_path / "cache"
        url = "https://example.com/page.html"
        html = "<html><body>cached</body></html>"

        _write_cache(url, html, cache_dir)

        result = fetch_html(url, cache_dir=cache_dir)
        assert result == html
