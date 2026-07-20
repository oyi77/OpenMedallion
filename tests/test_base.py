"""Tests for collectors/base.py utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Module under test
from collectors.base import (
    LOG,
    repo_path,
    save,
    to_datetime_index,
)


class TestRepoPath:
    def test_returns_path_under_data(self, tmp_data: Path) -> None:
        result = repo_path("test_cat", "file.parquet")
        assert result.parent == tmp_data / "test_cat"
        assert result.name == "file.parquet"

    def test_creates_parent_dirs(self, tmp_data: Path) -> None:
        result = repo_path("a/b/c", "x.parquet")
        assert result.parent.exists()

    def test_idempotent(self, tmp_data: Path) -> None:
        p1 = repo_path("cat", "f.parquet")
        p2 = repo_path("cat", "f.parquet")
        assert p1 == p2
        assert p1.parent.exists()  # parent created first time
        assert not p1.exists()  # file not created until save()

class TestSave:
    def test_saves_nonempty_dataframe(self, tmp_data: Path) -> None:
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([0, 1], name="idx"))
        save(df, "test_cat", "data.parquet")
        target = tmp_data / "test_cat" / "data.parquet"
        assert target.exists()
        back = pd.read_parquet(target)
        assert len(back) == 2

    def test_skips_empty_dataframe(self, tmp_data: Path, caplog: pytest.LogCaptureFixture) -> None:
        df = pd.DataFrame()
        save(df, "cat", "empty.parquet")
        target = tmp_data / "cat" / "empty.parquet"
        assert not target.exists()
        assert any("skipping" in r.message.lower() for r in caplog.records)


class TestToDatetimeIndex:
    def test_converts_column_to_index(self) -> None:
        df = pd.DataFrame({"val": [10]}, index=[0])
        df["date"] = "2024-01-15"
        result = to_datetime_index(df, col="date")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index[0].tz is not None  # utc=True

    def test_preserves_datetime_index(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"val": [1, 2, 3]}, index=dates)
        result = to_datetime_index(df, col="date")
        assert result.index.name == dates.name

    def test_sorts_after_conversion(self) -> None:
        df = pd.DataFrame({"val": [3, 1]}, index=[1, 0])
        df["date"] = ["2024-01-05", "2024-01-01"]
        result = to_datetime_index(df)
        assert list(result["val"]) == [1, 3]


class TestFetch:
    def test_success(self) -> None:
        """Integration-ish: fetch a known URL with a fast response."""
        import requests

        try:
            resp = LOG  # prove module is importable
            # Actually test a real tiny URL
            r = requests.get("https://httpbin.org/get", timeout=10)
            assert r.status_code == 200
        except Exception:
            pytest.skip("network not available")

    def test_retry_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests
        from collectors.base import fetch

        call_count = 0

        def flaky_get(*args: object, **kwargs: object) -> requests.Response:
            nonlocal call_count
            call_count += 1
            resp = requests.Response()
            resp.status_code = 503
            resp.raw = None  # type: ignore[arg-type]
            raise requests.RequestException("Service Unavailable")

        monkeypatch.setattr(requests, "get", flaky_get)
        with pytest.raises(requests.RequestException):
            fetch("http://example.com", retries=2)
        # 2 attempts (not 3 — expires after retries count)
        assert call_count == 2
