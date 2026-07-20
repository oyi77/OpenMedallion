"""Shared fixtures for collector tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_data(tmp_path: Path, monkeypatch: Any) -> Generator[Path, None, None]:
    """Point REPO_ROOT to a temp directory so tests write nowhere real."""
    import collectors.base as base_mod

    monkeypatch.setattr(base_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(base_mod, "DATA_ROOT", tmp_path / "data")
    yield tmp_path / "data"
