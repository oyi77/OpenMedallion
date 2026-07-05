"""
GitHub stars as tech sentiment proxy.
Source: GitHub REST API (no auth — 60 req/hr rate limit).
Captures point-in-time snapshot: stars, forks, open_issues for tracked repos.
Output: data/alternative/github_tech_stars_snapshot.parquet — date, repo, stars, forks, open_issues
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

GITHUB_API = "https://api.github.com/repos"

# (owner, repo) pairs covering AI/ML, infra, crypto, web
REPOS: list[tuple[str, str]] = [
    ("tensorflow", "tensorflow"),
    ("pytorch", "pytorch"),
    ("microsoft", "vscode"),
    ("vercel", "next.js"),
    ("tiangolo", "fastapi"),
    ("openai", "whisper"),
    ("facebookresearch", "llama"),
    ("huggingface", "transformers"),
    ("langchain-ai", "langchain"),
    ("bitcoin", "bitcoin"),
    ("ethereum", "go-ethereum"),
]

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "OpenMedallion-Collector/1.0",
}


def _fetch_repo(owner: str, repo: str) -> dict | None:
    """Fetch a single repo's metadata; return None on failure."""
    url = f"{GITHUB_API}/{owner}/{repo}"
    try:
        import requests

        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 403:
            print(f"    Rate limited on {owner}/{repo} — waiting 60s")
            time.sleep(60)
            resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"    WARNING: {owner}/{repo} returned HTTP {resp.status_code}")
            return None
        return resp.json()
    except Exception as exc:
        print(f"    WARNING: {owner}/{repo} — {exc}")
        return None


def collect_github_stars() -> None:
    """Snapshot star/fork/issue counts for tracked repos."""
    today = date.today().isoformat()
    rows: list[dict] = []

    for owner, repo in REPOS:
        print(f"  Fetching: {owner}/{repo}")
        data = _fetch_repo(owner, repo)
        if data is None:
            continue
        rows.append(
            {
                "date": today,
                "repo": f"{owner}/{repo}",
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "open_issues": data.get("open_issues_count", 0),
            }
        )
        # Stay well within the 60 req/hr limit
        time.sleep(1.2)

    if not rows:
        print("  WARNING: No GitHub data collected — skipping")
        return

    df = pd.DataFrame(rows)
    df = to_datetime_index(df, col="date")
    save(df, "alternative", "github_tech_stars_snapshot.parquet")


def main() -> None:
    print("Fetching: GitHub tech stars snapshot")
    try:
        collect_github_stars()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
