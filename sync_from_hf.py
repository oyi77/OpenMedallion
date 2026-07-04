#!/usr/bin/env python3
"""Sync full OpenMedallion dataset from HuggingFace to local directory.

Usage:
    python sync_from_hf.py                          # download all data
    python sync_from_hf.py --categories crypto,forex # specific categories
    python sync_from_hf.py --dry-run                # preview only
    python sync_from_hf.py --new-only               # skip existing files

New categories added in expansion:
    factors, options, energy, weather, agriculture, trade, fundamentals

Requires: pip install huggingface_hub
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF_REPO = "oyi77/OpenMedallion"
LOCAL_DATA = Path(__file__).parent / "data"

# All known categories (original + expanded)
ALL_CATEGORIES = [
    # Original
    "crypto", "derivatives", "equities", "countries", "etfs", "forex",
    "macro", "commodities", "indices", "bonds", "onchain", "defi",
    "sentiment", "prediction_markets", "environmental", "training",
    "alternative", "misc", "metadata",
    # Expanded
    "factors", "options", "energy", "weather", "agriculture",
    "trade", "fundamentals",
]


def get_hf_files(hf_repo: str, categories: list[str] | None = None) -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    all_files = list(api.list_repo_files(hf_repo, repo_type="dataset"))
    parquets = [f for f in all_files if f.startswith("data/") and f.endswith(".parquet")]
    if categories:
        parquets = [
            f for f in parquets
            if any(f.startswith(f"data/{cat}/") for cat in categories)
        ]
    return parquets


def download_file(hf_repo: str, remote_path: str, local_root: Path, new_only: bool = False) -> str:
    from huggingface_hub import hf_hub_download
    relative = remote_path.removeprefix("data/")
    local_path = local_root / relative
    if local_path.exists():
        if new_only:
            return f"SKIP {remote_path}"
        if os.path.getsize(local_path) > 0:
            return f"SKIP {remote_path}"
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=hf_repo,
            filename=remote_path,
            repo_type="dataset",
            local_dir=str(local_root.parent),
        )
        return f"OK {remote_path}"
    except Exception as e:
        return f"ERR {remote_path}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync OpenMedallion from HuggingFace")
    parser.add_argument("--categories", help="Comma-separated list of categories to download")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--threads", type=int, default=8, help="Download parallelism")
    parser.add_argument("--new-only", action="store_true", help="Only download files not present locally")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    if categories:
        invalid = [c for c in categories if c not in ALL_CATEGORIES]
        if invalid:
            print(f"WARNING: Unknown categories: {invalid}")
            print(f"Known: {', '.join(ALL_CATEGORIES)}")

    print(f"Fetching file list from {HF_REPO} ...")
    all_files = get_hf_files(HF_REPO, categories)
    print(f"Found {len(all_files)} parquet files")

    if categories:
        for cat in categories:
            n = sum(1 for f in all_files if f.startswith(f"data/{cat}/"))
            print(f"  {cat}: {n} files")

    if args.dry_run:
        print("\nDry run — not downloading.")
        return

    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(download_file, HF_REPO, f, LOCAL_DATA, args.new_only): f
            for f in all_files
        }
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(all_files)} done")

    ok = sum(1 for r in results if r.startswith("OK"))
    skipped = sum(1 for r in results if r.startswith("SKIP"))
    errors = sum(1 for r in results if r.startswith("ERR"))
    print(f"\nDone: {ok} downloaded, {skipped} skipped, {errors} errors")
    if errors:
        for r in results:
            if r.startswith("ERR"):
                print(f"  {r}")


if __name__ == "__main__":
    main()
