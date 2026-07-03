#!/usr/bin/env python3
"""Sync full OpenMedallion dataset from HuggingFace to local directory.

Usage:
    python sync_from_hf.py                         # download all data
    python sync_from_hf.py --categories crypto,forex  # specific categories
    python sync_from_hf.py --dry-run                # preview only

Requires: pip install huggingface_hub
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HF_REPO = "oyi77/OpenMedallion"
LOCAL_DATA = Path(__file__).parent / "data"


def get_hf_files(hf_repo: str) -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    all_files = list(api.list_repo_files(hf_repo, repo_type="dataset"))
    return [f for f in all_files if f.startswith("data/") and f.endswith(".parquet")]


def download_file(hf_repo: str, remote_path: str, local_root: Path) -> str:
    from huggingface_hub import hf_hub_download
    relative = remote_path.removeprefix("data/")
    local_path = local_root / relative
    if local_path.exists() and os.path.getsize(local_path) > 1_000_000:
        return f"SKIP {relative}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(hf_repo, remote_path, repo_type="dataset")
        os.system(f'cp "{downloaded}" "{local_path}"')
        return f"OK  {relative}"
    except Exception as e:
        return f"ERR {relative}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Sync OpenMedallion from HuggingFace")
    parser.add_argument("--categories", "-c", help="Comma-separated categories (e.g. crypto,forex)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without downloading")
    parser.add_argument("--threads", "-t", type=int, default=8, help="Download threads")
    args = parser.parse_args()

    print(f"Fetching file list from {HF_REPO}...")
    all_files = get_hf_files(HF_REPO)
    print(f"Found {len(all_files)} parquet files\n")

    if args.categories:
        selected = [c.strip() for c in args.categories.split(",")]
        all_files = [f for f in all_files if any(f"/{s}/" in f for s in selected)]
        print(f"Filtered to {len(all_files)} files in: {', '.join(selected)}\n")

    if args.dry_run:
        for f in sorted(all_files):
            print(f"  {f}")
        print(f"\nWould download {len(all_files)} files")
        return

    print(f"Downloading {len(all_files)} files ({args.threads} threads)...")
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(download_file, HF_REPO, f, LOCAL_DATA): f for f in all_files}
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
