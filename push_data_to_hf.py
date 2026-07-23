#!/usr/bin/env python3
"""Upload expanded OpenMedallion dataset to HuggingFace.

Direct per-category upload to avoid list_repo_files hang on large repos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi
from datetime import datetime

REPO_ID = "oyi77/OpenMedallion"
REPO_TYPE = "dataset"
LOCAL_DATA = Path(__file__).parent / "data"

# Categories to push (subset when testing)
ALL_CATEGORIES = sorted(
    d.name for d in LOCAL_DATA.iterdir()
    if d.is_dir() and not d.name.startswith(".") and d.name not in ("label_map.json", "train.jsonl")
)


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    # Push each category separately for resilience
    for cat in ALL_CATEGORIES:
        cat_dir = LOCAL_DATA / cat
        files = list(cat_dir.rglob("*.parquet"))
        if not files:
            continue
        total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"\n=== {cat} ({len(files)} files, {total_mb:.1f} MB) ===")
        try:
            result = api.upload_folder(
                folder_path=str(cat_dir),
                path_in_repo=cat,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Update {cat} — {ts}",
                allow_patterns="*.parquet",
                ignore_patterns=["__pycache__/*", ".cache/*"],
            )
            print(f"  ✅ Uploaded: {result.commit_message}")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            # Continue with other categories

    print("\nDone.")


if __name__ == "__main__":
    main()
