#!/usr/bin/env python3
"""Upload expanded OpenMedallion dataset to HuggingFace.

Uploads all local data/ contents to HF dataset repo root,
updating changed files and adding new categories.
"""
import os, sys, json
from pathlib import Path
from huggingface_hub import HfApi
from datetime import datetime

REPO_ID = "oyi77/OpenMedallion"
REPO_TYPE = "dataset"
LOCAL_DATA = Path(__file__).parent / "data"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token)
    commit_msg = f"Expand dataset with fresh collector data — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"

    # Get current HF file list to compute diff
    print("Fetching current HF repo state...")
    hf_all = set(api.list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    hf_data = {
        f for f in hf_all
        if not f.startswith(".") and f not in ("LICENSE", "README.md", "data")
    }

    # Enumerate local files
    local = {}
    for f in sorted(LOCAL_DATA.rglob("*")):
        if not f.is_file() or ".cache" in str(f):
            continue
        rel = str(f.relative_to(LOCAL_DATA))
        local[rel] = f.stat().st_size

    new_files = sorted(set(local.keys()) - hf_data)
    updated_files = []
    for f in sorted(set(local.keys()) & hf_data):
        if f not in local:
            continue
        local_size = local[f]
        # Check if differs from HF by attempting head request
        try:
            meta = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
        except:
            meta = None
            break
        # We'll just track all common files for upload; upload_folder handles dedup
        updated_files.append(f)

    print(f"  Local data files: {len(local)}")
    print(f"  HF data files:    {len(hf_data)}")
    print(f"  New files:        {len(new_files)}")
    print(f"  Total to upload:  {len(local)} files ({sum(local.values())/1024/1024:.1f} MB)")

    if not new_files and not updated_files:
        print("\nNothing new to upload.")
        return

    # Upload via upload_folder — handles bulk efficient upload
    print(f"\nUploading data/ contents to {REPO_ID}...")
    try:
        result = api.upload_folder(
            folder_path=str(LOCAL_DATA),
            path_in_repo=".",            # upload data/ contents to repo root
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=commit_msg,
            ignore_patterns=["__pycache__/*", ".cache/*", "*.pyc", "*.pyo"],
        )
        print(f"\n✅ Upload complete: {result.commit_message}")
        print(f"   Commit: {result.commit_url}")
        return result

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
