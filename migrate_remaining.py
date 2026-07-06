#!/usr/bin/env python3
"""Batch migrate remaining misc files to synthetic/ using upload_folder to avoid rate limit."""
import os, re, tempfile, shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

# Get files that still need migration
all_files = list(api.list_repo_files("oyi77/OpenMedallion", repo_type="dataset"))
misc = sorted(f for f in all_files if f.startswith("misc/platform_base/stockity/"))
synthetic = [f for f in all_files if f.startswith("synthetic/stockity/")]

print(f"Misc remaining: {len(misc)}")
print(f"Synthetic existing: {len(synthetic)}")

if not misc:
    print("All files already migrated!")
    exit(0)

def new_path(old: str) -> str:
    fname = old.split("/")[-1]
    if fname.startswith("Z-"):
        asset = re.sub(r"^Z-(.+?)2?_stockity_5m_\d+d\.parquet$", r"\1", fname)
        return f"synthetic/stockity/{asset}_stockity_5m_alt.parquet"
    asset = re.sub(r"_stockity_5m_\d+d\.parquet$", "", fname)
    return f"synthetic/stockity/{asset}_stockity_5m.parquet"

# Download all remaining files to temp dir
with tempfile.TemporaryDirectory() as tmpdir:
    synthetic_dir = Path(tmpdir) / "synthetic" / "stockity"
    synthetic_dir.mkdir(parents=True)
    
    print(f"\nDownloading {len(misc)} files to {tmpdir}...")
    for i, old_path in enumerate(misc, 1):
        local = hf_hub_download("oyi77/OpenMedallion", filename=old_path, repo_type="dataset", token=token)
        new_fname = new_path(old_path).split("/")[-1]
        shutil.copy(local, synthetic_dir / new_fname)
        print(f"  [{i:3d}/{len(misc)}] {new_fname}")
    
    # Upload entire synthetic/ folder in ONE commit
    print(f"\nUploading synthetic/ folder...")
    api.upload_folder(
        folder_path=Path(tmpdir) / "synthetic",
        path_in_repo="synthetic",
        repo_id="oyi77/OpenMedallion",
        repo_type="dataset",
        commit_message=f"batch migrate {len(misc)} remaining files from misc/ to synthetic/",
    )
    print("Upload complete!")

# Delete old misc/ files in ONE commit
print(f"\nDeleting old misc/ files...")
api.delete_folder(
    path_in_repo="misc",
    repo_id="oyi77/OpenMedallion",
    repo_type="dataset",
    commit_message="rm misc/ after migration to synthetic/",
)
print("Migration complete!")
