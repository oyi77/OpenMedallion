"""
Load OHLCV data from OpenMedallion parquet files.
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd


ASSET_CATEGORIES = {
    'crypto': ['crypto'],
    'forex': ['forex'],
    'commodities': ['commodities'],
    'equities': ['equities', 'etfs', 'indices'],
}


def load_parquet_file(
    file_path: Path,
    min_rows: int = 100
) -> Optional[pd.DataFrame]:
    """
    Load single parquet file with validation.
    
    Args:
        file_path: Path to parquet file
        min_rows: Skip files with fewer rows
    
    Returns:
        DataFrame with DatetimeIndex or None if invalid
    """
    try:
        df = pd.read_parquet(file_path)
        
        if len(df) < min_rows:
            return None
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Ensure datetime index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            return None
        
        # Validate OHLCV columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return None
        
        # Sort by date
        df = df.sort_index()
        
        return df
        
    except Exception:
        return None


def load_asset_class(
    data_dir: Path,
    asset_class: str,
    min_rows: int = 100,
    max_files: Optional[int] = None
) -> List[tuple[str, pd.DataFrame]]:
    """
    Load all files for a given asset class.
    
    Args:
        data_dir: Root training data directory
        asset_class: One of 'crypto', 'forex', 'commodities', 'equities'
        min_rows: Skip files with fewer rows
        max_files: Limit number of files loaded (for testing)
    
    Returns:
        List of (asset_name, dataframe) tuples
    """
    categories = ASSET_CATEGORIES.get(asset_class, [])
    
    results = []
    
    for category in categories:
        cat_dir = data_dir / category
        
        if not cat_dir.exists():
            continue
        
        parquet_files = sorted(cat_dir.glob('*.parquet'))
        
        if max_files:
            parquet_files = parquet_files[:max_files]
        
        for file_path in parquet_files:
            df = load_parquet_file(file_path, min_rows)
            
            if df is not None:
                asset_name = file_path.stem
                results.append((asset_name, df))
    
    return results
