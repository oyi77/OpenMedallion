#!/usr/bin/env python3
"""EIA Crude Oil & Petroleum Inventories - Weekly data."""
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from collectors.base import BaseFetcher, save_parquet

class EIAInventoriesFetcher(BaseFetcher):
    def fetch_crude_inventories(self):
        """Fetch US Crude Oil Commercial Stocks (weekly)."""
        url = "https://ir.eia.gov/wpsr/table9.csv"
        df = pd.read_csv(url, skiprows=2)
        df.columns = ['date', 'crude_stocks_mb']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna().sort_values('date')
        return df

    def fetch_gasoline_inventories(self):
        """Fetch US Gasoline Stocks (weekly)."""
        url = "https://ir.eia.gov/wpsr/table1.csv"
        df = pd.read_csv(url, skiprows=2)
        df.columns = ['date', 'gasoline_stocks_mb']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna().sort_values('date')
        return df

    def fetch_distillate_inventories(self):
        """Fetch US Distillate Fuel Oil Stocks (weekly)."""
        url = "https://ir.eia.gov/wpsr/table5.csv"
        df = pd.read_csv(url, skiprows=2)
        df.columns = ['date', 'distillate_stocks_mb']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna().sort_values('date')
        return df

if __name__ == "__main__":
    fetcher = EIAInventoriesFetcher()
    
    # Crude
    df = fetcher.fetch_crude_inventories()
    save_parquet(df, "data/energy/EIA_Crude_Stocks_1w.parquet", "date")
    print(f"OK EIA_Crude_Stocks_1w - {len(df)} rows")
    
    # Gasoline
    df = fetcher.fetch_gasoline_inventories()
    save_parquet(df, "data/energy/EIA_Gasoline_Stocks_1w.parquet", "date")
    print(f"OK EIA_Gasoline_Stocks_1w - {len(df)} rows")
    
    # Distillate
    df = fetcher.fetch_distillate_inventories()
    save_parquet(df, "data/energy/EIA_Distillate_Stocks_1w.parquet", "date")
    print(f"OK EIA_Distillate_Stocks_1w - {len(df)} rows")
