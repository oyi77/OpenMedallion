"""
OpenMeteo global weather collector.
Source: open-meteo.com — completely free, no API key required.
Covers: Historical weather for major financial cities (climate risk proxy),
        agricultural regions, and storm/drought indicators.
Output: data/weather/OpenMeteo_<city>_daily.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

OPENMETEO_HIST = "https://archive-api.open-meteo.com/v1/archive"

# Major financial cities + agricultural hubs
LOCATIONS = {
    "NewYork_US":          (40.7128,  -74.0060),
    "London_UK":           (51.5074,   -0.1278),
    "Tokyo_JP":            (35.6762,  139.6503),
    "Shanghai_CN":         (31.2304,  121.4737),
    "HongKong_HK":         (22.3193,  114.1694),
    "Frankfurt_DE":        (50.1109,    8.6821),
    "Singapore_SG":        (1.3521,   103.8198),
    "Sydney_AU":           (-33.8688, 151.2093),
    "Dubai_AE":            (25.2048,   55.2708),
    "Jakarta_ID":          (-6.2088,  106.8456),
    "Mumbai_IN":           (19.0760,   72.8777),
    "SaoPaulo_BR":         (-23.5505, -46.6333),
    "Chicago_US":          (41.8781,  -87.6298),
    # Agricultural hubs
    "Iowa_US":             (41.8780,  -93.0977),  # corn belt
    "Mato_Grosso_BR":      (-12.9373, -55.5210),  # soy
    "Punjab_IN":           (30.9010,   75.8573),  # wheat
    "Sumatra_ID":          (0.5897,   101.3431),  # palm oil
    "BlackSea_UA":         (46.9759,   31.9946),  # wheat
}

VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",  # drought indicator
    "soil_moisture_0_to_7cm",
]


def fetch_weather(city: str, lat: float, lon: float, start: str = "2000-01-01") -> pd.DataFrame:
    import datetime
    end = datetime.date.today().isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(VARIABLES),
        "timezone": "UTC",
    }
    resp = fetch(OPENMETEO_HIST, params=params, timeout=60)
    data = resp.json()

    if "daily" not in data:
        return pd.DataFrame()

    daily = data["daily"]
    dates = daily.pop("time", [])
    if not dates:
        return pd.DataFrame()

    df = pd.DataFrame(daily, index=pd.to_datetime(dates, utc=True))
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()
    return df.dropna(how="all")


def main() -> None:
    for city, (lat, lon) in LOCATIONS.items():
        print(f"Fetching weather: {city} ({lat:.2f}, {lon:.2f}) ...")
        try:
            df = fetch_weather(city, lat, lon)
            save(df, "weather", f"OpenMeteo_{city}_1d.parquet")
        except Exception as exc:
            print(f"  WARNING: {city} — {exc}")


if __name__ == "__main__":
    main()
