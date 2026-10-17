"""Convert the raw TfNSW crash workbooks to CSV for faster loading."""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

FILES = {
    "crash": "nsw_road_crash_data_2020-2024_crash.xlsx",
    "traffic_unit": "nsw_road_crash_data_2020-2024_traffic_unit.xlsx",
}


def convert():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fname in FILES.items():
        df = pd.read_excel(RAW / fname, engine="openpyxl")
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, df.shape)


def load(name):
    return pd.read_csv(OUT / f"{name}.csv", low_memory=False)


if __name__ == "__main__":
    convert()
