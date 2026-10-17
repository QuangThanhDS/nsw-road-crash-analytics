"""Data quality profile: writes reports/data_quality_profile.csv and prints a summary."""
from pathlib import Path
import pandas as pd
from load_data import load

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = {"Unknown", "Unknown / not stated", "Country unknown", "Rest of NSW - Unknown",
           "Unknown motor vehicle", "Other or unknown"}

c = load("crash")
rows = []
for col in c.columns:
    s = c[col]
    rows.append({
        "column": col,
        "dtype": str(s.dtype),
        "n_unique": s.nunique(),
        "pct_missing": round(100 * s.isna().mean(), 2),
        "pct_unknown_coded": round(100 * s.astype(str).isin(UNKNOWN).mean(), 2),
        "top_value": s.value_counts().index[0] if s.notna().any() else None,
    })
prof = pd.DataFrame(rows)
prof["pct_missing_or_unknown"] = prof.pct_missing + prof.pct_unknown_coded
prof.to_csv(ROOT / "reports" / "data_quality_profile.csv", index=False)
print(prof.sort_values("pct_missing_or_unknown", ascending=False).head(15).to_string(index=False))

bad = ~(c.Latitude.between(-37.6, -28.0) & c.Longitude.between(140.9, 159.2))  # incl. Lord Howe Island
print("\nRows:", len(c), "| crash year 2019 (late reported):", (c["Year of crash"] == 2019).sum(),
      "| coords outside NSW:", bad.sum())
print(c.loc[bad, ["Crash ID", "Latitude", "Longitude", "LGA", "Town"]].to_string(index=False))
