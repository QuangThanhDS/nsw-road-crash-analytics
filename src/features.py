"""Shared cleaning and feature engineering for the NSW crash data."""
import numpy as np
import pandas as pd
from load_data import load

HOUR_ORDER = ["00:01 - 01:59", "02:00 - 03:59", "04:00 - 05:59", "06:00 - 07:59",
              "08:00 - 09:59", "10:00 - 11:59", "12:00 - 13:59", "14:00 - 15:59",
              "16:00 - 17:59", "18:00 - 19:59", "20:00 - 21:59", "22:00 - Midnight"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEVERITY_ORDER = ["Fatal", "Serious Injury", "Moderate Injury", "Minor/Other Injury",
                  "Non-casualty (towaway)"]
REGION_MAP = {"Syd-Newc-Woll Gtr conurbation": "Greater Sydney/Newcastle/Wollongong",
              "Rest of NSW - Urban": "Regional urban",
              "Rest of NSW - Rural": "Regional rural"}
NSW_BBOX = dict(lat=(-37.6, -28.0), lon=(140.9, 159.2))  # 159 = Lord Howe Island


def clean_crash(c):
    c = c.copy()
    # Crash year inside 2020-2024 only (2019 crashes were reported late, in 2020)
    c = c[c["Year of crash"].between(2020, 2024)]
    # Target: killed or seriously injured (KSI)
    c["ksi"] = c["Degree of crash - detailed"].isin(["Fatal", "Serious Injury"]).astype(int)
    c["fatal"] = (c["Degree of crash"] == "Fatal").astype(int)
    c["region"] = c["Conurbation 1"].map(REGION_MAP)  # 'Unknown' -> NaN
    c["speed_kmh"] = pd.to_numeric(c["Speed limit"].str.replace(" km/h", ""), errors="coerce")
    # Coordinates outside NSW are treated as missing
    bad = ~(c.Latitude.between(*NSW_BBOX["lat"]) & c.Longitude.between(*NSW_BBOX["lon"]))
    c.loc[bad, ["Latitude", "Longitude"]] = np.nan
    return c


def tu_flags(t):
    """Crash-level flags describing who/what was involved (from the TRAFFIC UNIT table)."""
    g = t["TU type group"]
    flags = pd.DataFrame({
        "Crash ID": t["Crash ID"],
        "has_pedestrian": g.eq("Pedestrian"),
        "has_motorcycle": g.eq("Motorcycle"),
        "has_pedal_cycle": g.eq("Pedal cycle"),
        "has_heavy_vehicle": g.isin(["Heavy rigid truck", "Articulated truck", "Bus"]),
        "hit_tree_or_pole": t["Object hit 1"].isin(["Tree/bush", "Utility pole"])
                            | t["Object hit 2"].isin(["Tree/bush", "Utility pole"]),
        "has_parked_vehicle": t["TU controlled flag"].eq("N"),
    })
    return flags.groupby("Crash ID").max().astype(int).reset_index()


def build():
    c = clean_crash(load("crash"))
    t = load("traffic_unit")
    return c.merge(tu_flags(t), on="Crash ID", how="left"), t
