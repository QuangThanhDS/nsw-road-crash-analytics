"""Exploratory charts for the report. Writes PNGs to reports/figures/ and summary tables to reports/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from features import build, HOUR_ORDER, DAY_ORDER, SEVERITY_ORDER

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Palette (validated reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
REGION_COLORS = {"Greater Sydney/Newcastle/Wollongong": BLUE, "Regional urban": ORANGE,
                 "Regional rural": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


c, t = build()
tables = {}

# 1. Severity distribution
sev = c["Degree of crash - detailed"].value_counts().reindex(SEVERITY_ORDER)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(sev.index[::-1], sev.values[::-1], color=BLUE, height=0.6)
for i, v in enumerate(sev.values[::-1]):
    ax.text(v + 400, i, f"{v:,} ({v / sev.sum():.1%})", va="center", color=INK2, fontsize=9)
ax.set_xlim(0, sev.max() * 1.3); ax.grid(axis="y", visible=False)
ax.set_title("Crashes by severity, NSW 2020–2024"); ax.set_xlabel("Crashes")
save(fig, "01_severity_distribution.png")
tables["severity"] = sev

# 2. Yearly trend by severity group
c["sev_group"] = np.where(c.fatal == 1, "Fatal", np.where(c.ksi == 1, "Serious injury",
                 np.where(c["Degree of crash"] == "Injury", "Moderate/minor injury", "Non-casualty")))
yr = c.groupby(["Year of crash", "sev_group"]).size().unstack()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for col, color in [("Serious injury", BLUE), ("Moderate/minor injury", ORANGE), ("Non-casualty", AQUA)]:
    axes[0].plot(yr.index, yr[col], marker="o", lw=2, ms=6, color=color, label=col)
axes[0].set_title("Crashes per year by severity"); axes[0].legend(frameon=False, fontsize=8)
axes[0].set_ylim(0); axes[0].set_xticks(yr.index)
axes[1].plot(yr.index, yr["Fatal"], marker="o", lw=2, ms=6, color=BLUE)
for x, y in yr["Fatal"].items():
    axes[1].annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK2)
axes[1].set_title("Fatal crashes per year"); axes[1].set_ylim(0, yr["Fatal"].max() * 1.2)
axes[1].set_xticks(yr.index)
save(fig, "02_yearly_trend.png")
tables["yearly"] = yr

# 3. Hour x weekday heatmap (all crashes) and KSI share
hm = c[c["Two-hour intervals"] != "Unknown"].pivot_table(index="Day of week of crash", columns="Two-hour intervals",
                                                          values="ksi", aggfunc=["size", "mean"])
cnt = hm["size"].reindex(index=DAY_ORDER, columns=HOUR_ORDER)
rate = hm["mean"].reindex(index=DAY_ORDER, columns=HOUR_ORDER)
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
for ax, data, title, fmt in [(axes[0], cnt, "Number of crashes by day and time", "{:,.0f}"),
                             (axes[1], rate * 100, "Share of crashes that are fatal or serious (%)", "{:.0f}")]:
    im = ax.imshow(data.values, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(12), [h.split(" - ")[0] for h in HOUR_ORDER]); ax.set_yticks(range(7), [d[:3] for d in DAY_ORDER])
    ax.grid(False); ax.set_title(title)
    vmin, vmax = np.nanmin(data.values), np.nanmax(data.values)
    for i in range(7):
        for j in range(12):
            v = data.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (v - vmin) > 0.55 * (vmax - vmin) else INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
axes[1].set_xlabel("Two-hour interval starting")
save(fig, "03_time_heatmaps.png")

# 4. KSI and fatal share by speed limit: metro vs rest of NSW
# Note: TfNSW's urban/non-urban split is itself based on speed limit (<=80 km/h urban),
# so 'Regional urban' and 'Regional rural' barely overlap on speed. We compare metro vs rest of NSW.
c["metro"] = np.where(c.region.eq("Greater Sydney/Newcastle/Wollongong"), "Greater Sydney/Newcastle/Wollongong",
                      np.where(c.region.notna(), "Rest of NSW", None))
sp = c[c.speed_kmh.isin([40, 50, 60, 70, 80, 90, 100, 110]) & c.metro.notna()]
g = sp.groupby(["speed_kmh", "metro"]).agg(n=("ksi", "size"), ksi=("ksi", "mean"), fatal=("fatal", "mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for region, color in [("Greater Sydney/Newcastle/Wollongong", BLUE), ("Rest of NSW", ORANGE)]:
    d = g[(g.metro == region) & (g.n >= 100)]
    axes[0].plot(d.speed_kmh, d.ksi * 100, marker="o", lw=2, ms=6, color=color, label=region)
    axes[1].plot(d.speed_kmh, d.fatal * 100, marker="o", lw=2, ms=6, color=color, label=region)
axes[0].set_title("Fatal or serious share by speed limit"); axes[0].set_ylabel("% of crashes")
axes[1].set_title("Fatal share by speed limit"); axes[1].set_ylabel("% of crashes")
for ax in axes:
    ax.set_xlabel("Speed limit (km/h)"); ax.set_ylim(0); ax.set_xticks([40, 50, 60, 70, 80, 90, 100, 110])
axes[0].legend(frameon=False, fontsize=8)
fig.text(0.01, 0.01, "Groups with fewer than 100 crashes are hidden.", fontsize=8, color=INK2)
save(fig, "04_severity_by_speed_region.png")
tables["speed_region"] = g

# 5. Region summary
reg = c[c.region.notna()].groupby("region").agg(crashes=("ksi", "size"), ksi_rate=("ksi", "mean"),
                                                fatal_rate=("fatal", "mean"), killed=("No. killed", "sum"))
reg["share_of_crashes"] = reg.crashes / reg.crashes.sum()
reg["share_of_deaths"] = reg.killed / reg.killed.sum()
tables["region"] = reg

# 6. Severity by first impact type: 100% stacked bar (fatal or serious vs other outcomes)
fi = c[c["First impact type"] != "Unknown"].groupby("First impact type").agg(n=("ksi", "size"), ksi=("ksi", "mean"),
                                                                             fatal=("fatal", "mean"))
fi = fi[fi.n >= 100].sort_values("ksi")
ksi_pct, other_pct = fi.ksi * 100, 100 - fi.ksi * 100
fig, ax = plt.subplots(figsize=(8, 4.4))
y = np.arange(len(fi))
ax.barh(y, ksi_pct, color=BLUE, height=0.62, label="Fatal or serious injury (KSI)")
ax.barh(y, other_pct, left=ksi_pct, color=ORANGE, height=0.62, label="Moderate, minor or no injury",
        edgecolor=SURFACE, linewidth=2)
for i, (k, o, n) in enumerate(zip(ksi_pct, other_pct, fi.n)):
    ax.text(k / 2, i, f"{k:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(k + o / 2, i, f"{o:.0f}%", ha="center", va="center", fontsize=8, color=INK)
    ax.text(101.5, i, f"n={n:,}", ha="left", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, fi.index)
ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.grid(False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_title("Crash outcomes by first impact type", pad=26)
ax.set_xlabel("Share of crashes of that type")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 0.995),
          borderaxespad=0, handlelength=1.2)
fig.text(0.01, 0.01, "Impact types with at least 100 crashes, NSW 2020–2024.", fontsize=7, color=INK2)
save(fig, "05_ksi_by_impact_type.png")
tables["impact"] = fi

# 7. Crash location map (hexbin) with NSW / Sydney boundaries and major cities
import geopandas as gpd
from matplotlib.patches import Rectangle
BND = ROOT / "data" / "external" / "boundaries"
nsw_outline = gpd.read_file(BND / "nsw_state.geojson")
greater_syd = gpd.read_file(BND / "greater_sydney.geojson")
lgas = gpd.read_file(BND / "nsw_lga_2024.geojson")
CITIES = {"Sydney": (151.2093, -33.8688), "Newcastle": (151.7817, -32.9283), "Wollongong": (150.8931, -34.4278)}
SYD_BOX = dict(lon=(149.9, 151.75), lat=(-34.4, -32.95))  # Greater Sydney extent, padded

geo = c.dropna(subset=["Latitude", "Longitude"])
geo = geo[geo.Longitude < 154]  # exclude Lord Howe Island from the mainland map
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

# Left: all crashes in NSW, count per hexagon (log scale)
ax = axes[0]
lgas.boundary.plot(ax=ax, color=GRID, linewidth=0.4, zorder=1)
hb = ax.hexbin(geo.Longitude, geo.Latitude, gridsize=90, bins="log", cmap=SEQ, mincnt=1, zorder=2)
nsw_outline.boundary.plot(ax=ax, color=INK2, linewidth=0.9, zorder=3)
ax.add_patch(Rectangle((SYD_BOX["lon"][0], SYD_BOX["lat"][0]), np.diff(SYD_BOX["lon"])[0], np.diff(SYD_BOX["lat"])[0],
                       fill=False, ec=INK, lw=0.9, ls="--", zorder=4))
label_pos = {"Newcastle": (153.9, -32.3), "Sydney": (153.9, -33.6), "Wollongong": (153.9, -34.9)}
for city, (lon, lat) in CITIES.items():
    ax.plot(lon, lat, "o", ms=5, mfc="white", mec=INK, mew=1.2, zorder=5)
    ax.annotate(city, xy=(lon, lat), xytext=label_pos[city], fontsize=9, fontweight="bold", color=INK,
                va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8), zorder=5)
ax.set_xlim(140.6, 156.2); ax.set_ylim(-37.8, -27.9)
ax.set_title("Crash locations, NSW 2020–2024"); ax.set_aspect(1.2); ax.grid(False)
fig.colorbar(hb, ax=ax, shrink=0.7, label="Crashes per cell (log scale)")

# Right: Sydney, share of crashes that are fatal or serious
ax = axes[1]
syd = geo[geo.Latitude.between(*SYD_BOX["lat"]) & geo.Longitude.between(*SYD_BOX["lon"])]
hb2 = ax.hexbin(syd.Longitude, syd.Latitude, C=syd.ksi * 100, reduce_C_function=np.mean,
                gridsize=60, cmap=SEQ, mincnt=15, vmin=10, vmax=45, zorder=2)
lgas.boundary.plot(ax=ax, color=INK2, linewidth=0.35, alpha=0.6, zorder=3)
greater_syd.boundary.plot(ax=ax, color=INK, linewidth=1.2, zorder=4)
syd_labels = {"Sydney CBD": (151.45, -33.75), "Newcastle": (151.95, -33.2), "Wollongong": (151.2, -34.32)}
for city, xy in zip(syd_labels, [CITIES["Sydney"], CITIES["Newcastle"], CITIES["Wollongong"]]):
    if SYD_BOX["lon"][0] < xy[0] < SYD_BOX["lon"][1] + 0.3 and SYD_BOX["lat"][0] < xy[1] < SYD_BOX["lat"][1]:
        ax.plot(*xy, "o", ms=5, mfc="white", mec=INK, mew=1.2, zorder=5)
        ax.annotate(city, xy=xy, xytext=syd_labels[city], fontsize=8, fontweight="bold", color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8), zorder=5)
ax.set_xlim(SYD_BOX["lon"][0], SYD_BOX["lon"][1] + 0.45); ax.set_ylim(*SYD_BOX["lat"])
ax.set_title("Greater Sydney: fatal or serious share (%)"); ax.set_aspect(1.2); ax.grid(False)
fig.colorbar(hb2, ax=ax, shrink=0.7, label="% fatal or serious (cells with 15+ crashes; capped 10–45)")
for ax in axes:
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
fig.text(0.01, 0.01, "Boundaries: ABS ASGS Ed. 3 (state, Greater Sydney, LGAs 2024), CC BY 4.0. "
         "Thin lines are council areas; the dashed box marks the right panel; the bold line on the right is Greater Sydney.", fontsize=7, color=INK2)
save(fig, "06_crash_map.png")

# 8. Top LGAs by crashes and by KSI rate
lga = c.groupby("LGA").agg(crashes=("ksi", "size"), ksi=("ksi", "sum"), ksi_rate=("ksi", "mean"),
                           killed=("No. killed", "sum"))
tables["lga"] = lga.sort_values("crashes", ascending=False)

with pd.ExcelWriter(ROOT / "reports" / "eda_tables.xlsx") as xw:
    for k, v in tables.items():
        v.to_excel(xw, sheet_name=k)
for k in ["severity", "yearly", "region", "impact"]:
    print(f"\n== {k}\n", tables[k].round(3).to_string())
print("\n== top LGAs by KSI count\n", lga.sort_values("ksi", ascending=False).head(10).round(3).to_string())
print("\n== speed/region\n", g.round(3).to_string())
print("\nKSI overall", round(c.ksi.mean(), 4), "fatal", round(c.fatal.mean(), 4), "n", len(c))
