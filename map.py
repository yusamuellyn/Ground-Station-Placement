import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
# FOr every GDP location, you'll have one ground station
# -----------------------------
# FILES
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "cloud_fraction_2025_annual_avg (1).csv")
world_shp  = os.path.join(script_dir, "ne_110m_admin_0_countries.shp")
gdp_csv    = os.path.join(script_dir, "gdp.csv")

# -----------------------------
# TUNABLE PARAMETERS
# -----------------------------
CCF_THRESHOLD = 0.60   # max acceptable cloud fraction
GRID_STEP     = 2.0    # degrees
MAX_STATIONS  = 100    # total stations to place
MIN_DIST_DEG  = 15.0   # starting min degrees between stations (relaxed automatically)
ALPHA         = 0.0    # 0.0 = pure cloud score, 1.0 = pure GDP score

# -----------------------------
# LOAD CLOUD DATA
# -----------------------------
cloud_df         = pd.read_csv(csv_path, index_col=0)
cloud_df.index   = pd.to_numeric(cloud_df.index)
cloud_df.columns = pd.to_numeric(cloud_df.columns)

# -----------------------------
# LOAD WORLD MAP
# -----------------------------
world = gpd.read_file(world_shp)
world = world[world["CONTINENT"] != "Antarctica"]

# -----------------------------
# LOAD & MERGE GDP
# -----------------------------
gdp_raw = pd.read_csv(gdp_csv, skiprows=4, engine="python", on_bad_lines="skip")
gdp_raw = gdp_raw.loc[:, ~gdp_raw.columns.str.contains('^Unnamed')]

years = [c for c in gdp_raw.columns if str(c).isdigit()]
latest_year = next(
    (y for y in sorted(years, reverse=True) if gdp_raw[y].notna().sum() > 50), None
)
if latest_year is None:
    raise Exception("No valid GDP year found")

# Rename to GDP_COUNTRY to avoid collision with shapefile's own NAME column
gdp = gdp_raw[["Country Name", latest_year]].copy()
gdp.columns = ["GDP_COUNTRY", "GDP"]
gdp["GDP"] = pd.to_numeric(gdp["GDP"], errors="coerce")
gdp = gdp.dropna()

ALIASES = {
    "United States of America": "United States",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    "Republic of the Congo": "Congo, Rep.",
    "United Republic of Tanzania": "Tanzania",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Czechia": "Czech Republic",
    "South Korea": "Korea, Rep.",
    "North Korea": "Korea, Dem. People's Rep.",
    "Iran": "Iran, Islamic Rep.",
    "Russia": "Russian Federation",
    "Syria": "Syrian Arab Republic",
    "Venezuela": "Venezuela, RB",
    "Yemen": "Yemen, Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Laos": "Lao PDR",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Slovakia": "Slovak Republic",
    "Swaziland": "Eswatini",
}

world["GDP_NAME"] = world["ADMIN"].replace(ALIASES)
world_gdp = world.merge(gdp, left_on="GDP_NAME", right_on="GDP_COUNTRY", how="left")

median_gdp = world_gdp["GDP"].median()
world_gdp["GDP"] = world_gdp["GDP"].fillna(median_gdp * 0.01)
world_gdp["gdp_weight"] = world_gdp["GDP"] / world_gdp["GDP"].sum()

matched   = world_gdp["GDP_COUNTRY"].notna().sum()
unmatched = world_gdp[world_gdp["GDP_COUNTRY"].isna()]["ADMIN"].tolist()
print(f"GDP matched: {matched} countries")
if unmatched:
    print(f"Unmatched (using floor GDP): {unmatched[:15]}{'...' if len(unmatched) > 15 else ''}")

# -----------------------------
# BUILD LAND MASK & GDP LOOKUP
# -----------------------------
world_gdp = world_gdp.reset_index(drop=True)
sindex = world_gdp.sindex

print("Building land mask...")
try:
    world_union = world_gdp.geometry.union_all()
except AttributeError:
    world_union = world_gdp.geometry.unary_union

def is_land(lat, lon):
    return world_union.contains(Point(lon, lat))

def get_gdp_weight(lat, lon):
    point = Point(lon, lat)
    hits  = list(sindex.intersection(point.bounds))
    for i in hits:
        row = world_gdp.iloc[i]
        if row.geometry.contains(point):
            return float(row["gdp_weight"])
    return 0.0

# -----------------------------
# CLOUD COVERAGE FUNCTIONS
# -----------------------------
def get_cloud_coverage(lat, lon):
    lat_idx = int(np.abs(cloud_df.index.values   - lat).argmin())
    lon_idx = int(np.abs(cloud_df.columns.values - lon).argmin())
    value   = cloud_df.iloc[lat_idx, lon_idx]
    return None if pd.isna(value) else float(value)

def consider_cloud_coverage(lat, lon) -> tuple[float, float, bool]:
    ccf = get_cloud_coverage(lat, lon)
    if ccf is None:
        return lat, lon, False

    lat_diff = 0.555
    lon_diff = 61.7 / (111.1 * math.cos(math.radians(lat)))

    neighbors = [
        (lat,            lon + lon_diff),
        (lat,            lon - lon_diff),
        (lat + lat_diff, lon),
        (lat - lat_diff, lon),
        (lat + lat_diff, lon + lon_diff),
        (lat + lat_diff, lon - lon_diff),
        (lat - lat_diff, lon + lon_diff),
        (lat - lat_diff, lon - lon_diff),
    ]

    def score_point(lat_, lon_):
        c = get_cloud_coverage(lat_, lon_)
        if c is None:
            return None

        g = get_gdp_weight(lat_, lon_)

        cloud_score = 1 - c  # higher = better
        gdp_score = g        # already normalized

        return ALPHA * gdp_score + (1 - ALPHA) * cloud_score

    best_score = score_point(lat, lon)
    if best_score is None:
        return lat, lon, False

    best_lat, best_lon = lat, lon

    for new_lat, new_lon in neighbors:
        s = score_point(new_lat, new_lon)
        if s is None:
            continue

        if s > best_score:
            best_score = s
            best_lat = new_lat
            best_lon = new_lon

    # enforce cloud constraint at final position
    final_ccf = get_cloud_coverage(best_lat, best_lon)
    if final_ccf is not None and final_ccf <= CCF_THRESHOLD:
        return best_lat, best_lon, True

    return lat, lon, False

# -----------------------------
# GENERATE CANDIDATE STATIONS (land only)
# -----------------------------
print("Scanning grid for valid stations...")
accepted, rejected = [], []

for lat in np.arange(-60, 80, GRID_STEP):
    for lon in np.arange(-180, 180, GRID_STEP):
        if not is_land(lat, lon):
            continue
        best_lat, best_lon, valid = consider_cloud_coverage(lat, lon)
        if valid:
            accepted.append({
                "lat":        best_lat,
                "lon":        best_lon,
                "ccf":        get_cloud_coverage(best_lat, best_lon),
                "gdp_weight": get_gdp_weight(best_lat, best_lon),
            })
        else:
            rejected.append({"lat": lat, "lon": lon})

accepted_df = pd.DataFrame(accepted)
rejected_df = pd.DataFrame(rejected)

# -----------------------------
# COMBINED SCORE (cloud + GDP)
# -----------------------------
accepted_df["cloud_norm"] = (
    1 - (accepted_df["ccf"] - accepted_df["ccf"].min()) /
    (accepted_df["ccf"].max() - accepted_df["ccf"].min() + 1e-9)
)  # higher = clearer sky

accepted_df["gdp_norm"] = (
    (accepted_df["gdp_weight"] - accepted_df["gdp_weight"].min()) /
    (accepted_df["gdp_weight"].max() - accepted_df["gdp_weight"].min() + 1e-9)
)  # higher = more economically important

accepted_df["combined_score"] = (
    ALPHA       * accepted_df["gdp_norm"] +
    (1 - ALPHA) * accepted_df["cloud_norm"]
)

# -----------------------------
# SPREAD STATIONS GEOGRAPHICALLY
# -----------------------------
def spread_stations(df, max_stations, min_dist):
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    selected = []
    for _, row in df.iterrows():
        if len(selected) >= max_stations:
            break
        too_close = any(
            math.sqrt((row["lat"] - s["lat"])**2 + (row["lon"] - s["lon"])**2) < min_dist
            for s in selected
        )
        if not too_close:
            selected.append(row)
    return pd.DataFrame(selected)

# Relax MIN_DIST_DEG automatically until we hit MAX_STATIONS
min_dist = MIN_DIST_DEG
while True:
    result_df = spread_stations(accepted_df, MAX_STATIONS, min_dist)
    if len(result_df) >= MAX_STATIONS or min_dist < 0.5:
        break
    min_dist = round(min_dist - 0.5, 1)
    print(f"Only {len(result_df)} stations found, relaxing min distance to {min_dist}°...")

accepted_df = result_df.reset_index(drop=True)
print(f"Final: {len(accepted_df)} stations placed at min separation {min_dist}°")
print(f"Rejected: {len(rejected_df)}")

# Save
out_csv = os.path.join(script_dir, "optimal_ground_stations.csv")
accepted_df.to_csv(out_csv, index=False)
print(f"Stations saved to {out_csv}")

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(18, 9))

# Cloud fraction heatmap background
lons = cloud_df.columns.values
lats = cloud_df.index.values
lon_grid, lat_grid = np.meshgrid(lons, lats)

pcm = ax.pcolormesh(
    lon_grid, lat_grid, cloud_df.values,
    cmap="Blues", vmin=0, vmax=1,
    alpha=0.75, zorder=1
)
plt.colorbar(pcm, ax=ax, label="Annual Mean Cloud Fraction (0–1)", shrink=0.7)

# Country borders
world.boundary.plot(ax=ax, linewidth=0.5, color="black", zorder=2)

# Rejected land points
if not rejected_df.empty:
    ax.scatter(
        rejected_df["lon"], rejected_df["lat"],
        c="lightgray", s=8, marker="o",
        alpha=0.4, zorder=3,
        label=f"Rejected ({len(rejected_df)})"
    )

# Accepted stations colored by combined score
if not accepted_df.empty:
    sc = ax.scatter(
        accepted_df["lon"], accepted_df["lat"],
        c=accepted_df["combined_score"],
        cmap="plasma",
        vmin=0, vmax=1,
        s=55, marker="^",
        edgecolors="black", linewidths=0.4,
        alpha=0.95, zorder=4,
        label=f"Accepted stations ({len(accepted_df)})"
    )
    plt.colorbar(sc, ax=ax, label=f"Combined Score (α={ALPHA}: GDP / cloud)", shrink=0.7)

ax.set_xlim(-180, 180)
ax.set_ylim(-65, 85)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    f"GDP + Cloud-Coverage–Weighted Ground Station Candidates\n"
    f"Grid step: {GRID_STEP}°  |  CCF threshold: ≤ {CCF_THRESHOLD}  |  "
    f"Stations: {len(accepted_df)}  |  α={ALPHA}  |  Min separation: {min_dist}°"
)
ax.legend(loc="lower left", fontsize=9)
plt.grid(True, linewidth=0.2, color="gray")
plt.tight_layout()

out_png = os.path.join(script_dir, "ground_stations_cloud_only.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Map saved to {out_png}")
plt.show()