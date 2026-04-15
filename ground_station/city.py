"""
Top U.S. counties by GDP → suggested ground-station locations
(with water points snapped to nearest land).

Requires:
- counties_gdp_latlon_2024.csv OR source datasets
- Land shapefile (e.g. cb_2023_us_state_20m.shp or ne_10m_land.shp)
"""

from __future__ import annotations
from cloud_coverage_check import get_na, consider_availability, consider_availability_greedy, MIN_AVAILABILITY
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point

MERGED_PATH = "counties_gdp_latlon_2024.csv"
GDP_PATH = "CAGDP2.csv"
GAZ_PATH = "2024_Gaz_counties_national.txt"

# 👉 CHANGE THIS to your shapefile path
LAND_SHAPEFILE = "cb_2024_us_state_20m.shp"

YEAR_COL = "2024"
TOP_N = 100

OUT_STATIONS = "ground_stations_top10_gdp.csv"
OUT_MAP = "ground_stations_top10_gdp_map.html"
OUT_BAR = "ground_stations_top10_gdp_bar.html"


# -----------------------------
# DATA LOADING
# -----------------------------
def _load_gdp_counties(path: str) -> pd.DataFrame:
    g = pd.read_csv(path, encoding="latin1", low_memory=False)
    g["GeoFIPS"] = g["GeoFIPS"].astype(str).str.strip().str.strip('"')

    g = g[g["LineCode"] == 1].copy()
    g = g[(g["GeoFIPS"] != "00000") & (~g["GeoFIPS"].str.endswith("000"))].copy()

    out = g[["GeoFIPS", "GeoName", YEAR_COL]].copy()
    out.columns = ["fips", "county_bea", "gdp_2024"]

    out["fips"] = out["fips"].astype(str).str.zfill(5)
    out["county_bea"] = out["county_bea"].str.strip()
    out["gdp_2024"] = pd.to_numeric(out["gdp_2024"], errors="coerce")

    return out.dropna(subset=["gdp_2024"])


def _load_gazetteer(path: str) -> pd.DataFrame:
    gaz = pd.read_csv(path, sep="\t", encoding="latin1", dtype=str)

    gaz.columns = gaz.columns.str.strip()
    for c in gaz.columns:
        if gaz[c].dtype == object:
            gaz[c] = gaz[c].str.strip()

    gaz["fips"] = gaz["GEOID"].astype(str).str.zfill(5)
    gaz["latitude"] = pd.to_numeric(gaz["INTPTLAT"], errors="coerce")
    gaz["longitude"] = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")

    return gaz[["fips", "USPS", "NAME", "latitude", "longitude"]].dropna(
        subset=["latitude", "longitude"]
    )


def _normalize_fips(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\D", "", regex=True)
    return s.str.zfill(5)


def load_counties_gdp_geo() -> pd.DataFrame:
    if Path(MERGED_PATH).exists():
        df = pd.read_csv(MERGED_PATH)
    else:
        gdp = _load_gdp_counties(GDP_PATH)
        gaz = _load_gazetteer(GAZ_PATH)

        df = gdp.merge(gaz, on="fips", how="left")
        df = df.rename(columns={"USPS": "state", "NAME": "county_name_census"})

    df["fips"] = _normalize_fips(df["fips"])
    return df


# -----------------------------
# LAND SNAPPING LOGIC
# -----------------------------
def snap_points_to_land(df: pd.DataFrame, shapefile_path: str) -> pd.DataFrame:
    print("\nLoading land shapefile...")
    land = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    print("Building land geometry union (this may take a moment)...")
    land_union = land.geometry.union_all()

    # Convert points to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )

    def snap(point):
        if land_union.contains(point):
            return point
        else:
            # Find nearest point on land boundary
            nearest = land_union.boundary.interpolate(
                land_union.boundary.project(point)
            )
            return nearest

    print("Snapping points that fall in water...")
    gdf["geometry"] = gdf["geometry"].apply(snap)

    # Update lat/lon
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x

    return pd.DataFrame(gdf.drop(columns="geometry"))


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    df = load_counties_gdp_geo()

    if "gdp_2024" not in df.columns:
        raise ValueError(f"{MERGED_PATH} must include column gdp_2024")

    geo = df.dropna(subset=["latitude", "longitude", "gdp_2024"]).copy()

    top = geo.sort_values("gdp_2024", ascending=False).head(TOP_N).copy()

    top["gdp_billions"] = top["gdp_2024"] / 1_000_000.0
    top.insert(0, "station_id", range(1, len(top) + 1))

    # ✅ APPLY LAND SNAP HERE
    top = snap_points_to_land(top, LAND_SHAPEFILE)

    print(f"\nTOP {TOP_N} COUNTIES (LAND-CORRECTED POINTS)\n")
    print(
        top[
            ["station_id", "fips", "county_bea", "latitude", "longitude"]
        ].to_string(index=False)
    )

    # Save CSV
    top.to_csv(OUT_STATIONS, index=False)
    
    results = []

    for _, row in top.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        ogs1, ogs2, availability, valid = consider_availability_greedy(lat, lon)

        results.append({
            "station_id": row["station_id"],
            "ogs1_lon": ogs1[0],
            "ogs1_lat": ogs1[1],
            "ogs2_lon": ogs2[0],
            "ogs2_lat": ogs2[1],
            "availability": availability,
            "meets_threshold": valid,
        })

    availability_df = pd.DataFrame(results)
    OUT_AVAILABILITY = "ground_station_availability.csv"
    availability_df.to_csv(OUT_AVAILABILITY, index=False)
    
    # Bar chart
    bar = px.bar(
        top,
        x="gdp_billions",
        y="county_bea",
        orientation="h",
        title=f"Top {TOP_N} U.S. counties by GDP ({YEAR_COL})",
    )
    bar.update_layout(yaxis={"categoryorder": "total ascending"})
    bar.write_html(OUT_BAR, include_plotlyjs="cdn")

    # Map
    map_fig = px.scatter_geo(
        top,
        lat="latitude",
        lon="longitude",
        scope="usa",
        projection="albers usa",
        hover_name="county_bea",
        hover_data={
            "station_id": True,
            "fips": True,
            "gdp_billions": ":.1f",
        },
        title="Ground stations (snapped to land)",
    )

    map_fig.update_traces(
        marker=dict(
            size=14,
            color="#c0392b",
            opacity=0.92,
            line=dict(width=1.5, color="black"),
        )
    )

    map_path = Path(OUT_MAP).resolve()
    map_fig.write_html(map_path, include_plotlyjs=True)

    print(f"\nOpening map: {map_path}")
    webbrowser.open(map_path.as_uri())

    print("\nDone.\n")


if __name__ == "__main__":
    main()
