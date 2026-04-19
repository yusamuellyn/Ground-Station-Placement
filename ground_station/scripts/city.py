"""
Top U.S. counties by GDP → suggested ground-station locations
(with water points snapped to nearest land + optimized OGS pairs + maps)
"""

from __future__ import annotations
from cloud_coverage_check import consider_availability_greedy
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.express as px
import geopandas as gpd

_ROOT = Path(__file__).resolve().parent.parent
MERGED_PATH = _ROOT / "data" / "processed" / "counties_gdp_latlon_2024.csv"
GDP_PATH = _ROOT / "data" / "raw" / "CAGDP2.csv"
GAZ_PATH = _ROOT / "data" / "raw" / "2024_Gaz_counties_national.txt"

LAND_SHAPEFILE = _ROOT / "data" / "raw" / "cb_2024_us_state_20m.shp"

YEAR_COL = "2024"
TOP_N = 30

OUT_STATIONS = _ROOT / "output" / "csv" / "ground_stations_top_gdp.csv"
OUT_MAP = _ROOT / "output" / "maps" / "ground_stations_map.html"
OUT_BAR = _ROOT / "output" / "maps" / "gdp_bar.html"
OUT_OGS = _ROOT / "output" / "csv" / "optimized_ground_stations.csv"
OUT_OGS_MAP = _ROOT / "output" / "maps" / "optimized_ground_stations_map.html"
OUT_COMBINED_MAP = _ROOT / "output" / "maps" / "comparison_map.html"


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

    return gaz[["fips", "latitude", "longitude"]].dropna()


def load_counties_gdp_geo() -> pd.DataFrame:
    if Path(MERGED_PATH).exists():
        df = pd.read_csv(MERGED_PATH)
    else:
        gdp = _load_gdp_counties(GDP_PATH)
        gaz = _load_gazetteer(GAZ_PATH)
        df = gdp.merge(gaz, on="fips", how="left")

    return df

def snap_points_to_land(df: pd.DataFrame, shapefile_path: Path) -> pd.DataFrame:
    land = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    land_union = land.geometry.union_all()

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )

    def snap(point):
        if land_union.contains(point):
            return point
        return land_union.boundary.interpolate(
            land_union.boundary.project(point)
        )

    gdf["geometry"] = gdf["geometry"].apply(snap)
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x

    return pd.DataFrame(gdf.drop(columns="geometry"))

def main():
    df = load_counties_gdp_geo()

    geo = df.dropna(subset=["latitude", "longitude", "gdp_2024"]).copy()
    top = geo.sort_values("gdp_2024", ascending=False).head(TOP_N).copy()

    top["gdp_billions"] = top["gdp_2024"] / 1_000_000
    top.insert(0, "station_id", range(1, len(top) + 1))

    # Snap to land
    top = snap_points_to_land(top, LAND_SHAPEFILE)

    top.to_csv(OUT_STATIONS, index=False)
    
    results = []

    for _, row in top.iterrows():
        ogs1, ogs2, availability, valid = consider_availability_greedy(
            row["latitude"], row["longitude"]
        )

        results.append({
            "station_id": row["station_id"],
            "ogs1_lon": ogs1[0],
            "ogs1_lat": ogs1[1],
            "ogs2_lon": ogs2[0],
            "ogs2_lat": ogs2[1],
            "availability": availability,
            "valid": valid,
        })

    availability_df = pd.DataFrame(results)
    ogs_points = []

    for _, row in availability_df.iterrows():
        ogs_points.append({
            "station_id": row["station_id"],
            "type": "OGS1",
            "latitude": row["ogs1_lat"],
            "longitude": row["ogs1_lon"],
        })
        ogs_points.append({
            "station_id": row["station_id"],
            "type": "OGS2",
            "latitude": row["ogs2_lat"],
            "longitude": row["ogs2_lon"],
        })

    ogs_df = pd.DataFrame(ogs_points)
    ogs_df.to_csv(OUT_OGS, index=False)

    # orginal map
    map_fig = px.scatter_geo(
        top,
        lat="latitude",
        lon="longitude",
        scope="usa",
        title="Original Ground Stations",
    )
    map_fig.write_html(OUT_MAP, include_plotlyjs=True)


    # ogs map
    ogs_map = px.scatter_geo(
        ogs_df,
        lat="latitude",
        lon="longitude",
        scope="usa",
        color="type",
        title="Optimized Ground Stations",
    )
    ogs_map.write_html(OUT_OGS_MAP, include_plotlyjs=True)
    
#  combined map 

    top_map = top.copy()
    top_map["type"] = "Original"
    top_map["size"] = 9 

    ogs_plot = ogs_df.copy()
    ogs_plot["size"] = 4   

    combined = pd.concat([
        top_map[["station_id", "latitude", "longitude", "type", "size"]],
        ogs_plot[["station_id", "latitude", "longitude", "type", "size"]],
    ])

    combined_map = px.scatter_geo(
        combined,
        lat="latitude",
        lon="longitude",
        scope="usa",
        color="type",
        size="size",              
        size_max=9,              
        hover_data={"station_id": True},
        title="Original vs Optimized Ground Stations",
    )

    combined_map.update_traces(marker=dict(opacity=0.85, line=dict(width=1)))

    combined_map.write_html(OUT_COMBINED_MAP, include_plotlyjs=True)

    # Open maps
    webbrowser.open(Path(OUT_MAP).resolve().as_uri())
    webbrowser.open(Path(OUT_OGS_MAP).resolve().as_uri())
    webbrowser.open(Path(OUT_COMBINED_MAP).resolve().as_uri())

    print("\nDone.\n")


if __name__ == "__main__":
    main()