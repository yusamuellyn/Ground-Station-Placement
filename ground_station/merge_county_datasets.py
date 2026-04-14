"""
Merge CAGDP2.csv (county GDP) with 2024_Gaz_counties_national.txt (county lat/lon)
on 5-digit county FIPS. Writes counties_gdp_latlon_2024.csv
"""

from __future__ import annotations

import pandas as pd

GDP_PATH = "CAGDP2.csv"
GAZ_PATH = "2024_Gaz_counties_national.txt"
OUT_PATH = "counties_gdp_latlon_2024.csv"
YEAR_COL = "2024"


def load_gdp_counties(path: str) -> pd.DataFrame:
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


def load_gazetteer(path: str) -> pd.DataFrame:
    gaz = pd.read_csv(path, sep="\t", encoding="latin1", dtype=str)
    gaz.columns = gaz.columns.str.strip()
    for c in gaz.columns:
        gaz[c] = gaz[c].str.strip() if gaz[c].dtype == object else gaz[c]
    gaz["fips"] = gaz["GEOID"].astype(str).str.zfill(5)
    gaz["latitude"] = pd.to_numeric(gaz["INTPTLAT"], errors="coerce")
    gaz["longitude"] = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")
    return gaz[["fips", "USPS", "NAME", "latitude", "longitude"]].dropna(
        subset=["latitude", "longitude"]
    )


def main() -> None:
    gdp = load_gdp_counties(GDP_PATH)
    gaz = load_gazetteer(GAZ_PATH)

    # Left: keep every county that appears in BEA GDP; NaN lat/lon if not in gazetteer
    merged = gdp.merge(gaz, on="fips", how="left")

    # Friendly column order / names
    merged = merged.rename(
        columns={
            "USPS": "state",
            "NAME": "county_name_census",
        }
    )
    merged = merged[
        [
            "fips",
            "state",
            "county_bea",
            "county_name_census",
            "gdp_2024",
            "latitude",
            "longitude",
        ]
    ]

    merged.to_csv(OUT_PATH, index=False)

    n_gdp = len(gdp)
    n_with_geo = merged["latitude"].notna().sum()
    print(f"GDP county rows: {n_gdp}")
    print(f"Gazetteer county rows: {len(gaz)}")
    print(f"Merged rows: {len(merged)} (all GDP counties)")
    print(f"Rows with lat/lon: {int(n_with_geo)}; missing coordinates: {int(n_gdp - n_with_geo)}")
    if n_with_geo < n_gdp:
        miss = merged.loc[merged["latitude"].isna(), "fips"].tolist()
        print(f"  FIPS missing from gazetteer (sample): {miss[:8]}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
