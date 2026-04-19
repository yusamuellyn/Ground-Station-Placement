import pandas as pd
import numpy as np
import os
import math

# Per every GDP location find optimal ground statin based on formula

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(_root, "data", "processed", "lat_long_non_availability.csv")

# Annual avg CSV already has lat/lon labels embedded from when we saved it
cloud_df_clean = pd.read_csv(csv_path, index_col=0)
cloud_df_clean.index = pd.to_numeric(cloud_df_clean.index)
cloud_df_clean.columns = pd.to_numeric(cloud_df_clean.columns)

MIN_AVAILABILITY = 0.75

# For availability optimization
N = 2
M = 1

def get_na(lat, lon):
    nearest_lat = min(cloud_df_clean.index, key=lambda x: abs(x - lat))
    nearest_lon = min(cloud_df_clean.columns, key=lambda x: abs(x - lon))
    value = cloud_df_clean.loc[nearest_lat, nearest_lon]
    if pd.isna(value):
        return None
    return float(value)



def get_na(lat, lon):
    nearest_lat = min(cloud_df_clean.index, key=lambda x: abs(x - lat))
    nearest_lon = min(cloud_df_clean.columns, key=lambda x: abs(x - lon))
    value = cloud_df_clean.loc[nearest_lat, nearest_lon]
    if pd.isna(value):
        return None
    return float(value)
 
 
def consider_availability_greedy(lat, lon) -> tuple[tuple[float, float], tuple[float, float], float, bool]:
    ogs_na = get_na(lat, lon)
 
    # BUG FIX 2: early return now matches 4-value signature
    if ogs_na is None:
        return (lon, lat), (lon, lat), 0.0, False
 
    lat_diff = 0.555
    lon_diff = 61.7 / (111.1 * math.cos(math.radians(lat)))
 
    # BUG FIX 1: added missing comma after (lat, lon)
    ogs_possibilities = [
        (lat, lon),
        (lat,            lon + lon_diff),
        (lat,            lon - lon_diff),
        (lat + lat_diff, lon),
        (lat - lat_diff, lon),
        (lat + lat_diff, lon + lon_diff),
        (lat + lat_diff, lon - lon_diff),
        (lat - lat_diff, lon + lon_diff),
        (lat - lat_diff, lon - lon_diff),
    ]
 
    # Pass 1: find OGS1 — the single best (lowest non-availability) candidate
    lowest_na = ogs_na
    ogs1_lat = lat
    ogs1_lon = lon
 
    for new_lat, new_lon in ogs_possibilities:
        neighbor_na = get_na(new_lat, new_lon)
        if neighbor_na is None:
            continue
        if neighbor_na < lowest_na:
            lowest_na = neighbor_na
            ogs1_lat = new_lat
            ogs1_lon = new_lon
 
    # Pass 2: find OGS2 — best remaining candidate, skipping OGS1
    lowest_na = ogs_na
    if (ogs1_lat == lat and ogs1_lon == lon):
        ogs2_lat = lat
        ogs2_lon = lon + lon_diff
    else:
        ogs2_lat = lat
        ogs2_lon = lon
 
    for new_lat, new_lon in ogs_possibilities:
        if new_lat == ogs1_lat and new_lon == ogs1_lon:
            continue
        neighbor_na = get_na(new_lat, new_lon)
        if neighbor_na is None:
            continue
        if neighbor_na < lowest_na:

            lowest_na = neighbor_na
            ogs2_lat = new_lat
            ogs2_lon = new_lon
    print(get_na(ogs1_lat, ogs1_lon) , " " , get_na(ogs2_lat, ogs2_lon) , "\n")
    greedy_availability = 1.0 - (get_na(ogs1_lat, ogs1_lon) * get_na(ogs2_lat, ogs2_lon))
 
    if greedy_availability >= MIN_AVAILABILITY:
        return (ogs1_lon, ogs1_lat), (ogs2_lon, ogs2_lat), greedy_availability, True
 
    return (ogs1_lon, ogs1_lat), (ogs2_lon, ogs2_lat), greedy_availability, False



def consider_availability(lat, lon) -> tuple[tuple[float, float], tuple[float, float], float, bool]:
    ogs_na = get_na(lat, lon)

    if ogs_na is None:
        return (lon, lat), (lon, lat), 0.0, False

    lat_diff = 0.555
    lon_diff = 61.7 / (111.1 * math.cos(math.radians(lat)))

    neighbors = [
        (lat, lon),
        (lat,             lon + lon_diff),
        (lat,             lon - lon_diff),
        (lat + lat_diff,  lon),
        (lat - lat_diff,  lon),
        (lat + lat_diff,  lon + lon_diff),
        (lat + lat_diff,  lon - lon_diff),
        (lat - lat_diff,  lon + lon_diff),
        (lat - lat_diff,  lon - lon_diff)
    ]

    opt_a = 0.0
    lowest_na = ogs_na

    ogs1_lat = lat
    ogs1_lon = lon

    ogs2_lat = lat
    ogs2_lon = lon

    for new_lat1, new_lon1 in neighbors:
        for new_lat2, new_lon2 in neighbors:
            if (new_lat1 == new_lat2 and new_lon1 == new_lon2):
                continue
            na1 = get_na(new_lat1, new_lon1)
            na2 = get_na(new_lat2, new_lon2)
            if na1 is None or na2 is None:
                continue
            availability = 1 - (na1 * na2)
            if availability > opt_a:
                opt_a = availability
                ogs1_lat = new_lat1
                ogs1_lon = new_lon1
                ogs2_lat = new_lat2
                ogs2_lon = new_lon2

    if opt_a >= MIN_AVAILABILITY:
        return (ogs1_lon, ogs1_lat), (ogs2_lon, ogs2_lat), opt_a, True
    
    return (ogs1_lon, ogs1_lat), (ogs2_lon, ogs2_lat), opt_a, False
