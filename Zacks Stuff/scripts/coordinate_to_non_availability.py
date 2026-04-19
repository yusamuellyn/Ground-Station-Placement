import pandas as pd
import numpy as np
import os
import glob

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Match monthly files in data/raw/
pattern = os.path.join(_root, "data", "raw", "MODAL2_M_CLD_FR_*_rgb_720x360.CSV")
files = sorted(glob.glob(pattern))

print(f"Found {len(files)} files:")
for f in files:
    print(" ", os.path.basename(f))

if len(files) == 0:
    raise FileNotFoundError("No matching CSV files found. Check filenames and directory.")

# Generate coordinate labels
lats = np.arange(89.75, -90, -0.5)   # 360 values
lons = np.arange(-179.75, 180, 0.5)  # 720 values

# Stack all monthly data
monthly_arrays = []

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Loading {filename}...")

    raw = pd.read_csv(filepath, header=None)
    data = raw.values.astype(float)
    data = data[:360, :720]
    data[data == 99999.0] = np.nan
    monthly_arrays.append(data)

if len(monthly_arrays) != 12:
    print(f"WARNING: Expected 12 files but got {len(monthly_arrays)}. Average will be across {len(monthly_arrays)} months.")

# Stack into 3D array (months, lats, lons) and average
stacked = np.stack(monthly_arrays, axis=0)
CLOUD_THRESHOLD = 0.6  # ← your tunable weight, adjust as needed

# Apply threshold: 1 if cloudy, 0 if not, NaN stays NaN
binary = np.where(stacked > CLOUD_THRESHOLD, 1.0, 0.0)
# NaN cells should stay NaN (not get a 0 or 1)
binary[np.isnan(stacked)] = np.nan

# Sum the 1s across months, divide by 12 → fraction of months that are "too cloudy"
cloudy_fraction = np.nansum(binary, axis=0) / 12.0

# Save
output_df = pd.DataFrame(cloudy_fraction, index=lats, columns=lons)
output_df.to_csv(os.path.join(_root, "output", "lat_long_non_availability.csv"))



