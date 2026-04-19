import math
from cloud_coverage_check import get_na, consider_availability, consider_availability_greedy, MIN_AVAILABILITY

print("=" * 60)
print("CLOUD COVERAGE CHECK — DEBUG & VERIFICATION")
print("=" * 60)

# ── 1. get_na sanity checks ───────────────────────────────────
print("\n── 1. get_na() spot checks ─────────────────────────────")

test_coords = [
    ("Sahara Desert",      23.50,   13.00),
    ("Amazon Rainforest",  -3.00,  -60.00),
    ("London",             51.50,   -0.25),
    ("Los Angeles",        34.00, -118.25),
    ("New York City",      40.75,  -74.00),
    ("Atacama Desert",    -23.00,  -68.00),
]

for name, lat, lon in test_coords:
    val = get_na(lat, lon)
    if val is None:
        print(f"   {name:<22} ({lat:>6}, {lon:>7})  → None (NaN in CSV)")
    else:
        print(f"   {name:<22} ({lat:>6}, {lon:>7})  → na={val:.4f}  avail={1-val:.4f}")

# ── 2. Return value structure ─────────────────────────────────
print("\n── 2. Return value structure ───────────────────────────")

for name, lat, lon in test_coords[:3]:
    result_g = consider_availability_greedy(lat, lon)
    result_b = consider_availability(lat, lon)

    assert len(result_g) == 4, f"[FAIL] greedy returned {len(result_g)} values, expected 4"
    assert len(result_b) == 4, f"[FAIL] brute returned {len(result_b)} values, expected 4"

    ogs1, ogs2, avail, ok = result_g
    assert len(ogs1) == 2, f"[FAIL] ogs1 should be (lon, lat) tuple, got {ogs1}"
    assert len(ogs2) == 2, f"[FAIL] ogs2 should be (lon, lat) tuple, got {ogs2}"
    assert isinstance(avail, float), f"[FAIL] availability should be float, got {type(avail)}"
    assert isinstance(ok, bool), f"[FAIL] ok should be bool, got {type(ok)}"

    print(f"   [PASS] {name:<22} — structure correct")

# ── 3. Availability value is in [0, 1] ────────────────────────
print("\n── 3. Availability values in [0, 1] ────────────────────")

for name, lat, lon in test_coords:
    _, _, avail_g, _ = consider_availability_greedy(lat, lon)
    _, _, avail_b, _ = consider_availability(lat, lon)

    g_ok = 0.0 <= avail_g <= 1.0
    b_ok = 0.0 <= avail_b <= 1.0

    print(f"   {'[PASS]' if g_ok else '[FAIL]'} {name:<22} greedy={avail_g:.4f}  brute={avail_b:.4f}")

# ── 4. OGS1 != OGS2 ──────────────────────────────────────────
print("\n── 4. OGS1 and OGS2 are not the same location ──────────")

for name, lat, lon in test_coords:
    ogs1_g, ogs2_g, _, _ = consider_availability_greedy(lat, lon)
    ogs1_b, ogs2_b, _, _ = consider_availability(lat, lon)

    g_diff = ogs1_g != ogs2_g
    b_diff = ogs1_b != ogs2_b

    print(f"   {'[PASS]' if g_diff else '[WARN]'} {name:<22} greedy  OGS1={ogs1_g}  OGS2={ogs2_g}")
    print(f"   {'[PASS]' if b_diff else '[WARN]'} {name:<22} brute   OGS1={ogs1_b}  OGS2={ogs2_b}")

# ── 5. ok flag matches MIN_AVAILABILITY ──────────────────────
print(f"\n── 5. ok flag consistent with MIN_AVAILABILITY={MIN_AVAILABILITY} ──")

for name, lat, lon in test_coords:
    _, _, avail_g, ok_g = consider_availability_greedy(lat, lon)
    _, _, avail_b, ok_b = consider_availability(lat, lon)

    g_consistent = ok_g == (avail_g >= MIN_AVAILABILITY)
    b_consistent = ok_b == (avail_b >= MIN_AVAILABILITY)

    print(f"   {'[PASS]' if g_consistent else '[FAIL]'} {name:<22} greedy avail={avail_g:.3f} ok={ok_g}")
    print(f"   {'[PASS]' if b_consistent else '[FAIL]'} {name:<22} brute  avail={avail_b:.3f} ok={ok_b}")

# ── 6. Greedy vs brute comparison ────────────────────────────
print("\n── 6. Greedy vs brute-force availability comparison ────")
print("   (brute should always be >= greedy since it checks all pairs)")

all_ok = True
for name, lat, lon in test_coords:
    _, _, avail_g, _ = consider_availability_greedy(lat, lon)
    _, _, avail_b, _ = consider_availability(lat, lon)
    diff = avail_b - avail_g
    ok = avail_b >= avail_g - 1e-9  # small tolerance for float precision
    all_ok = all_ok and ok
    flag = "[PASS]" if ok else "[FAIL]"
    print(f"   {flag} {name:<22} brute={avail_b:.4f}  greedy={avail_g:.4f}  diff={diff:+.4f}")

if all_ok:
    print("\n   [PASS] Brute-force always >= greedy as expected")
else:
    print("\n   [FAIL] Greedy beat brute-force — logic error in one of the functions")

# ── 7. NaN input handling ─────────────────────────────────────
print("\n── 7. NaN / ocean coordinate handling ──────────────────")

ocean_coords = [
    ("Mid-Atlantic Ocean", 30.0, -40.0),
    ("South Pacific",     -40.0, -130.0),
]

for name, lat, lon in ocean_coords:
    val = get_na(lat, lon)
    if val is None:
        result = consider_availability_greedy(lat, lon)
        assert len(result) == 4, "[FAIL] Wrong return length for None input"
        print(f"   [PASS] {name:<22} get_na=None, function returned safely: {result}")
    else:
        print(f"   [INFO] {name:<22} has data (val={val:.4f}), not a NaN cell in this CSV")

print()
print("=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
