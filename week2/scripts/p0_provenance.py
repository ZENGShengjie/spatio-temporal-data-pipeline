"""
P0: DATA PROVENANCE VERIFICATION
================================

Independently verify the new cleaned dataset against the raw H5 file.
Process:
  1. Load raw H5 from disk
  2. Independently aggregate raw slots to hourly
  3. Compare with the saved npz - bit-for-bit (or sum-for-sum)
  4. Verify each missing hour is filled from claimed source
  5. Report exact discrepancies with cell-level resolution
"""
import numpy as np
import h5py
from datetime import datetime, timedelta
import pandas as pd
import os

RAW_H5 = '/home/ubuntu/data/raw_bj/taxi_bj_gitee/BJ16_M32x32_T30_InOut.h5'
NEW_NPZ = '/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz'
HOURLY_COUNT = 'e:/amazon/evidence/hourly_count.npy'

# ============================================================
# STEP A: Load raw H5 (independent)
# ============================================================
print('=' * 80)
print('P0 - PROVENANCE VERIFICATION')
print('=' * 80)

with h5py.File(RAW_H5, 'r') as f:
    raw_data = f['data'][:]
    raw_dates = f['date'][:]
print(f'Raw data: shape={raw_data.shape}, dtype={raw_data.dtype}')
print(f'Raw dates: {len(raw_dates)} entries (bytes)')

# ============================================================
# STEP B: Aggregate raw slots to hourly (independent calculation)
# ============================================================
P4_START = datetime(2015, 11, 1, 0, 0)
P4_DAYS = 162
n_hours = P4_DAYS * 24  # 3888

# Build maps
slot_to_idx = {}  # (date_str, slot_1) -> raw_data index
for i, d in enumerate(raw_dates):
    ds = d.decode()
    date_str = ds[:8]
    slot_1 = int(ds[8:])
    slot_to_idx[(date_str, slot_1)] = i

# Aggregate to hourly
raw_hourly = np.full((n_hours, 2, 32, 32), np.nan)
raw_count = np.zeros(n_hours, dtype=int)

for h in range(n_hours):
    ts = P4_START + timedelta(hours=int(h))
    date_str = ts.strftime('%Y%m%d')
    slot_a = 2 * h + 1
    slot_b = 2 * h + 2

    has_a = (date_str, slot_a) in slot_to_idx
    has_b = (date_str, slot_b) in slot_to_idx

    if has_a:
        raw_hourly[h] = raw_data[slot_to_idx[(date_str, slot_a)]]
        raw_count[h] += 1
    if has_b:
        if has_a:
            raw_hourly[h] = raw_hourly[h] + raw_data[slot_to_idx[(date_str, slot_b)]]
        else:
            raw_hourly[h] = raw_data[slot_to_idx[(date_str, slot_b)]]
        raw_count[h] += 1

# Stats
n_complete = (raw_count == 2).sum()
n_partial = (raw_count == 1).sum()
n_missing = (raw_count == 0).sum()
print(f'\nIndependent hourly aggregation:')
print(f'  Complete (both slots): {n_complete} / {n_hours} = {n_complete/n_hours:.1%}')
print(f'  Partial (1 slot):       {n_partial}')
print(f'  Missing (no slots):     {n_missing}')

# ============================================================
# STEP C: Load new npz
# ============================================================
loaded = np.load(NEW_NPZ)
new_flow = loaded['flow']
print(f'\nNew npz: shape={new_flow.shape}')

# ============================================================
# STEP D: Verify complete hours (must match raw exactly)
# ============================================================
print('\n' + '=' * 80)
print('CHECK 1: Complete hours - new == raw (bit-for-bit)')
print('=' * 80)

exact_matches = 0
exact_mismatches = []
sum_matches = 0
sum_mismatches = []

for h in range(n_hours):
    if raw_count[h] == 2:
        # Exact bit-for-bit
        if np.array_equal(new_flow[h], raw_hourly[h]):
            exact_matches += 1
        else:
            exact_mismatches.append(h)
        # Sum-level (allow float tolerance)
        if abs(new_flow[h].sum() - raw_hourly[h].sum()) < 1e-3:
            sum_matches += 1
        else:
            sum_mismatches.append(h)

print(f'Complete hours: {n_complete}')
print(f'  Bit-for-bit matches: {exact_matches}')
print(f'  Bit-for-bit mismatches: {len(exact_mismatches)}')
print(f'  Sum-level matches: {sum_matches}')

if exact_mismatches:
    print(f'\nFirst 5 mismatches: {exact_mismatches[:5]}')
    for h in exact_mismatches[:3]:
        diff = np.abs(new_flow[h] - raw_hourly[h])
        print(f'  Hour {h}: max diff = {diff.max():.6f}, sum diff = {(new_flow[h]-raw_hourly[h]).sum():.6f}')

# ============================================================
# STEP E: Verify partial hours
# ============================================================
print('\n' + '=' * 80)
print('CHECK 2: Partial hours - new should equal the single raw slot')
print('=' * 80)

partial_correct = 0
partial_wrong = []
for h in range(n_hours):
    if raw_count[h] == 1:
        if np.allclose(new_flow[h], raw_hourly[h], atol=1e-6):
            partial_correct += 1
        else:
            partial_wrong.append(h)

print(f'Partial hours: {n_partial}')
print(f'  Match the single slot: {partial_correct}')
print(f'  Wrong:                  {len(partial_wrong)}')

# ============================================================
# STEP F: Verify missing hours (claimed to be filled from DoW)
# ============================================================
print('\n' + '=' * 80)
print('CHECK 3: Missing hours - verify DoW fill')
print('=' * 80)

# For each missing hour, check if new_flow[h] matches any same-DoW hour
dow_check_results = {'same_dow_match': 0, 'no_match': 0, 'wrong_match': []}

for h in range(n_hours):
    if raw_count[h] == 0:
        ts = P4_START + timedelta(hours=int(h))
        target_dow = ts.weekday()
        target_hour_of_day = h % 24

        found = False
        for offset_days in [7, -7, 14, -14, 21, -21, 28, -28, 35, -35, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1]:
            t = h + 24 * offset_days
            if 0 <= t < n_hours:
                tt = P4_START + timedelta(hours=int(t))
                if tt.weekday() == target_dow and raw_count[t] == 2:
                    # Found candidate with same DoW and raw data
                    if np.array_equal(new_flow[h], new_flow[t]):
                        dow_check_results['same_dow_match'] += 1
                        found = True
                        break
        if not found:
            dow_check_results['no_match'] += 1
            dow_check_results['wrong_match'].append(h)

print(f'Missing hours: {n_missing}')
print(f'  Matches same-DoW (verified): {dow_check_results["same_dow_match"]}')
print(f'  No match found:               {dow_check_results["no_match"]}')
if dow_check_results['wrong_match']:
    print(f'  Examples: {dow_check_results["wrong_match"][:5]}')

# ============================================================
# STEP G: Cross-check the exact fill source for first 20 missing hours
# ============================================================
print('\n' + '=' * 80)
print('CHECK 4: Trace each missing hour to its specific source')
print('=' * 80)

traces = []
for h in range(n_hours):
    if raw_count[h] == 0:
        ts = P4_START + timedelta(hours=int(h))
        target_dow = ts.weekday()

        # Find first matching same-DoW hour
        for offset_days in [7, -7, 14, -14, 21, -21, 28, -28, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1]:
            t = h + 24 * offset_days
            if 0 <= t < n_hours:
                tt = P4_START + timedelta(hours=int(t))
                if tt.weekday() == target_dow and raw_count[t] == 2:
                    if np.array_equal(new_flow[h], new_flow[t]):
                        traces.append((h, ts.strftime('%Y-%m-%d %H:%M'), t, tt.strftime('%Y-%m-%d %H:%M')))
                        break

print(f'First 20 trace records (missing hour <- source hour):')
print('  Hour_idx | Missing Date           | Src_idx | Source Date          | Match?')
for h, dt, src, sdt in traces[:20]:
    ts_match = (P4_START + timedelta(hours=int(h))).weekday() == (P4_START + timedelta(hours=int(src))).weekday()
    print(f'  {h:8d} | {dt}  | {src:7d} | {sdt}  | {"YES" if ts_match else "NO"}')

# ============================================================
# STEP H: Spatial pattern verification
# ============================================================
print('\n' + '=' * 80)
print('CHECK 5: Spatial pattern - do flows concentrate in city center?')
print('=' * 80)

# Sum all inflow over all hours and check spatial distribution
total_inflow = new_flow[:, 0, :, :].sum(axis=0)
total_outflow = new_flow[:, 1, :, :].sum(axis=0)

print('Inflow hotspots (top 5 cells):')
flat_idx = np.argsort(total_inflow.flatten())[::-1][:5]
for idx in flat_idx:
    r, c = idx // 32, idx % 32
    print(f'  Cell (row={r:2d}, col={c:2d}): inflow={total_inflow[r,c]:.0f}')

print('Outflow hotspots (top 5 cells):')
flat_idx = np.argsort(total_outflow.flatten())[::-1][:5]
for idx in flat_idx:
    r, c = idx // 32, idx % 32
    print(f'  Cell (row={r:2d}, col={c:2d}): outflow={total_outflow[r,c]:.0f}')

# Check if hotspots are in expected city center area (rows 12-20, cols 15-22)
center_sum_in = total_inflow[12:21, 15:23].sum()
center_sum_total = total_inflow.sum()
print(f'\nInflow in center (rows 12-20, cols 15-23): {center_sum_in:.0f}')
print(f'Inflow total: {center_sum_total:.0f}')
print(f'Center concentration: {center_sum_in/center_sum_total:.1%} (city center should be highest)')

# ============================================================
# STEP I: POI spatial alignment verification
# ============================================================
print('\n' + '=' * 80)
print('CHECK 6: POI - confirm grid bounds are correct via anchor positions')
print('=' * 80)

poi = np.load('/home/ubuntu/data/poi_grid_2015_11.npy')
print(f'POI grid shape: {poi.shape}, total: {int(poi.sum())}')

# Test anchors
LAT_MIN, LAT_MAX = 39.75, 40.09
LON_MIN, LON_MAX = 116.10, 116.60
GRID_SIZE = 32

def project(lat, lon):
    row = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * GRID_SIZE
    col = (lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE
    return row, col

anchors = [
    ('Capital Airport', 40.0801, 116.5846, 'PEK area, should be top-right'),
    ('Beijing Station', 39.9028, 116.4286, 'Central'),
    ('Tiananmen',       39.9087, 116.3975, 'Central'),
    ('Summer Palace',   39.9999, 116.2755, 'Northwest'),
    ('Daxing Airport',  39.5098, 116.4107, 'OUTSIDE grid (too south)'),
    ('Beijing West',    39.8949, 116.3221, 'Central-west'),
    ('Beijing North',   39.9408, 116.3527, 'Central-north'),
]

print('\nAnchor verification:')
for name, lat, lon, expected in anchors:
    r, c = project(lat, lon)
    in_grid = (0 <= r < 32) and (0 <= c < 32)
    poi_count = poi[:, int(min(max(r,0),31)), int(min(max(c,0),31))].sum()
    print(f'  {name:20s}: row={r:5.1f}, col={c:5.1f} {"INSIDE" if in_grid else "OUTSIDE"} ({expected})')
    print(f'    POIs in that cell: {poi_count}')

# ============================================================
# STEP J: Final summary
# ============================================================
print('\n' + '=' * 80)
print('P0 SUMMARY')
print('=' * 80)
print(f'Raw H5 contains {len(raw_dates)} 30-min slots for {P4_DAYS} days.')
print(f'Independently aggregated to {n_hours} hours:')
print(f'  Complete (2 slots): {n_complete} ({n_complete/n_hours:.1%})')
print(f'  Partial (1 slot):   {n_partial}')
print(f'  Missing (0 slots):  {n_missing}')
print(f'')
print(f'Verification of new npz against independently-aggregated raw:')
print(f'  Complete hours exact match: {exact_matches}/{n_complete}')
print(f'  Partial hours correct:      {partial_correct}/{n_partial}')
print(f'  Missing hours DoW-sourced:  {dow_check_results["same_dow_match"]}/{n_missing}')

# Save provenance file
provenance = {
    'raw_h5_path': RAW_H5,
    'raw_total_slots': int(len(raw_dates)),
    'raw_expected_slots': P4_DAYS * 48,
    'n_hours': int(n_hours),
    'complete_hours': int(n_complete),
    'partial_hours': int(n_partial),
    'missing_hours': int(n_missing),
    'complete_match_exact': int(exact_matches),
    'partial_correct': int(partial_correct),
    'missing_dow_match': int(dow_check_results["same_dow_match"]),
}
import json
with open('e:/amazon/evidence/provenance.json', 'w') as f:
    json.dump(provenance, f, indent=2)
print(f'\nSaved e:/amazon/evidence/provenance.json')