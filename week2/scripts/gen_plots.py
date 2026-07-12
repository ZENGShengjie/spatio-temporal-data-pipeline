"""
Generate visualizations as evidence:
  1. Hour coverage heatmap (calendar)
  2. Daily time series of inflow vs outflow
  3. Spatial heatmaps of total flow
  4. POI distribution per category
  5. Anchor position verification on the grid
  6. Comparison: Step1 linear-interpolated vs our DoW-filled (sample days)
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

os.makedirs('e:/amazon/evidence/plots', exist_ok=True)

P4_START = datetime(2015, 11, 1, 0, 0)
P4_DAYS = 162
n_hours = P4_DAYS * 24

# Load data
flow = np.load('/home/ubuntu/data/cleaned_bj/taxi_p4_4d.npz')['flow']
hourly_count = np.load('e:/amazon/evidence/hourly_count.npy')
with h5py.File('/home/ubuntu/data/raw_bj/taxi_bj_gitee/BJ16_M32x32_T30_InOut.h5', 'r') as f:
    raw_dates = f['date'][:]

# Aggregate raw
raw_per_hour = {}
for i, d_bytes in enumerate(raw_dates):
    ds = d_bytes.decode()
    date_str = ds[:8]
    slot_1 = int(ds[8:])
    h_of_day = (slot_1 - 1) // 2
    day_offset = (datetime.strptime(date_str, '%Y%m%d') - P4_START).days
    h_abs = day_offset * 24 + h_of_day
    if 0 <= h_abs < n_hours:
        if h_abs not in raw_per_hour:
            raw_per_hour[h_abs] = 0
        raw_per_hour[h_abs] += 1

hourly_count_correct = np.array([raw_per_hour.get(h, 0) for h in range(n_hours)])
np.save('e:/amazon/evidence/hourly_count.npy', hourly_count_correct)
print(f'Saved correct hourly_count.npy')

# ============================================================
# PLOT 1: Hour coverage calendar
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
coverage_matrix = hourly_count_correct.reshape(P4_DAYS, 24)

cmap = plt.cm.colors.ListedColormap(['lightgray', 'yellow', 'green'])
im = ax.imshow(coverage_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2)

# Mark dates
date_labels = []
for d in range(0, P4_DAYS, 7):
    date_labels.append((P4_START + timedelta(days=d)).strftime('%b-%d'))
ax.set_yticks(range(0, P4_DAYS, 7))
ax.set_yticklabels(date_labels, fontsize=8)
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], fontsize=8)
ax.set_xlabel('Hour of day')
ax.set_ylabel('Date')
ax.set_title('Hour coverage: gray=missing, yellow=partial, green=both slots present')
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.set_label('# 30-min slots present')
plt.tight_layout()
plt.savefig('e:/amazon/evidence/plots/01_coverage_calendar.png', dpi=120)
plt.close()
print('Saved 01_coverage_calendar.png')

# ============================================================
# PLOT 2: Daily inflow vs outflow time series
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Compute daily totals
daily_in = flow[:, 0].sum(axis=(1, 2)).reshape(P4_DAYS, 24).sum(axis=1)
daily_out = flow[:, 1].sum(axis=(1, 2)).reshape(P4_DAYS, 24).sum(axis=1)
dates = [P4_START + timedelta(days=d) for d in range(P4_DAYS)]

# Color days with missing data
daily_missing = (hourly_count_correct.reshape(P4_DAYS, 24) < 2).any(axis=1)

ax1 = axes[0]
ax1.plot(dates, daily_in, 'b-', alpha=0.7)
ax1.scatter([d for d, m in zip(dates, daily_missing) if m],
            [di for di, m in zip(daily_in, daily_missing) if m],
            color='red', s=15, label='Days with missing hours', zorder=5)
ax1.set_ylabel('Daily total inflow')
ax1.set_title('Daily inflow totals (red = days with missing data filled by DoW)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(dates, daily_out, 'g-', alpha=0.7)
ax2.scatter([d for d, m in zip(dates, daily_missing) if m],
            [do for do, m in zip(daily_out, daily_missing) if m],
            color='red', s=15, label='Days with missing hours', zorder=5)
ax2.set_ylabel('Daily total outflow')
ax2.set_xlabel('Date')
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/amazon/evidence/plots/02_daily_in_out.png', dpi=120)
plt.close()
print('Saved 02_daily_in_out.png')

# ============================================================
# PLOT 3: Spatial heatmap of total flow
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

total_in = flow[:, 0].sum(axis=0)
total_out = flow[:, 1].sum(axis=0)

im1 = axes[0].imshow(total_in, cmap='hot', aspect='auto')
axes[0].set_title('Total inflow across all hours\n(over 162 days, log scale)')
axes[0].set_xlabel('Column (West to East)')
axes[0].set_ylabel('Row (North to South)')
import matplotlib.colors as mcolors
norm = mcolors.LogNorm(vmin=max(total_in.min(), 1), vmax=total_in.max())
im1.set_norm(norm)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(total_out, cmap='hot', aspect='auto')
axes[1].set_title('Total outflow (log scale)')
axes[1].set_xlabel('Column (West to East)')
im2.set_norm(norm)
plt.colorbar(im2, ax=axes[1])

# Mark anchor positions
LAT_MIN, LAT_MAX = 39.75, 40.09
LON_MIN, LON_MAX = 116.10, 116.60
anchors = [
    ('PEK', 40.0801, 116.5846, 'cyan'),
    ('BJS', 39.9028, 116.4286, 'lime'),
    ('TJN', 39.9087, 116.3975, 'lime'),
    ('SP', 39.9999, 116.2755, 'cyan'),
]
for name, lat, lon, color in anchors:
    r = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * 32
    c = (lon - LON_MIN) / (LON_MAX - LON_MIN) * 32
    for ax in axes:
        ax.plot(c, r, marker='*', markersize=15, color=color, markeredgecolor='black', markeredgewidth=1)
        ax.annotate(name, (c, r), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='white', weight='bold')

plt.tight_layout()
plt.savefig('e:/amazon/evidence/plots/03_spatial_heatmap.png', dpi=120)
plt.close()
print('Saved 03_spatial_heatmap.png')

# ============================================================
# PLOT 4: POI distribution per category
# ============================================================
poi = np.load('/home/ubuntu/data/poi_grid_2015_11.npy')
cat_names = ['food', 'shopping', 'transport', 'work', 'leisure',
             'residence', 'education', 'health', 'tourism', 'finance']

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i, (ax, cat) in enumerate(zip(axes.flat, cat_names)):
    data = poi[i]
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    ax.set_title(f'{cat} ({int(data.sum())} POIs)', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('POI distribution per category (2015-11, ohsome API)')
plt.tight_layout()
plt.savefig('e:/amazon/evidence/plots/04_poi_per_category.png', dpi=120)
plt.close()
print('Saved 04_poi_per_category.png')

# ============================================================
# PLOT 5: Anchor position verification on the grid
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10))

# Background: total flow
ax.imshow(total_in, cmap='gray_r', aspect='auto', alpha=0.5)

# All anchors with labels
anchors = [
    ('Capital Airport', 40.0801, 116.5846, 'red', '*', 25),
    ('Beijing Station', 39.9028, 116.4286, 'blue', 'o', 15),
    ('Beijing North', 39.9408, 116.3527, 'blue', 'o', 15),
    ('Beijing West', 39.8949, 116.3221, 'blue', 'o', 15),
    ('Tiananmen', 39.9087, 116.3975, 'gold', '*', 20),
    ('Summer Palace', 39.9999, 116.2755, 'green', '^', 15),
    ('Daxing Airport (outside)', 39.5098, 116.4107, 'purple', 'x', 15),
]

for name, lat, lon, color, marker, size in anchors:
    r = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * 32
    c = (lon - LON_MIN) / (LON_MAX - LON_MIN) * 32
    ax.plot(c, r, marker=marker, markersize=size, color=color, markeredgecolor='black', markeredgewidth=1)
    ax.annotate(name, (c, r), xytext=(8, 8), textcoords='offset points',
               fontsize=9, color=color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Grid bounds
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Grid N boundary')
ax.axhline(y=32, color='red', linestyle='--', alpha=0.5, label='Grid S boundary')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=32, color='red', linestyle='--', alpha=0.5)

ax.set_xlim(-2, 34)
ax.set_ylim(34, -2)
ax.set_xlabel('Column (West → East)')
ax.set_ylabel('Row (North → South)')
ax.set_title('Anchor positions on Beijing 32x32 grid\n(bg = total flow darkness)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('e:/amazon/evidence/plots/05_anchor_positions.png', dpi=120)
plt.close()
print('Saved 05_anchor_positions.png')

# ============================================================
# PLOT 6: Linear interpolation vs DoW fill comparison
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Hourly sums
hourly_sums = flow.sum(axis=(1, 2, 3))
hourly_ts = [P4_START + timedelta(hours=int(h)) for h in range(n_hours)]

# Compute "what Step 1 would have done" (linear interp of valid hours)
import pandas as pd
valid_mask = hourly_count_correct == 2
sums_with_nan = np.where(valid_mask, hourly_sums, np.nan)
df = pd.Series(sums_with_nan)
step1_linear = df.interpolate(method='linear').values

# Hour-of-day pattern
hod_pattern_in = np.zeros(24)
hod_count_in = np.zeros(24)
for h in range(n_hours):
    if hourly_count_correct[h] == 2:
        hod = h % 24
        hod_pattern_in[hod] += flow[h].sum()
        hod_count_in[hod] += 1
hod_pattern_in = hod_pattern_in / np.maximum(hod_count_in, 1)

ax1 = axes[0]
ax1.plot(hourly_ts, step1_linear, 'b-', alpha=0.5, label='Step1 (linear interp)')
ax1.plot(hourly_ts, hourly_sums, 'r-', alpha=0.7, label='Our (DoW fill)')
ax1.set_xlim([P4_START, P4_START + timedelta(days=10)])
ax1.set_ylabel('Hourly sum')
ax1.set_title('First 10 days: Step 1 (linear) vs Our (DoW fill)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Show hour-of-day pattern averaged over all weeks
hours = list(range(24))
ax2 = axes[1]
ax2.bar(hours, hod_pattern_in, color='steelblue', alpha=0.7, label='Average by hour-of-day')
ax2.set_xlabel('Hour of day')
ax2.set_ylabel('Average total flow')
ax2.set_title('Average hour-of-day pattern (only complete hours averaged)')
ax2.set_xticks(range(0, 24, 2))
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('e:/amazon/evidence/plots/06_linear_vs_dow.png', dpi=120)
plt.close()
print('Saved 06_linear_vs_dow.png')

print('\nAll visualizations saved to e:/amazon/evidence/plots/')