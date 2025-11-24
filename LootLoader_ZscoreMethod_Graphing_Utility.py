import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates


# === Parameters ===
ticker_symbol = "amd"
# Subtract 20 years using relativedelta
todaysDate = dt.datetime.today()
date_minus_20_years = todaysDate - relativedelta(years=20)
start_date = date_minus_20_years
end_date = dt.datetime.today()
window_size = 20               # rolling window for std
sigma_plot_max = 10            # how many sigma bands to draw
buy_threshold1 = -2            # threshold1 (-2 sigma)
buy_threshold2 = -3            # threshold2 (-3 sigma)

# === Fetch data ===
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
closePrice_EOD = stock_data['Close']

# === Indicator calculations ===
stock_data['ema200'] = closePrice_EOD.ewm(span=200, adjust=False).mean()
stock_data['ema_slope'] = stock_data['ema200'].diff()
stock_data['angle_ema200'] = np.degrees(np.arctan(stock_data['ema_slope']))

# rolling std (NaNs for first window_size-1 rows)
rolling_std = stock_data['angle_ema200'].rolling(window=window_size).std()
stock_data['rolling_std'] = rolling_std

# compute z-score
stock_data['z'] = stock_data['angle_ema200'] / stock_data['rolling_std']

# === Create sigma bands ===
for i in range(1, sigma_plot_max + 1):
    stock_data[f'std_pos_{i}'] = stock_data['rolling_std'] * i
    stock_data[f'std_neg_{i}'] = -stock_data['rolling_std'] * i

# === Compute 10sigma average ===
stock_data['10sigma_avg'] = (stock_data['rolling_std'] * 10).rolling(window=window_size).mean()

# === Compute HLEV percentage for 1 year intervals ===


# =======================================================
# === FIXED CROSSING DETECTION (no false signals) =======
# =======================================================

z = stock_data['z']

crosses_mask1 = (
    #z.shift(1).notna() &
     #z.notna() &
    (z.shift(1) < buy_threshold1) &
    (z >= buy_threshold1)
)
cross_dates1 = stock_data.index[crosses_mask1]

crosses_mask2 = (
    # z.shift(1).notna() &
    # z.notna() &
    (z.shift(1) < buy_threshold2) &
    (z >= buy_threshold2)
)
cross_dates2 = stock_data.index[crosses_mask2]

# === 10sigma_avg Rule ===

# === Plotting ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.set_facecolor('black')

# Top: price + EMA
ax1.plot(closePrice_EOD, label='Close', linewidth=1)
ax1.plot(stock_data['ema200'], label='EMA200', linewidth=1)
ax1.set_ylabel('Price (USD)')
ax1.set_title(f"{ticker_symbol} Price & EMA (to {end_date.date()})")
ax1.grid(True)

# Bottom: angle and sigma bands
ax2.plot(stock_data['angle_ema200'], label='angle_ema200', linewidth=1)

# Custom sigma colors
sigma_colors = [
    'rebeccapurple', 'mediumvioletred', 'crimson', 'red', 'orangered',
    'darkorange', 'goldenrod', 'gold', 'yellowgreen', 'lawngreen'
]

# Sigma bands
for i in range(1, sigma_plot_max + 1):
    color = sigma_colors[(i-1) % len(sigma_colors)]
    
    ax2.plot(
        stock_data[f'std_pos_{i}'], linestyle='--', linewidth=0.8,
        alpha=0.8, color=color, label=f'+{i}σ' if i == 1 else None
    )
    ax2.plot(
        stock_data[f'std_neg_{i}'], linestyle='--', linewidth=0.8,
        alpha=0.8, color=color, label=f'-{i}σ' if i == 1 else None
    )

# Zero-line origin
ax2.axhline(0, color='blue', linewidth=1.2, alpha=0.9, label='origin')

# Threshold lines
ax2.plot(
    stock_data['rolling_std'] * buy_threshold1,
    linestyle='--', linewidth=1, alpha=0.9,
    label=f'{buy_threshold1}σ (threshold)'
)
ax2.plot(
    stock_data['rolling_std'] * buy_threshold2,
    linestyle='--', linewidth=1, alpha=0.9,
    label=f'{buy_threshold2}σ (threshold)'
)

# === Plot signals for threshold1
if not cross_dates1.empty:
    first = True
    for date in cross_dates1:
        ax2.axvline(x=date, color='cyan', linestyle='--', alpha=0.8, linewidth=1.0,
                    label='Buy Signal' if first else None)
        ax1.axvline(x=date, color='cyan', linestyle='--', alpha=0.35, linewidth=1.0)
        
        ax2.scatter(date, stock_data.loc[date, 'angle_ema200'], color='cyan', zorder=6)
        ax1.scatter(date, closePrice_EOD.loc[date], color='cyan', zorder=6)
        first = False

# === Plot signals for threshold2
if not cross_dates2.empty:
    first = True
    for date in cross_dates2:
        ax2.axvline(x=date, color='cyan', linestyle='--', alpha=0.8, linewidth=1.0,
                    label='Buy Signal' if first else None)
        ax1.axvline(x=date, color='cyan', linestyle='--', alpha=0.35, linewidth=1.0)
        
        ax2.scatter(date, stock_data.loc[date, 'angle_ema200'], color='cyan', zorder=6)
        ax1.scatter(date, closePrice_EOD.loc[date], color='cyan', zorder=6)
        first = False

# === Labels, grid, legend ===
ax1.legend(loc='upper left', fontsize='small')
ax2.set_ylabel('Angle (degrees)')
ax2.set_xlabel('Date')
ax2.grid(True)

ax1.set_facecolor('black')
ax2.set_facecolor('black')

# === Make x-axis grid show every year ===
year_locator = mdates.YearLocator()           # one tick/grid per year
year_fmt = mdates.DateFormatter('%Y')         # show just the year

ax2.xaxis.set_major_locator(year_locator)
ax2.xaxis.set_major_formatter(year_fmt)

ax1.xaxis.set_major_locator(year_locator)
ax1.xaxis.set_major_formatter(year_fmt)

# Ensure gridlines are visible
ax1.grid(True, which='major')
ax2.grid(True, which='major')

plt.tight_layout()
plt.show()
