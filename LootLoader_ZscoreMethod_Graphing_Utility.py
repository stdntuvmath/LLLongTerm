import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates


# === Parameters ===
ticker_symbol = "VV"

#get ticker symbol name from yf
ticker_info = yf.Ticker(ticker_symbol)
ticker_name = ticker_info.info.get('shortName', ticker_symbol)

# Subtract 20 years using relativedelta
todaysDate = dt.datetime.today()
date_minus_20_years = todaysDate - relativedelta(years=25)
start_date = date_minus_20_years
end_date = dt.datetime.today()
window_size = 20               # rolling window for std
sigma_plot_max = 10            # how many sigma bands to draw


# ==== METHODS ====
def append_hlev_to_format(orig_format):
    # returns a function that calls the original formatter and appends HLEV
    def _fmt(x, y):
        # call original to get exact original x,y text
        base = orig_format(x, y)

        # try to find nearest HLEV value for this x (works with date axes)
        try:
            x_dt = mdates.num2date(x).replace(tzinfo=None)
            ix = stock_data.index.get_indexer([x_dt], method='nearest')[0]
            date = stock_data.index[ix]
            hlev = stock_data['HLEV_percentage'].iloc[ix]

            # show '---' if NaN so the UI isn't confusing
            hlev_text = f"{hlev:.4f}" if (pd.notna(hlev) and np.isfinite(hlev)) else "---"
            return f"{base}   HLEV={hlev_text}"
        except Exception:
            # if anything goes wrong, just return the original string
            return base

    return _fmt

# --- DEBUG FIX: ensure every cell is a scalar ---
def flatten_if_series(val):
    # if a Series sneaks into a cell, convert to float
    if isinstance(val, pd.Series):
        return float(val.iloc[0]) if len(val) > 0 else np.nan
    return val




# ===== SIGMA LINES =====


buy_threshold_neg5 = -5            # threshold1 (-2 sigma)
buy_threshold_neg8 = -8            # threshold2 (-3 sigma)
buy_threshold_neg9 = -9
buy_threshold_neg10 = -10

sell_threshold5 = 5
sell_threshold8 = 8
sell_threshold9 = 9
sell_threshold10 = 10

# === Fetch data ===
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
closePrice_EOD = stock_data['Close'].iloc[:, 0] if isinstance(stock_data['Close'], pd.DataFrame) else stock_data['Close']

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

# find the highest and lowest close prices in the for every 1 year intervals in stock_data

stock_data['Highest_close_1yr'] = closePrice_EOD.rolling(window=504, min_periods=1).max()
stock_data['Lowest_close_1yr'] = closePrice_EOD.rolling(window=504, min_periods=1).min()


stock_data['HLEV_Origin'] = stock_data['Lowest_close_1yr'] + (stock_data['Highest_close_1yr'] - stock_data['Lowest_close_1yr'])/2



stock_data['HLEV_percentage'] = (closePrice_EOD - stock_data['Lowest_close_1yr']) / (stock_data['HLEV_Origin'] - stock_data['Lowest_close_1yr'])


# === FIX: prevent divide-by-zero issues in HLEV calculation ===
den = (stock_data['HLEV_Origin'] - stock_data['Lowest_close_1yr'])

# Replace any 0 denominators with NaN so they don't kill the entire series
den_safe = den.replace(0, np.nan)

# Recompute HLEV percentage using safe denominator (keeps your original formula intact)
stock_data['HLEV_percentage'] = (closePrice_EOD - stock_data['HLEV_Origin']) / den_safe

# === Smoothing EMA's for the classifier ===
stock_data['ema30'] = closePrice_EOD.ewm(span=30, adjust=False).mean()
stock_data['ema90'] = closePrice_EOD.ewm(span=90, adjust=False).mean()


stock_data['angle_ema200'] = stock_data['angle_ema200'].apply(flatten_if_series)
stock_data['HLEV_percentage'] = stock_data['HLEV_percentage'].apply(flatten_if_series)

# -----------------------
# HLEV-driven stored-low -> angle bounce buy logic
# -----------------------

# parameters
ANGLE_BOUNCE_DEG = 10
HLEV_TRIGGER = -1.0

# prepare columns
stock_data['stored_low_angle'] = np.nan
stock_data['angle_hlev_buy'] = False

# internal state
stored = np.nan

# iterate rows
for idx, row in stock_data.iterrows():

    # FORCE SCALAR VALUES (fixes your error!)
    angle = float(row['angle_ema200']) if pd.notna(row['angle_ema200']) else np.nan
    hlev = float(row['HLEV_percentage']) if pd.notna(row['HLEV_percentage']) else np.nan

    # start/reset stored low
    if pd.notna(hlev) and hlev <= HLEV_TRIGGER and pd.notna(angle):
        stored = angle

    # update stored low & detect bounce
    if pd.notna(stored) and pd.notna(angle):

        if angle < stored:
            stored = angle

        if angle >= stored + ANGLE_BOUNCE_DEG:
            stock_data.at[idx, 'angle_hlev_buy'] = True
            stored = np.nan  # reset after buy

    stock_data.at[idx, 'stored_low_angle'] = stored

# extract buy dates
cross_dates_buy = stock_data.index[stock_data['angle_hlev_buy']]


















# =======================================================
# === FIXED CROSSING DETECTION (BUYS) =======
# =======================================================

z = stock_data['z']

# cross_buy = (

#     (angles >= stock_data['last_bottom_angle'] + 10)

#     # ((stock_data['ema30'].shift(1) > stock_data['ema90'].shift(1)) &
#     # (stock_data['ema30'] < stock_data['ema90']) &
#     # (stock_data['ema90'] > stock_data['HLEV_Origin']))
    
#     # ((stock_data['ema30'].shift(1) < stock_data['ema90'].shift(1)) &
#     # (stock_data['ema30'] > stock_data['ema90'])) &
#     # (stock_data['ema90'] < stock_data['HLEV_Origin'])&
#     # (closePrice_EOD < stock_data['HLEV_Origin'])&
#     # ((stock_data['HLEV_percentage'] <= -0.4))
#     # |
    
#     # ((stock_data['angle_ema200'].shift(1) < -10*stock_data['rolling_std'].shift(1)) &
#     # (stock_data['angle_ema200'] > -10*stock_data['rolling_std']))&
#     # ((stock_data['10sigma_avg'] < 10*stock_data['rolling_std']))&
#     # (stock_data['HLEV_percentage'] <= -1)
#     # |    
#     # ((stock_data['10sigma_avg'] < 10*stock_data['rolling_std'])&
#     #  (stock_data['HLEV_percentage'] <= -1))


# )
# cross_dates_buy = stock_data.index[cross_buy]

cross_sell = (

    ((stock_data['ema30'].shift(1) > stock_data['ema90'].shift(1)) &
    (stock_data['ema30'] < stock_data['ema90']) &
    (stock_data['ema90'] > stock_data['HLEV_Origin']))
    
    # ((stock_data['ema30'].shift(1) > stock_data['ema90'].shift(1)) &
    # (stock_data['ema30'] < stock_data['ema90']) &
    # (stock_data['HLEV_percentage'] > 0.5))
    # |
    # ((stock_data['HLEV_percentage'] >= .9)&
    #  (stock_data['angle_ema200'].shift(1) > 8*stock_data['rolling_std'].shift(1)) &
    #  (stock_data['angle_ema200'] < 8*stock_data['rolling_std']))
    # |
    # ((stock_data['HLEV_percentage'] >= .9)&
    #  (stock_data['angle_ema200'].shift(1) > 9*stock_data['rolling_std'].shift(1)) &
    #  (stock_data['angle_ema200'] < 9*stock_data['rolling_std']))
)
cross_dates_sell = stock_data.index[cross_sell]





# === Plotting ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.set_facecolor('Gray')

# Top: price + EMA and HLEV origin
ax1.plot(stock_data['HLEV_Origin'], linewidth=1, color='yellow')
ax1.plot(closePrice_EOD, label='Daily Close Price', linewidth=1)
ax1.plot(stock_data['ema200'], label='EMA200', linewidth=1)
ax1.plot(stock_data['ema30'], label='EMA30', linewidth=1, color='white')
ax1.plot(stock_data['ema90'], label='EMA90', linewidth=1, color='red')
ax1.set_ylabel('Price (USD)')
ax1.set_title(f"{ticker_name} ({ticker_symbol}) Price & EMA 25 years ago to present day")
ax1.grid(True)

# Bottom: angle and sigma bands
ax2.set_title(f"Turner Bands Indicator and HLEV Signals (from {date_minus_20_years.date()} to {end_date.date()})")
ax2.plot(stock_data['angle_ema200'], label='angle_ema200', linewidth=1)
ax2.plot(stock_data['10sigma_avg'], color='gray',linewidth=1)
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

# # Threshold lines
# ax2.plot(
#     stock_data['rolling_std'] * buy_threshold1,
#     linestyle='--', linewidth=1, alpha=0.9,
#     label=f'{buy_threshold1}σ (threshold)'
# )
# ax2.plot(
#     stock_data['rolling_std'] * buy_threshold2,
#     linestyle='--', linewidth=1, alpha=0.9,
#     label=f'{buy_threshold2}σ (threshold)'
# )
print(stock_data['local_bottom'].sum())

# === Plot signals for buy_threshold1
if not cross_dates_buy.empty:
    first = True
    for date in cross_dates_buy:
        ax2.axvline(x=date, color='cyan', linestyle='--', alpha=0.8, linewidth=1.0,
                    label='Buy Signal' if first else None)
        ax1.axvline(x=date, color='cyan', linestyle='--', alpha=0.35, linewidth=1.0)
        
        ax2.scatter(date, stock_data.loc[date, 'angle_ema200'], color='cyan', zorder=6)
        ax1.scatter(date, closePrice_EOD.loc[date], color='cyan', zorder=6)
        first = False


# === Plot signals for buy_threshold1
if not cross_dates_sell.empty:
    first = True
    for date in cross_dates_sell:
        ax2.axvline(x=date, color='Magenta', linestyle='--', alpha=0.8, linewidth=1.0,
                    label='Sell Signal' if first else None)
        ax1.axvline(x=date, color='Magenta', linestyle='--', alpha=0.35, linewidth=1.0)
        
        ax2.scatter(date, stock_data.loc[date, 'angle_ema200'], color='Magenta', zorder=6)
        ax1.scatter(date, closePrice_EOD.loc[date], color='Magenta', zorder=6)
        first = False


# === Preserve the original x,y display and append HLEV_percentage (non-invasive) ===
# Save original formatters
orig_ax1_format = ax1.format_coord
orig_ax2_format = ax2.format_coord

def append_hlev_to_format(orig_format):
    # returns a function that calls the original formatter and appends HLEV
    def _fmt(x, y):
        # call original to get exact original x,y text
        base = orig_format(x, y)

        # try to find nearest HLEV value for this x (works with date axes)
        try:
            x_dt = mdates.num2date(x).replace(tzinfo=None)
            ix = stock_data.index.get_indexer([x_dt], method='nearest')[0]
            date = stock_data.index[ix]
            hlev = stock_data['HLEV_percentage'].iloc[ix]

            # show '---' if NaN so the UI isn't confusing
            hlev_text = f"{hlev:.4f}" if (pd.notna(hlev) and np.isfinite(hlev)) else "---"
            return f"{base}   HLEV={hlev_text}"
        except Exception:
            # if anything goes wrong, just return the original string
            return base

    return _fmt

# Apply only to the bottom axis as you requested
ax1.format_coord = orig_ax1_format     # leave top axis exactly as-is
ax2.format_coord = append_hlev_to_format(orig_ax2_format)


# === Labels, grid, legend ===
ax1.legend(loc='upper left', fontsize='small')
ax2.set_ylabel('Angle (degrees)')
ax2.set_xlabel('Date')
ax2.grid(True)

ax1.set_facecolor('black')
ax2.set_facecolor('black')

# === Make x-axis grid show every year ===
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))

# year_locator = mdates.YearLocator()           # one tick/grid per year
# year_fmt = mdates.DateFormatter('%Y')         # show just the year

# ax2.xaxis.set_major_locator(year_locator)
# ax2.xaxis.set_major_formatter(year_fmt)

# ax1.xaxis.set_major_locator(year_locator)
# ax1.xaxis.set_major_formatter(year_fmt)

# Ensure gridlines are visible
ax1.grid(True, which='major')
ax2.grid(True, which='major')

#plt.tight_layout()
plt.show()
