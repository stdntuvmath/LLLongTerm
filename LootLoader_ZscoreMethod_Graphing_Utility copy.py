import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

# === Parameters ===
ticker_symbol = "amd"
start_date = "2005-01-01"
end_date = dt.datetime.today()
window_size = 20               # rolling window for std
sigma_plot_max = 10            # how many sigma bands to draw
threshold_sigma = -2           # threshold (-2 sigma)

# === Fetch data ===
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
closePrice_EOD = stock_data['Close']

# === Indicator calculations ===
stock_data['ema200'] = closePrice_EOD.ewm(span=200, adjust=False).mean()
stock_data['ema_slope'] = stock_data['ema200'].diff()
stock_data['angle_ema200'] = np.degrees(np.arctan(stock_data['ema_slope']))

# rolling std (NaNs for first window_size-1 rows)
rolling_std = stock_data['angle_ema200'].rolling(window=window_size).std()

# keep rolling_std in dataframe for plotting
stock_data['rolling_std'] = rolling_std

# compute z-score
stock_data['z'] = stock_data['angle_ema200'] / stock_data['rolling_std']


# create sigma bands (±1..±sigma_plot_max)
for i in range(1, sigma_plot_max + 1):
    stock_data[f'std_pos_{i}'] = stock_data['rolling_std'] * i
    stock_data[f'std_neg_{i}'] = -stock_data['rolling_std'] * i

# === Detect crossings ===
crosses_mask = (stock_data['z'].shift(1) < threshold_sigma) & (stock_data['z'] > threshold_sigma)
cross_dates = stock_data.index[crosses_mask]

# === Plotting ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.set_facecolor('black')  # background color

# Top: price + EMA
ax1.plot(closePrice_EOD, label='Close', linewidth=1)
ax1.plot(stock_data['ema200'], label='EMA200', linewidth=1)
ax1.set_ylabel('Price (USD)')
ax1.set_title(f"{ticker_symbol} Price & EMA (to {end_date.date()})")
ax1.grid(True)

# === Bottom: angle and sigma bands ===
ax2.plot(stock_data['angle_ema200'], label='angle_ema200', linewidth=1)


# Custom sigma colors for dark background
sigma_colors = [
    'rebeccapurple', 'mediumvioletred', 'crimson', 'red', 'orangered',
    'darkorange', 'goldenrod', 'gold', 'yellowgreen', 'lawngreen'
]

# Plot sigma bands (custom colors)
for i in range(1, sigma_plot_max + 1):
    color = sigma_colors[(i-1) % len(sigma_colors)]
    
    ax2.plot(
        stock_data[f'std_pos_{i}'],
        linestyle='--', linewidth=0.8, alpha=0.8,
        color=color,
        label=f'+{i}σ' if i == 1 else None
    )
    
    ax2.plot(
        stock_data[f'std_neg_{i}'],
        linestyle='--', linewidth=0.8, alpha=0.8,
        color=color,
        label=f'-{i}σ' if i == 1 else None
    )

# === Zero-line origin ===
ax2.axhline(0, color='blue', linewidth=1.2, alpha=0.9, label='origin')

# Plot threshold sigma line
ax2.plot(
    stock_data['rolling_std'] * threshold_sigma,
    linestyle='--', linewidth=1, alpha=0.9,
    label=f'{threshold_sigma}σ (threshold)'
)

# Vertical lines + scatter markers for cross points
if not cross_dates.empty:
    first = True
    for date in cross_dates:
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
#ax2.legend(loc='upper left', fontsize='small')

# Make both subplot backgrounds black
ax1.set_facecolor('black')
ax2.set_facecolor('black')


plt.tight_layout()
plt.show()

# === Optional: print summary ===
if not cross_dates.empty:
    print(f"Detected {len(cross_dates)} crossings above {threshold_sigma}σ at:")
    for d in cross_dates:
        z_val = stock_data.loc[d, 'z']
        angle_val = stock_data.loc[d, 'angle_ema200']
        price_val = closePrice_EOD.loc[d]
        print(f"  {d.date()}: z={z_val:.2f}, angle={angle_val:.3f}, price={price_val:.2f}")
else:
    print(f"No crossings above {threshold_sigma}σ detected.")
