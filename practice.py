import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import json


# === Parameters ===
ticker_symbol = "ACMR"

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


with open("Indicators.cfg", "r") as f:
    logic_config = json.load(f)


column_map = {
    "close": "Close",
    "ema200": "ema200",
    "10sigma_avg": "10sigma_avg",
    "hlev_origin": "HLEV_Origin"
}




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







def Parse_Indicator(indicatorList1, crossingList, indicatorList2):
    tasks = []

    # Combine the lists into one set to parse only desired indicators
    indicator_set = set(indicatorList1 + indicatorList2)

    for ind_name, rule in logic_config.items():
        if ind_name not in indicator_set:
            continue  # skip indicators not in the input lists

        tasks.append({
            "col": column_map.get(ind_name, ind_name),
            "cross_above": rule.get("crosses_above"),
            "cross_below": rule.get("crosses_below"),
            "window": rule.get("window", None)
        })

    return tasks



       
def Evaluate_Crossings(stock_data, tasks):
    cross_buy = []
    cross_sell = []

    for task in tasks:
        col = task["col"]
        ca = task["cross_above"]
        cb = task["cross_below"]

        for i in range(1, len(stock_data)):
            prev = stock_data[col].iloc[i-1]
            curr = stock_data[col].iloc[i]

            params = {
                "indicator": curr,
                "prev_indicator": prev,
                "target": 0
            }

            if ca and eval(ca, {"__builtins__": {}}, params):
                cross_buy.append(stock_data.index[i])

            if cb and eval(cb, {"__builtins__": {}}, params):
                cross_sell.append(stock_data.index[i])

    return cross_buy, cross_sell



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.set_facecolor('Gray')

tasks = Parse_Indicator()
buy_dates, sell_dates = Evaluate_Crossings(stock_data, tasks)


for date in buy_dates:
    ax2.axvline(x=date, color='cyan', linestyle='--', alpha=0.8)

for date in sell_dates:
    ax2.axvline(x=date, color='magenta', linestyle='--', alpha=0.8)











