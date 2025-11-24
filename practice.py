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

threshold1 = -2                # angle threshold for crosses
threshold2 = -3                # unused for now

# === Fetch data ===
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
closePrice_EOD = stock_data['Close']


print(stock_data.index[:5])
type(stock_data.index[0])


#put cross_dates in a txt file or viewing

for date in cross_dates:
    with open("my_file.txt", "a") as f:
        f.write(date+"\n")
















