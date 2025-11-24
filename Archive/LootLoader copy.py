import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
import statistics
import numpy as np




# 1. Define the stock ticker and date range
ticker_symbol = "AAPL"
start_date = "2000-01-01"
end_date = dt.datetime.today()

# 2. Fetch the historical data
# The yf.download function returns a pandas DataFrame
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

closePrice_EOD = stock_data['Close']
# calculations

stock_data['ema200'] = closePrice_EOD.ewm(span=200,adjust=False).mean()

stock_data['angle_ema200'] = np.arctan(stock_data['ema200'])*180/math.pi

window_size = 20

stock_data['stdDev1'] = stock_data['angle_ema200'].rolling(window=window_size).std()
stock_data['stdDev2'] = stock_data['angle_ema200'].rolling(window=window_size).std()*2
stock_data['stdDev3'] = stock_data['angle_ema200'].rolling(window=window_size).std()*3
stock_data['stdDev4'] = stock_data['angle_ema200'].rolling(window=window_size).std()*4
stock_data['stdDev5'] = stock_data['angle_ema200'].rolling(window=window_size).std()*5
stock_data['stdDev6'] = stock_data['angle_ema200'].rolling(window=window_size).std()*6
origin = 0


plt.figure(figsize=(10, 6)) # Set the figure size
plt.plot(closePrice_EOD, label='Closing Price', color='darkgreen')
plt.plot(stock_data['ema200'], label='EMA200', color='red')

# 4. Customize the plot
plt.title(f"{ticker_symbol} Stock Closing Prices ({start_date} to {end_date})")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

# 5. Display the plot
plt.show()
