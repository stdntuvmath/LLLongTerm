import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 1. Data
ticker_symbol = "AAPL"
start_date = "2005-01-01"
end_date = dt.datetime.today()

stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
closePrice_EOD = stock_data['Close']

# 2. Calculations
stock_data['ema200'] = closePrice_EOD.ewm(span=200, adjust=False).mean()
stock_data['ema_slope'] = stock_data['ema200'].diff()
stock_data['angle_ema200'] = np.degrees(np.arctan(stock_data['ema_slope']))

window_size = 20
rolling_std = stock_data['angle_ema200'].rolling(window=window_size).std()

for i in range(1,7):
    stock_data[f'stdDev_pos{i}'] = stock_data['angle_ema200'] + rolling_std*i
    stock_data[f'stdDev_neg{i}'] = stock_data['angle_ema200'] - rolling_std*i

stock_data = stock_data.dropna()

# 3. Make stacked subplots
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(f"{ticker_symbol} Closing Price + EMA200", "Angle EMA200 ± StdDev")
)

# Top plot: price
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close', line=dict(color='darkgreen')), row=1, col=1)
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['ema200'], mode='lines', name='EMA200', line=dict(color='red')), row=1, col=1)

# Bottom plot: angle ± std
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['angle_ema200'], mode='lines', name='Angle EMA200', line=dict(color='blue')), row=2, col=1)

for i in range(1,7):
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[f'stdDev_pos{i}'], mode='lines', name=f'+{i}σ', line=dict(dash='dash', color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[f'stdDev_neg{i}'], mode='lines', name=f'-{i}σ', line=dict(dash='dash', color='orange')), row=2, col=1)

# 4. Layout
fig.update_layout(
    height=800,
    title=f"{ticker_symbol} Stock Price + EMA200 + Angle ± StdDev",
    showlegend=True
)

# Add range slider to bottom x-axis
fig.update_xaxes(rangeslider=dict(visible=True), row=2, col=1)

# Zoom to last 2 years
two_years = pd.Timedelta(days=365*2)
fig.update_xaxes(range=[stock_data.index[-1]-two_years, stock_data.index[-1]])

fig.show()
