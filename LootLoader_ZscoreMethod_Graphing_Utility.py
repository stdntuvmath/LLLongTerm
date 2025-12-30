import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import LLLib_Charles_Schwab as lib


# ======================================================
# === PARAMETERS =======================================
# ======================================================
ticker_symbol = "CAT"
ACCOUNT_START = 25000

# CSP parameters
CSP_ENABLED = True
CSP_DTE = 30                     # trading days
CSP_VOL_LOOKBACK = 252            # 1 year
CSP_VOL_THRESHOLD = 0.60          # ATR% percentile
CSP_MAX_CAP_PCT = 0.30            # max capital per CSP
CSP_STRIKE_ATR_MULT = 1.0         # strike distance


# ======================================================
# === DATA =============================================
# ======================================================
ticker_info = yf.Ticker(ticker_symbol)
ticker_name = ticker_info.info.get('shortName', ticker_symbol)

todaysDate = dt.datetime.today()
start_date = todaysDate - relativedelta(years=25)
end_date = todaysDate

stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

close = stock_data["Close"].squeeze()
high = stock_data["High"].squeeze()
low = stock_data["Low"].squeeze()
stock_data["Close"] = close


# ======================================================
# === INDICATORS (UNCHANGED) ===========================
# ======================================================
stock_data["ema200"] = close.ewm(span=200).mean()
stock_data["ema90"] = close.ewm(span=90).mean()
stock_data["ema30"] = close.ewm(span=30).mean()

stock_data["ema_slope"] = stock_data["ema200"].diff()
stock_data["angle_ema200"] = np.degrees(np.arctan(stock_data["ema_slope"]))

rolling_std = stock_data["angle_ema200"].rolling(20).std()
stock_data["rolling_std"] = rolling_std
stock_data["z"] = stock_data["angle_ema200"] / rolling_std
stock_data["10sigma_avg"] = (rolling_std * 10).rolling(20).mean()

# --- HLEV ---
stock_data["Highest_close_1yr"] = close.rolling(504, min_periods=1).max()
stock_data["Lowest_close_1yr"] = close.rolling(504, min_periods=1).min()
stock_data["HLEV_Origin"] = (
    stock_data["Lowest_close_1yr"]
    + (stock_data["Highest_close_1yr"] - stock_data["Lowest_close_1yr"]) / 2
)

den = (stock_data["HLEV_Origin"] - stock_data["Lowest_close_1yr"]).replace(0, np.nan)
stock_data["HLEV_percentage"] = (stock_data["ema30"] - stock_data["HLEV_Origin"]) / den

# --- ATR ---
tr1 = high - low
tr2 = (high - close.shift()).abs()
tr3 = (low - close.shift()).abs()
true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

stock_data["ATR_20"] = true_range.rolling(20).mean()
stock_data["ATR_pct"] = stock_data["ATR_20"] / close


# ======================================================
# === EQUITY ENGINE (UNCHANGED) ========================
# ======================================================
cross_buy = (
    (
        (stock_data["ema200"] < stock_data["HLEV_Origin"]) &
        (stock_data["ema90"] < stock_data["HLEV_Origin"]) &
        (stock_data["ema30"] < stock_data["ema90"]) &
        (close.shift() < stock_data["ema30"].shift()) &
        (close > stock_data["ema30"])
    )
    |
    (
        (stock_data["angle_ema200"].shift() < -2 * stock_data["rolling_std"].shift()) &
        (stock_data["angle_ema200"] > -2 * stock_data["rolling_std"]) &
        (stock_data["10sigma_avg"] < 10 * stock_data["rolling_std"])
    )
)

cross_sell = (
    (
        (stock_data["ema200"] > stock_data["HLEV_Origin"]) &
        (stock_data["ema90"] > stock_data["HLEV_Origin"]) &
        (stock_data["ema30"] > stock_data["ema90"]) &
        (close.shift() > stock_data["ema30"].shift()) &
        (close < stock_data["ema30"])
    )
)

cross_dates_buy = stock_data.index[cross_buy]
cross_dates_sell = stock_data.index[cross_sell]


# --- Filter multiple buys/sells (stateful) ---
okToBuy = True
okToSell = False

filtered_buy_dates = []
filtered_sell_dates = []

signals = [(d, "BUY") for d in cross_dates_buy] + [(d, "SELL") for d in cross_dates_sell]
signals.sort(key=lambda x: x[0])

for date, signal in signals:
    if signal == "BUY" and okToBuy:
        filtered_buy_dates.append(date)
        okToBuy = False
        okToSell = True
    elif signal == "SELL" and okToSell:
        filtered_sell_dates.append(date)
        okToSell = False
        okToBuy = True

cross_dates_buy = pd.Index(filtered_buy_dates)
cross_dates_sell = pd.Index(filtered_sell_dates)


# ======================================================
# === EQUITY P/L REPORT ================================
# ======================================================
account_equity = ACCOUNT_START
equity_start = account_equity
equity_total_pl = 0.0

num_trades = min(len(cross_dates_buy), len(cross_dates_sell))

print(f"\n===== EQUITY TRADES =====")
print(f"STARTING ACCOUNT BALANCE: {equity_start:.2f}\n")

for i in range(num_trades):
    buy_date = cross_dates_buy[i]
    sell_date = cross_dates_sell[i]

    buy_price = close.loc[buy_date]
    sell_price = close.loc[sell_date]

    atr_val = float(stock_data.loc[buy_date, "ATR_20"])
    shares = int((account_equity * 0.20) / buy_price)
    shares = max(shares, 1)

    trade_pl = (sell_price - buy_price) * shares
    equity_total_pl += trade_pl
    account_equity += trade_pl

    print(
        f"trade{i+1}: "
        f"P/L={trade_pl:.2f}, shares={shares}, "
        f"bought {buy_date.date()} @ {buy_price:.2f}, "
        f"sold {sell_date.date()} @ {sell_price:.2f}, "
        f"account={account_equity:.2f}"
    )

print("\n----------------------------------")
print(f"EQUITY TOTAL P/L  : {equity_total_pl:.2f}")
print(f"EQUITY END BAL    : {account_equity:.2f}")
print("==================================\n")


# ======================================================
# === CSP ENGINE + REPORT ==============================
# ======================================================
csp_trades = []
account_csp = ACCOUNT_START
csp_total_pl = 0.0

if CSP_ENABLED:
    atr_pct = stock_data["ATR_pct"]
    atr_rank = atr_pct.rolling(CSP_VOL_LOOKBACK).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    print("===== CSP TRADES =====")
    print(f"CSP STARTING BALANCE: {ACCOUNT_START:.2f}\n")

    for i in range(CSP_VOL_LOOKBACK, len(stock_data) - CSP_DTE):

        if atr_rank.iloc[i] < CSP_VOL_THRESHOLD:
            continue
        if atr_pct.iloc[i] > atr_pct.iloc[i - 5]:
            continue
        if close.iloc[i] < close.iloc[i - 10] * 0.90:
            continue

        strike = close.iloc[i] - CSP_STRIKE_ATR_MULT * stock_data["ATR_20"].iloc[i]
        strike = round(strike, 2)

        if strike * 100 > ACCOUNT_START * CSP_MAX_CAP_PCT:
            continue

        expiry_i = i + CSP_DTE
        expiry_close = close.iloc[expiry_i]

        premium = round(0.005 * strike, 2)   # conservative model
        credit = premium * 100

        assigned = expiry_close < strike
        if assigned:
            pl = (expiry_close - strike) * 100 + credit
        else:
            pl = credit

        account_csp += pl
        csp_total_pl += pl

        csp_trades.append((stock_data.index[i], strike, stock_data.index[expiry_i], pl, assigned))

        status = "ASSIGNED" if assigned else "EXPIRED"

        print(
            f"entry {stock_data.index[i].date()} "
            f"strike={strike:.2f} "
            f"exp={stock_data.index[expiry_i].date()} "
            f"{status} P/L={pl:.2f} "
            f"csp_account={account_csp:.2f}"
        )

    print("\n----------------------------------")
    print(f"CSP TOTAL P/L   : {csp_total_pl:.2f}")
    print(f"CSP END BAL     : {account_csp:.2f}")
    print("==================================\n")


# ======================================================
# === GRAPH ============================================
# ======================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.set_facecolor("gray")

ax1.plot(close, label="Close")
ax1.plot(stock_data["ema200"], label="EMA200")
ax1.plot(stock_data["ema90"], label="EMA90")
ax1.plot(stock_data["ema30"], label="EMA30")
ax1.plot(stock_data["HLEV_Origin"], color="yellow")

for d in cross_dates_buy:
    ax1.axvline(d, color="cyan", alpha=0.3)
for d in cross_dates_sell:
    ax1.axvline(d, color="magenta", alpha=0.3)

for entry, strike, expiry, _, _ in csp_trades:
    ax1.hlines(strike, entry, expiry, color="orange", linestyle="--")
    ax1.scatter(entry, strike, color="orange", marker="v")

ax2.plot(stock_data["angle_ema200"])
ax2.plot(stock_data["10sigma_avg"], color="gray")
ax3.plot(stock_data["ATR_pct"], color="orange")

for ax in (ax1, ax2, ax3):
    ax.set_facecolor("black")
    ax.grid(True)

plt.show()
