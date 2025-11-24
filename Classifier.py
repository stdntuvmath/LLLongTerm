# Classifier.py (refactored)

import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
from datetime import datetime

# ---------------------------
# CONFIGURATION
# ---------------------------

TICKER_FILE = r"C:\Users\stdnt\Desktop\LootLoader\LongTerm\stock_list_all.csv"
OUTPUT_DIR = r"C:\Users\stdnt\Desktop\LootLoader\LongTerm\classified_data"
BATCH_SIZE = 100
RESUME_FILE = r"C:\Users\stdnt\Desktop\LootLoader\LongTerm\resume.pkl"

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def load_tickers(file_path):
    if not os.path.exists(file_path):
        print(f"[WARN] Ticker file not found: {file_path}")
        # Create an empty template if you like:
        pd.DataFrame({"Ticker": []}).to_csv(file_path, index=False)
        return []
    df = pd.read_csv(file_path)
    if 'Ticker' not in df.columns:
        raise ValueError("Ticker file must contain a column named 'Ticker'")
    return df['Ticker'].dropna().astype(str).tolist()

def fetch_stock_data(tickers, period="1y", interval="1d"):
    try:
        data = yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=True, auto_adjust=True)
        return data
    except Exception as e:
        print(f"[ERROR] Error fetching data: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['std20'] = df['Close'].rolling(20).std()
    df['momentum'] = df['Close'] - df['ma20']
    df.fillna(0, inplace=True)
    return df

def classify_stock(df):
    last_row = df.iloc[-1]
    if last_row['momentum'] > 0:
        return "Long"
    elif last_row['momentum'] < 0:
        return "Short"
    else:
        return "Neutral"

def save_classification(ticker, classification, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump({
            'ticker': ticker,
            'classification': classification,
            'timestamp': datetime.now()
        }, f)

# ---------------------------
# MAIN PROCESS
# ---------------------------

def main():
    tickers = load_tickers(TICKER_FILE)
    print(f"[INFO] Loaded {len(tickers)} tickers from {TICKER_FILE}")
    start_index = 0

    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "rb") as f:
            start_index = pickle.load(f)
        print(f"[INFO] Resuming from index {start_index} ({tickers[start_index] if start_index < len(tickers) else 'N/A'})")

    for i in range(start_index, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"[INFO] Processing batch indices {i} to {i + len(batch) - 1}, {len(batch)} tickers")
        data = fetch_stock_data(batch)

        if data.empty:
            print(f"[WARN] No data for batch {i}-{i+len(batch)-1}. Skipping.")
            continue

        for ticker in batch:
            try:
                if len(batch) > 1:
                    df = data[ticker].copy()
                else:
                    df = data.copy()
                df_indicators = calculate_indicators(df)
                classification = classify_stock(df_indicators)
                save_classification(ticker, classification, OUTPUT_DIR)
            except Exception as e:
                print(f"[ERROR] Processing ticker {ticker}: {e}")
                continue

        # Save progress
        with open(RESUME_FILE, "wb") as f:
            pickle.dump(i + BATCH_SIZE, f)

    print("[INFO] Classification complete!")

if __name__ == "__main__":
    main()
