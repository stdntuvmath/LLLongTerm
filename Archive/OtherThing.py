# Classifier.py
# Optimized for throughput: classify as many stocks as possible per day

import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
from datetime import datetime

# ---------------------------
# CONFIGURATION
# ---------------------------

# List of stock tickers (can be hundreds or thousands)
TICKER_FILE = "stock_list.csv"  # CSV with a column 'Ticker'
OUTPUT_DIR = "classified_data"
BATCH_SIZE = 100  # Adjust to limit memory usage
RESUME_FILE = "resume.pkl"  # To resume from last processed stock

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def load_tickers(file_path):
    df = pd.read_csv(file_path)
    return df['Ticker'].tolist()

def fetch_stock_data(tickers, period="1y", interval="1d"):
    """Fetch stock data in batch using yfinance"""
    try:
        data = yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=True, auto_adjust=True)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculate basic indicators for classification"""
    df = df.copy()
    
    # Price changes
    df['returns'] = df['Close'].pct_change()
    
    # Rolling stats
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['std20'] = df['Close'].rolling(20).std()
    
    # Example classification feature: momentum
    df['momentum'] = df['Close'] - df['ma20']
    
    # Fill NaNs
    df.fillna(0, inplace=True)
    return df

def classify_stock(df):
    """Simple classification logic. Modify as needed for your algo."""
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
        pickle.dump({'ticker': ticker, 'classification': classification, 'timestamp': datetime.now()}, f)

# ---------------------------
# MAIN PROCESS
# ---------------------------

def main():
    tickers = load_tickers(TICKER_FILE)
    start_index = 0

    # Resume logic
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "rb") as f:
            start_index = pickle.load(f)
        print(f"Resuming from index {start_index} ({tickers[start_index]})")

    for i in range(start_index, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"Processing batch {i}-{i + len(batch)-1} ({len(batch)} stocks)")
        data = fetch_stock_data(batch)
        
        if data.empty:
            print(f"No data fetched for batch {i}-{i + len(batch)-1}, skipping")
            continue

        # yfinance returns multi-index if multiple tickers
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
                print(f"Error processing {ticker}: {e}")
                continue

        # Save progress
        with open(RESUME_FILE, "wb") as f:
            pickle.dump(i + BATCH_SIZE, f)

    print("Classification complete!")

if __name__ == "__main__":
    main()




