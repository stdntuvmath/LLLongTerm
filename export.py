import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
from datetime import datetime

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = r"C:\Users\stdnt\Desktop\LootLoader\LongTerm"
TICKER_FILE = os.path.join(BASE_DIR, "stock_list_all.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "classified_data")
RESUME_FILE = os.path.join(BASE_DIR, "resume.pkl")
RESULT_CSV = os.path.join(BASE_DIR, "classified_results.csv")
RESULT_TXT = os.path.join(BASE_DIR, "classified_results.txt")

BATCH_SIZE = 100


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_tickers(file_path):
    if not os.path.exists(file_path):
        print(f"[WARN] Ticker file missing: {file_path}")
        pd.DataFrame({"Ticker": []}).to_csv(file_path, index=False)
        return []
    df = pd.read_csv(file_path)
    if "Ticker" not in df.columns:
        raise ValueError("Ticker file must contain a column named 'Ticker'")
    return df["Ticker"].dropna().astype(str).tolist()


def fetch_stock_data(tickers, period="1y", interval="1d"):
    try:
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            threads=True,
            auto_adjust=True
        )
        return data
    except Exception as e:
        print(f"[ERROR] fetch failed: {e}")
        return pd.DataFrame()


def calculate_indicators(df):
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["std20"] = df["Close"].rolling(20).std()
    df["momentum"] = df["Close"] - df["ma20"]
    df.fillna(0, inplace=True)
    return df


def classify_stock(df):
    last = df.iloc[-1]
    if last["momentum"] > 0:
        return "Long"
    elif last["momentum"] < 0:
        return "Short"
    else:
        return "Neutral"


def save_classification(ticker, classification):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"{ticker}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump({
            "ticker": ticker,
            "classification": classification,
            "timestamp": datetime.now()
        }, f)


# =====================================================
# EXPORT RESULTS TO CSV AND TXT
# =====================================================

def export_results():
    print("\n[INFO] Exporting final results...")

    records = []

    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".pkl"):
            try:
                with open(os.path.join(OUTPUT_DIR, file), "rb") as f:
                    records.append(pickle.load(f))
            except Exception as e:
                print(f"Could not read {file}: {e}")

    if not records:
        print("[WARN] No classification files found.")
        return

    df = pd.DataFrame(records)
    df.sort_values("ticker", inplace=True)

    # Write CSV
    df.to_csv(RESULT_CSV, index=False)

    # Write TXT
    with open(RESULT_TXT, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['ticker']}\t{row['classification']}\t{row['timestamp']}\n")

    print(f"[INFO] CSV saved → {RESULT_CSV}")
    print(f"[INFO] TXT saved → {RESULT_TXT}")
    print(f"[INFO] Total tickers exported: {len(df)}")


# =====================================================
# MAIN PROCESS
# =====================================================

def main():
    tickers = load_tickers(TICKER_FILE)
    print(f"[INFO] Loaded {len(tickers)} tickers.")

    start_index = 0

    # Resume ability
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "rb") as f:
            start_index = pickle.load(f)
        print(f"[INFO] Resuming from index {start_index}")

    # Batch processing
    for i in range(start_index, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"\n[INFO] Processing batch {i} to {i + len(batch) - 1}")

        data = fetch_stock_data(batch)
        if data.empty:
            print(f"[WARN] No data for batch {i} - skipping")
            continue

        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data.copy()
                df_ind = calculate_indicators(df)
                classification = classify_stock(df_ind)
                save_classification(ticker, classification)
            except Exception as e:
                print(f"[ERROR] Failed on {ticker}: {e}")

        # Save resume position
        with open(RESUME_FILE, "wb") as f:
            pickle.dump(i + BATCH_SIZE, f)

    print("\n[INFO] Classification complete!")

    # EXPORT RESULTS
    export_results()


if __name__ == "__main__":
    main()
