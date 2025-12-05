import os 
import yfinance as yf 
import pandas as pd 
import numpy as np 
import pickle 
from datetime import datetime, timedelta 
from concurrent.futures import ProcessPoolExecutor, as_completed 
from pathlib import Path 
import time 

# ------------------------------------------- # CONFIG # ------------------------------------------- 
CACHE_DIR = Path("./data_cache") 
CACHE_DIR.mkdir(exist_ok=True) 
LOOKBACK = "2y" # how far back to download 
MAX_WORKERS = 8 # number of CPU processes 
BATCH_SIZE = 300 # number of tickers per batch 
RETRY_COUNT = 3 # retry download attempts 
RECENT_DAYS = 2 # skip cache if older than X days 
# ------------------------------------------- 
# 
# 
# ============================================================ # Indicator calculations (your custom engine goes HERE) # ============================================================ 
def compute_indicators(df): 
    df['SMA20'] = df['Close'].rolling(20).mean() 
    df['SMA50'] = df['Close'].rolling(50).mean() 
    delta = df['Close'].diff() 
    gain = delta.clip(lower=0) 
    loss = (-delta).clip(lower=0) 
    avg_gain = gain.rolling(14).mean() 
    avg_loss = loss.rolling(14).mean() 
    rs = avg_gain / avg_loss 
    df['RSI14'] = 100 - (100 / (1 + rs)) 
    return df 



# ============================================================ # Smart caching logic # ============================================================ def is_cache_valid(symbol): cache_file = CACHE_DIR / f"{symbol}.pkl" if not cache_file.exists(): return False mtime = datetime.fromtimestamp(cache_file.stat().st_mtime) if datetime.now() - mtime < timedelta(days=RECENT_DAYS): return True return False def load_cache(symbol): try: with open(CACHE_DIR / f"{symbol}.pkl", "rb") as f: return pickle.load(f) except: return None def save_cache(symbol, df): try: with open(CACHE_DIR / f"{symbol}.pkl", "wb") as f: pickle.dump(df, f) except Exception as e: print(f"Cache save failed for {symbol}: {e}") # ============================================================ # Worker: download + indicators + cache # ============================================================ def process_symbol(symbol): # --- return cached version if valid if is_cache_valid(symbol): df = load_cache(symbol) if df is not None: return df # --- try downloading with retry for attempt in range(RETRY_COUNT): try: df = yf.download(symbol, period=LOOKBACK, progress=False, threads=False) if not df.empty: df = compute_indicators(df) df['Symbol'] = symbol save_cache(symbol, df) return df except Exception as e: if attempt == RETRY_COUNT - 1: print(f"[FAIL] {symbol}: {e}") time.sleep(0.5) return None # ============================================================ # Batch executor # ============================================================ def process_batch(symbols): results = [] with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex: futures = {ex.submit(process_symbol, sym): sym for sym in symbols} for fut in as_completed(futures): df = fut.result() if df is not None: results.append(df) return results # ============================================================ # SUPERCHARGED ALGO 2 # ============================================================ def super_algo2_all(symbol_list): all_results = [] total = len(symbol_list) batches = [symbol_list[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)] print(f"🔥 Starting SUPERCHARGED ALGO 2") print(f"📈 Total symbols: {total}") print(f"📦 Batch size: {BATCH_SIZE}") print(f"⚙️ Workers: {MAX_WORKERS}") print(f"🧠 Cache age limit: {RECENT_DAYS} days") print("---------------------------------------------------") start_time = time.time() for i, batch in enumerate(batches, 1): print(f"[Batch {i}/{len(batches)}] Processing {len(batch)} symbols...") batch_results = process_batch(batch) all_results.extend(batch_results) print(f" ✔️ Batch {i} complete ({len(batch_results)} successful)") duration = time.time() - start_time print("---------------------------------------------------") print(f"🏁 Completed in {duration:.2f} seconds") if not all_results: return pd.DataFrame() final = pd.concat(all_results, ignore_index=True) return final 
________________________________________






