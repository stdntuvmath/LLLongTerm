import yfinance as yf
import numpy as np
import datetime as dt


# =====================================
# Normalization of full 25-year curve
# =====================================
def normalize_full_curve(prices):
    prices = np.array(prices)
    norm = prices - prices[0]
    denom = norm.max() if norm.max() != 0 else 1
    return norm / denom


# =====================================
# Generate basis functions (x, x^2, sin, exp, etc.)
# =====================================
def generate_basis_functions(N):
    x = np.linspace(0, 1, N)

    funcs = {
        "sin(x)": np.sin(2 * np.pi * x),
        "e^x": np.exp(x),
        "e^-x": np.exp(-x),
        "-e^x": -np.exp(x),
        "x": x,
        "-x": -x,
        "x^2": x**2
    }

    # Normalize functions same way the price curve is normalized
    for key in funcs:
        f = funcs[key]
        f = f - f[0]
        denom = f.max() if f.max() != 0 else 1
        funcs[key] = f / denom

    return funcs


# =====================================
# Compare normalized price curve to each function
# =====================================
def compare_to_basis(normalized_curve, basis_funcs):
    errors = {}
    for name, f in basis_funcs.items():
        mse = np.mean((normalized_curve - f)**2)
        errors[name] = mse
    return errors


# =====================================
# Rank function matches and attach date range
# =====================================
def rank_function_matches_with_dates(errors, start_date, end_date):
    ranked = sorted(errors.items(), key=lambda x: x[1])
    return [
        (name, float(error), f"{start_date} → {end_date}")
        for name, error in ranked
    ]


# =====================================
# Rolling Shape Detection (365-day windows)
# =====================================
def rolling_shape_detection(prices, dates, window=365):
    N = len(prices)
    results = []

    for i in range(N - window):
        # extract window prices
        segment = prices[i:i+window]

        # normalize segment
        seg = segment - segment[0]
        denom = seg.max() if seg.max() != 0 else 1
        seg = seg / denom

        # generate basis functions for this window
        basis = generate_basis_functions(window)

        # compute errors
        errors = compare_to_basis(seg, basis)

        # pick best-matching shape
        best_shape = min(errors, key=errors.get)
        best_error = float(errors[best_shape])

        results.append({
            "start": dates[i],
            "end": dates[i+window],
            "shape": best_shape,
            "error": best_error,
        })

    return results


# =====================================
# Consolidate consecutive windows with same shape
# =====================================
def consolidate_shape_periods(results):
    if not results:
        return []

    consolidated = []
    current_shape = results[0]["shape"]
    start_date = results[0]["start"]
    last_end = results[0]["end"]

    for r in results[1:]:
        if r["shape"] == current_shape:
            last_end = r["end"]
        else:
            consolidated.append((current_shape, start_date, last_end))
            current_shape = r["shape"]
            start_date = r["start"]
            last_end = r["end"]

    consolidated.append((current_shape, start_date, last_end))
    return consolidated


# =====================================
# MAIN PROGRAM
# =====================================
def main():
    ticker = "AAL"  # change to any stock, e.g., "AAPL", "MSFT", etc.
    print(f"\nDownloading 25-year history for {ticker}...\n")

    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365 * 25)

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print("Error: No data downloaded.")
        return

    close_prices = data["Close"].values
    dates = data.index

    # ============================
    # 1. FULL CURVE SHAPE MATCHING
    # ============================
    norm_curve = normalize_full_curve(close_prices)
    basis = generate_basis_functions(len(close_prices))
    errors = compare_to_basis(norm_curve, basis)

    start_str = dates[0].strftime("%Y-%m-%d")
    end_str = dates[-1].strftime("%Y-%m-%d")
    long_term_results = rank_function_matches_with_dates(errors, start_str, end_str)

    print("=== LONG-TERM SHAPE RESEMBLANCE (FULL 25 YEARS) ===\n")
    for row in long_term_results:
        print(row)

    # ============================
    # 2. ROLLING SHAPE DETECTION
    # ============================
    print("\n\nRunning rolling shape detection (365-day windows)...\n")
    rolling = rolling_shape_detection(close_prices, dates, window=365)
    periods = consolidate_shape_periods(rolling)

    print("=== SHAPE PERIODS OVER TIME ===\n")
    for shape, start, end in periods:
        print(f"{start.date()} → {end.date()} : {shape}")


# =====================================
# RUN MAIN
# =====================================
if __name__ == "__main__":
    main()
