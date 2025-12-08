import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
from hmmlearn import hmm


# ============================================================
# MODULE 1 — Normalization & Basis Functions
# ============================================================

def normalize_curve(curve):
    curve = np.array(curve)
    curve = curve - curve[0]
    denom = curve.max() if curve.max() != 0 else 1
    return curve / denom


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

    for key in funcs:
        funcs[key] = normalize_curve(funcs[key])

    return funcs


def compare_to_basis(norm_segment, basis_funcs):
    errors = {}
    for name, f in basis_funcs.items():
        mse = np.mean((norm_segment - f[:len(norm_segment)]) ** 2)
        errors[name] = mse
    return errors


# ============================================================
# MODULE 2 — Rolling Shape Detection + Consolidation
# ============================================================

def rolling_shape_detection(prices, dates, window=365):
    N = len(prices)
    results = []

    for i in range(N - window):
        segment = prices[i:i+window]
        seg_norm = normalize_curve(segment)
        basis = generate_basis_functions(window)
        errors = compare_to_basis(seg_norm, basis)

        best_shape = min(errors, key=errors.get)
        best_error = float(errors[best_shape])

        results.append({
            "start": dates[i],
            "end": dates[i+window],
            "shape": best_shape,
            "error": best_error
        })

    return results


def consolidate_shape_periods(results):
    if not results:
        return []

    out = []
    curr_shape = results[0]["shape"]
    start_date = results[0]["start"]
    last_end = results[0]["end"]

    for r in results[1:]:
        if r["shape"] == curr_shape:
            last_end = r["end"]
        else:
            out.append((curr_shape, start_date, last_end))
            curr_shape = r["shape"]
            start_date = r["start"]
            last_end = r["end"]

    out.append((curr_shape, start_date, last_end))
    return out


# ============================================================
# MODULE 3 — HMM Regime Learning
# ============================================================

SHAPE_TO_INT = {
    "x": 0,
    "x^2": 1,
    "e^x": 2,
    "e^-x": 3,
    "-x": 4,
    "-e^x": 5,
    "sin(x)": 6
}

INT_TO_REGIME = {
    0: "Flat",
    1: "Acceleration",
    2: "Growth",
    3: "Stabilization",
    4: "Down Drift",
    5: "Crash",
    6: "Oscillation"
}


def encode_sequence(consolidated_periods):
    seq = []
    for shape, _, _ in consolidated_periods:
        seq.append([SHAPE_TO_INT[shape]])
    return np.array(seq)


def train_hmm(seq, n_states=5):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200)
    model.fit(seq)
    return model


def infer_current_regime(model, seq):
    hidden_states = model.predict(seq)
    return hidden_states[-1]


# ============================================================
# MODULE 4 — Indicator Map
# ============================================================

indicator_map = {
    "Growth": ["EMA30", "MACD", "angle_up", "volume_trend"],
    "Acceleration": ["momentum", "breakouts", "MACD_strong"],
    "Crash": ["RSI_oversold", "volatility_spike"],
    "Stabilization": ["EMA_contraction", "MACD_reversal"],
    "Flat": ["RSI_mean_reversion", "Bollinger_mid"],
    "Down Drift": ["bottom_finder", "divergence_scan"],
    "Oscillation": ["mean_revert", "Bollinger", "stoch_cross"]
}


# ============================================================
# MODULE 5 — EMA200 Angle Strategy
# ============================================================

def angle(series):
    dy = series[-1] - series[-5]
    dx = 5
    return np.degrees(np.arctan(dy/dx))


def ema200_angle_signal(prices):
    prices = np.array(prices).flatten()
    ema200 = pd.Series(prices).ewm(span=200).mean().values
    ang = angle(ema200)

    if ang > 12:
        return "BUY", ang
    elif ang < -8:
        return "SELL", ang
    return "HOLD", ang


# ============================================================
# MODULE 6 — HLEV (25-year high-low elevation)
# ============================================================

def hlev_25yrs(prices):
    prices = np.array(prices).flatten()
    low = np.min(prices)
    high = np.max(prices)
    if high == low:
        return 0.5
    return (prices[-1] - low) / (high - low)


def hlev_signal(prices):
    h = hlev_25yrs(prices)

    if h < 0.20:
        return "BUY", h
    elif h > 0.80:
        return "SELL", h
    return "HOLD", h


# ============================================================
# MODULE 7 — Decision Engine
# ============================================================

def trading_decision(regime_name, ema_signal, hlev_signal):

    decisions = []

    # EMA200 angle
    if ema_signal == "BUY":
        decisions.append("BUY")
    elif ema_signal == "SELL":
        decisions.append("SELL")

    # HLEV
    if hlev_signal == "BUY":
        decisions.append("BUY")
    elif hlev_signal == "SELL":
        decisions.append("SELL")

    # Final hierarchy
    if "BUY" in decisions:
        final = "BUY"
    elif "SELL" in decisions:
        final = "SELL"
    else:
        final = "HOLD"

    return final


# ============================================================
# MAIN PIPELINE + TEXT OUTPUT
# ============================================================

def main():
    ticker = "AAL"

    end = dt.datetime.today()
    start = end - dt.timedelta(days=365*25)

    print(f"Downloading {ticker}...")
    data = yf.download(ticker, start=start, end=end)

    prices = data["Close"].values.flatten()
    dates = data.index

    # 1. SHAPE REGIMES
    rolling = rolling_shape_detection(prices, dates)
    consolidated = consolidate_shape_periods(rolling)

    # 2. HMM
    seq = encode_sequence(consolidated)
    model = train_hmm(seq)
    state = infer_current_regime(model, seq)
    regime_name = INT_TO_REGIME[state]

    # 3. EMA200 ANGLE
    ema_sig, ema_ang = ema200_angle_signal(prices)

    # 4. HLEV
    hlev_sig, hlev_val = hlev_signal(prices)

    # 5. FINAL DECISION
    final = trading_decision(regime_name, ema_sig, hlev_sig)

    # ========================================================
    # WRITE TO TEXT FILE
    # ========================================================
    with open("RegimeOutput.txt", "w") as f:
        f.write(f"=== REGIME CLASSIFICATION RESULTS ===\n\n")
        f.write(f"Ticker: {ticker}\n\n")

        f.write("=== SHAPE REGIMES ===\n")
        for shape, s, e in consolidated:
            f.write(f"{s.date()} -> {e.date()} : {shape}\n")
        f.write("\n")

        f.write(f"Current Inferred Regime: {regime_name}\n")
        f.write(f"EMA200 Angle: {ema_ang:.2f}°, Signal: {ema_sig}\n")
        f.write(f"HLEV 25-Year: {hlev_val:.3f}, Signal: {hlev_sig}\n")
        f.write(f"\nFINAL DECISION: {final}\n")

    # PRINT FOR USER
    print("\n=== RESULTS ===")
    print(f"Current Regime: {regime_name}")
    print(f"EMA200 Angle: {ema_ang:.2f}°, Signal: {ema_sig}")
    print(f"HLEV: {hlev_val:.3f}, Signal: {hlev_sig}")
    print(f"FINAL DECISION: {final}")
    print("\nOutput also written to RegimeOutput.txt")


if __name__ == "__main__":
    main()
