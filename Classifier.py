import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
import yfinance as yf
import datetime as dt


# ============================================================
# RUN CLASSIFIER FOR A STOCK
# ============================================================

def run_classifier(ticker):
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365 * 5)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    if df.empty:
        print(f"No data returned for {ticker}")
        return

    results = classify_waveforms(df)

    print(f"\nWaveform classification for {ticker}:\n")

    for r in results:
        print(
            f"{r['start_date'].date()} → {r['end_date'].date()} "
            f"| cluster={r['waveform_cluster']}"
        )



# ============================================================
# CONFIG
# ============================================================

N_CLUSTERS = 4          # number of waveform types to discover
MIN_SEGMENT_LENGTH = 60  # days
WINDOW_YEARS = 5


# ============================================================
# STEP 1 — LOAD PRICE DATA (expects Date index)
# ============================================================

def load_price_series(df):
    """
    df must contain:
    - index: datetime
    - column: 'Close'
    """
    return df['Close'].dropna()


# ============================================================
# STEP 2 — CHANGE POINT SEGMENTATION
# ============================================================

def segment_waveform(price_series):
    """
    Uses raw price only to find structural changes.
    """
    signal = price_series.values.reshape(-1, 1)

    algo = rpt.Pelt(model="rbf").fit(signal)
    breakpoints = algo.predict(pen=10)

    segments = []
    start = 0

    for bp in breakpoints:
        if bp - start >= MIN_SEGMENT_LENGTH:
            segments.append((start, bp))
        start = bp

    return segments


# ============================================================
# STEP 3 — FEATURE EXTRACTION (ALL COMMENTED BY DESIGN)
# ============================================================

def extract_features(price_series, segments):
    feature_rows = []

    for start, end in segments:
        model = KMeans(n_clusters=N_CLUSTERS, random_state=42)
        labels = model.fit_predict(X)    
        segment = price_series.iloc[start:end]

        # -------------------------------
        # Normalized price
        # Scaled to [0, 1] or z-scored
        # -------------------------------
        # norm_price = (segment - segment.min()) / (segment.max() - segment.min())

        # -------------------------------
        # First derivative (slope)
        # Captures trend direction & strength
        # -------------------------------
        # slope = np.gradient(norm_price).mean()

        # -------------------------------
        # Second derivative (curvature)
        # Captures acceleration / deceleration
        # -------------------------------
        # curvature = np.gradient(np.gradient(norm_price)).mean()

        # -------------------------------
        # Volatility proxy
        # Rolling std or log-return variance
        # -------------------------------
        # returns = np.log(segment / segment.shift(1)).dropna()
        # volatility = returns.std()

        # -------------------------------
        # Normalized price
        # Scaled to [0, 1]
        # -------------------------------
        norm_price = (segment - segment.min()) / (segment.max() - segment.min())

        # Directional shape descriptors
        start_end_delta = float((norm_price.iloc[-1] - norm_price.iloc[0]).iloc[0])
        mean_level = float(norm_price.mean().iloc[0])
        price_range = float((norm_price.max() - norm_price.min()).iloc[0])

        features = [
            start_end_delta,   # overall direction
            mean_level,        # where price lives in its range
            price_range        # structure amplitude
        ]

        feature_rows.append(features)

        return np.array(feature_rows)


# ============================================================
# STEP 4 — CLUSTER SEGMENTS BY SHAPE
# ============================================================

def cluster_segments(feature_matrix):
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    n_samples = X.shape[0]
    n_clusters = min(N_CLUSTERS, n_samples)

    if n_clusters < 2:
        return np.zeros(n_samples, dtype=int)




        return labels


# ============================================================
# STEP 5 — HUMAN-READABLE OUTPUT
# ============================================================

def build_output(price_series, segments, labels):
    output = []

    for (start, end), label in zip(segments, labels):
        output.append({
            "start_date": price_series.index[start],
            "end_date": price_series.index[end - 1],
            "waveform_cluster": int(label)
        })

    return output


# ============================================================
# MAIN PIPELINE
# ============================================================

def classify_waveforms(df):
    price = load_price_series(df)
    segments = segment_waveform(price)
    features = extract_features(price, segments)
    labels = cluster_segments(features)
    return build_output(price, segments, labels)



# ============================================================
# ENTRY POINT
# ============================================================

run_classifier("AMD")





