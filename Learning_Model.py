from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd



# Create the model
model = LGBMClassifier(
    n_estimators=500, #How many small decision trees the model builds. More trees = more learning (but takes longer).
    learning_rate=0.01, # How big each learning step is. Small value = slower, steadier learning. (0.01 is a safe, stable choice.)
    max_depth=-1, # How deep each tree can go. -1 means “no limit — let LightGBM decide.”
    num_leaves=31, # How many final branches each tree can have. Controls how detailed each tree gets. (31 is a common, balanced default.)
    objective='binary' # Tells the model it’s a yes/no classifier. Perfect for buy/sell, up/down, good/bad type decisions.
)

# Later, when you have your data ready:
# model.fit(X_train, y_train)

# And then it can accept new data:
# predictions = model.predict(X_new)




def extract_waveforms_with_dates(prices, dates, window=30, normalize=True):
    waves = []
    wave_dates = []  # <-- NEW

    for i in range(len(prices) - window):
        segment = prices[i:i+window]
        start_date = dates[i]
        end_date = dates[i+window]

        if normalize:
            segment = segment - segment[0]

        waves.append(np.array(segment))
        wave_dates.append((start_date, end_date))

    return np.array(waves), wave_dates




def feature_slope(wave):
    return wave[-1] - wave[0]


def feature_curvature(wave):
    return np.mean(np.diff(wave, n=2))


def feature_smoothness(wave):
    return np.std(np.diff(wave))


def feature_fft_freq(wave):
    fft_vals = np.abs(np.fft.rfft(wave))
    return np.argmax(fft_vals[1:]) + 1  # skip zero-frequency


def feature_fft_strength(wave):
    fft_vals = np.abs(np.fft.rfft(wave))
    return np.max(fft_vals[1:])


def waveform_to_features(wave):
    return np.array([
        feature_slope(wave),
        feature_curvature(wave),
        feature_smoothness(wave),
        feature_fft_freq(wave),
        feature_fft_strength(wave)
    ])



def build_feature_matrix(waves):
    return np.array([waveform_to_features(w) for w in waves])


def normalize_full_curve(prices):
    prices = np.array(prices)
    norm = prices - prices[0]
    denom = norm.max() if norm.max() != 0 else 1
    return norm / denom


def generate_basis_functions(N):
    x = np.linspace(0, 1, N)

    funcs = {
        "sin(x)": np.sin(2 * np.pi * x),
        "e^x": np.exp(x),
        "e^-x": np.exp(-x),
        "-e^x": -np.exp(x),
        "x": x,
        "x^2": x**2
    }

    # Normalize each function to match price normalization
    for key in funcs:
        f = funcs[key]
        f = f - f[0]
        denom = f.max() if f.max() != 0 else 1
        funcs[key] = f / denom

    return funcs


def compare_to_basis(normalized_curve, basis_funcs):
    errors = {}
    for name, func in basis_funcs.items():
        mse = np.mean((normalized_curve - func)**2)
        errors[name] = mse
    return errors



def rank_function_matches_with_dates(errors, start_date, end_date):
    ranked = sorted(errors.items(), key=lambda x: x[1])
    
    return [
        (name, error, f"{start_date} → {end_date}")
        for name, error in ranked
    ]


def rolling_shape_detection(prices, dates, window=365):
    N = len(prices)
    results = []

    for i in range(N - window):
        segment = prices[i:i+window]

        # Normalize segment like before
        seg = segment - segment[0]
        denom = seg.max() if seg.max() != 0 else 1
        seg = seg / denom

        # Generate matching basis functions
        basis = generate_basis_functions(window)

        # Compute MSE errors
        errors = compare_to_basis(seg, basis)

        # Get best shape
        best_shape = min(errors, key=errors.get)
        best_error = errors[best_shape]

        # Add result
        results.append({
            "start": dates[i],
            "end": dates[i+window],
            "shape": best_shape,
            "error": float(best_error),
        })

    return results

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




