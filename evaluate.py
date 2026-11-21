"""Will it rain in the next HORIZON_H hours? Model comparison + calibrated probability forecast."""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_FILE = "data.csv"
RAIN_THRESHOLD_MM = 0.3
HORIZON_H = 3
TEST_FRACTION = 0.2
CALIBRATION_FIT_FRACTION = 0.8   # share of the training set used to fit the model; the rest calibrates it
RAIN_WINDOWS_H = (1, 3, 6, 24)
PCA_VARIANCE = 0.95

SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")

LOCATIONS = ["", "_verona", "_mantova", "_brescia", "_cremona", "_parma"]   # "" = home station
LEVEL_BASE = ["temperature_2m", "surface_pressure", "relative_humidity_2m", "cloud_cover",
              "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m"]
DIRECTION_BASE = ["wind_direction_10m", "wind_direction_100m"]

LEVEL_VARS = [f"{v}{loc}" for loc in LOCATIONS for v in LEVEL_BASE]
DIRECTION_VARS = [f"{v}{loc}" for loc in LOCATIONS for v in DIRECTION_BASE]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date").sort_index().asfreq("h")

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def level_features(df: pd.DataFrame) -> dict:
    feats = {}
    for c in (c for c in LEVEL_VARS if c in df.columns):
        feats[c] = df[c]
        feats[f"{c}_d3"] = df[c].diff(3)
        feats[f"{c}_d6"] = df[c].diff(6)
    return feats


def direction_features(df: pd.DataFrame) -> dict:
    feats = {}
    for c in (c for c in DIRECTION_VARS if c in df.columns):
        rad = np.deg2rad(df[c])
        feats[f"{c}_sin"] = np.sin(rad)
        feats[f"{c}_cos"] = np.cos(rad)
    return feats


def rain_history_features(df: pd.DataFrame) -> dict:
    return {f"rain_{w}h": df["rain"].rolling(w).sum() for w in RAIN_WINDOWS_H}


def calendar_features(index: pd.DatetimeIndex) -> dict:
    hour, month = index.hour.to_numpy(), index.month.to_numpy()
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = {
        **level_features(df),
        **direction_features(df),
        **rain_history_features(df),
        **calendar_features(df.index),
    }
    return pd.DataFrame(feats, index=df.index)   # built once, so no fragmentation warning


def build_labels(df: pd.DataFrame):
    """y[t] = 1 if at least RAIN_THRESHOLD_MM falls during t+1 .. t+HORIZON_H."""
    future_rain = df["rain"].rolling(HORIZON_H).sum().shift(-HORIZON_H)
    y = (future_rain >= RAIN_THRESHOLD_MM).astype(int)
    return y, future_rain.notna()


def build_dataset(df: pd.DataFrame):
    X = build_features(df)
    y, has_label = build_labels(df)

    complete = X.notna().all(axis=1)
    X_latest = X[complete].iloc[[-1]]   # most recent hour with complete features (no label needed)
    ok = complete & has_label
    return X[ok], y[ok], X_latest


def time_split(X: pd.DataFrame, y: pd.Series):
    """Chronological split. The gap stops training labels from overlapping the test period."""
    cut = X.index[int(len(X) * (1 - TEST_FRACTION))]
    train = X.index < cut - pd.Timedelta(hours=HORIZON_H)
    test = X.index >= cut
    return X[train], y[train], X[test], y[test], cut


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def report(name, y_true, y_pred, y_score=None):
    pr_auc = average_precision_score(y_true, y_score) if y_score is not None else float("nan")
    print(
        f"{name:<30} precision={precision_score(y_true, y_pred, zero_division=0):.3f}  "
        f"recall={recall_score(y_true, y_pred, zero_division=0):.3f}  "
        f"F1={f1_score(y_true, y_pred, zero_division=0):.3f}  PR-AUC={pr_auc:.3f}"
    )


def fit_and_report(name, model, X_tr, y_tr, X_te, y_te):
    """Fit, then report using predict_proba when available, else decision_function."""
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_te)[:, 1]
    else:
        score = model.decision_function(X_te)
    report(name, y_te, model.predict(X_te), score)
    return model


def make_logreg():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def make_gb():
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=0)


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def evaluate_persistence(X_te, y_te):
    """Baseline: "it rains soon if it rained in the last hour"."""
    report("Persistence baseline", y_te, (X_te["rain_1h"] >= RAIN_THRESHOLD_MM).astype(int))


def evaluate_unscaled_models(X_tr, y_tr, X_te, y_te):
    fit_and_report("Logistic regression", make_pipeline(StandardScaler(), make_logreg()),
                   X_tr, y_tr, X_te, y_te)
    fit_and_report("Gradient boosting", make_gb(), X_tr, y_tr, X_te, y_te)


def scale_and_reduce(X_tr, X_te):
    """Fit scaler and PCA on the training set only, then apply them to the test set."""
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    pca = PCA(n_components=PCA_VARIANCE).fit(X_tr_s)
    X_tr_p, X_te_p = pca.transform(X_tr_s), pca.transform(X_te_s)
    print(f"\nPCA keeps {pca.n_components_} of {X_tr.shape[1]} components "
          f"({PCA_VARIANCE:.0%} of the variance)")
    return X_tr_s, X_te_s, X_tr_p, X_te_p


def evaluate_scaled_models(X_tr, y_tr, X_te, y_te):
    X_tr_s, X_te_s, X_tr_p, X_te_p = scale_and_reduce(X_tr, X_te)
    fit_and_report("Logistic regression + PCA", make_logreg(), X_tr_p, y_tr, X_te_p, y_te)
    fit_and_report("SVM (RBF kernel)", SVC(**SVM_PARAMS), X_tr_s, y_tr, X_te_s, y_te)   # slow


# --------------------------------------------------------------------------- #
# Probability forecast
# --------------------------------------------------------------------------- #
def fit_calibrated_gb(X_tr, y_tr):
    """Fit gradient boosting on the first part of the training set, calibrate on the rest."""
    n_fit = int(len(X_tr) * CALIBRATION_FIT_FRACTION)
    X_fit, y_fit = X_tr.iloc[: n_fit - HORIZON_H], y_tr.iloc[: n_fit - HORIZON_H]   # gap: no label overlap
    X_cal, y_cal = X_tr.iloc[n_fit:], y_tr.iloc[n_fit:]

    model = HistGradientBoostingClassifier(random_state=0).fit(X_fit, y_fit)
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X_cal)[:, 1], y_cal)
    return model, iso


def print_probability_scores(y_te, forecasts: dict):
    print("\n--- Probability forecast (lower is better) ---")
    for name, p in forecasts.items():
        print(f"{name:<24} Brier={brier_score_loss(y_te, p):.4f}  "
              f"log-loss={log_loss(y_te, np.clip(p, 1e-4, 1 - 1e-4)):.4f}")


def print_reliability_table(y_te, p):
    """When the model says X%, how often does it really rain?"""
    bins = pd.cut(p, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    table = (pd.DataFrame({"p": p, "rained": y_te.values})
             .groupby(bins, observed=True)
             .agg(n=("rained", "size"), predicted=("p", "mean"), observed=("rained", "mean")))
    print("\nReliability (predicted vs observed frequency):")
    print(table.round(3).to_string())


def probability_forecast(X_tr, y_tr, X_te, y_te, X_latest):
    model, iso = fit_calibrated_gb(X_tr, y_tr)

    p_raw = model.predict_proba(X_te)[:, 1]
    p_cal = np.clip(iso.predict(p_raw), 1e-4, 1 - 1e-4)
    p_clim = np.full(len(y_te), y_tr.mean())   # always predict the average rain frequency

    print_probability_scores(y_te, {
        "Climatology (constant)": p_clim,
        "Gradient boosting": p_raw,
        "GB + calibration": p_cal,
    })
    skill = 1 - brier_score_loss(y_te, p_cal) / brier_score_loss(y_te, p_clim)
    print(f"Brier skill score vs climatology: {skill:.3f}  (0 = no better than the average, 1 = perfect)")

    print_reliability_table(y_te, p_cal)

    p_now = iso.predict(model.predict_proba(X_latest)[:, 1])[0]
    print(f"\nProbability of rain in the {HORIZON_H} h after {X_latest.index[0]:%Y-%m-%d %H:%M} UTC: {p_now:.0%}")


def main():
    df = load()
    X, y, X_latest = build_dataset(df)
    print(f"{len(X)} hourly samples from {X.index.min():%Y-%m-%d} to {X.index.max():%Y-%m-%d}, "
          f"{X.shape[1]} features")
    print(f"Share of 'rain in next {HORIZON_H}h': {y.mean():.1%}\n")

    X_tr, y_tr, X_te, y_te, cut = time_split(X, y)
    print(f"Train: {len(X_tr)} rows   Test: {len(X_te)} rows (from {cut:%Y-%m-%d})")
    print(f"Always-'no rain' accuracy on test: {1 - y_te.mean():.1%}  <- why accuracy is not reported\n")

    evaluate_persistence(X_te, y_te)
    evaluate_unscaled_models(X_tr, y_tr, X_te, y_te)
    evaluate_scaled_models(X_tr, y_tr, X_te, y_te)

    probability_forecast(X_tr, y_tr, X_te, y_te, X_latest)


if __name__ == "__main__":
    main()