import sklearn
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score,brier_score_loss,log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVC

DATA_FILE = "data.csv"
RAIN_THRESHOLD_MM = 0.1
HORIZON_H = 6
TEST_FRACTION = 0.2

LEVEL_VARS = ["temperature_2m", "surface_pressure", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m",
              "temperature_2m_verona", "surface_pressure_verona", "relative_humidity_2m_verona", "cloud_cover_verona", "wind_speed_10m_verona", "wind_speed_100m_verona", "wind_gusts_10m_verona",
              "temperature_2m_mantova", "surface_pressure_mantova", "relative_humidity_2m_mantova", "cloud_cover_mantova", "wind_speed_10m_mantova", "wind_speed_100m_mantova", "wind_gusts_10m_mantova"
              "temperature_2m_brescia", "surface_pressure_brescia", "relative_humidity_2m_brescia", "cloud_cover_brescia", "wind_speed_10m_brescia", "wind_speed_100m_brescia", "wind_gusts_10m_brescia",
              "temperature_2m_cremona", "surface_pressure_cremona", "relative_humidity_2m_cremona", "cloud_cover_cremona", "wind_speed_10m_cremona", "wind_speed_100m_cremona", "wind_gusts_10m_cremona",
              "temperature_2m_parma", "surface_pressure_parma", "relative_humidity_2m_parma", "cloud_cover_parma", "wind_speed_10m_parma", "wind_speed_100m_parma", "wind_gusts_10m_parma"]

DIRECTION_VARS = ["wind_direction_10m", "wind_direction_100m",
                  "wind_direction_10m_verona", "wind_direction_100m_verona",
                  "wind_direction_10m_mantova", "wind_direction_100m_mantova",
                  "wind_direction_10m_brescia", "wind_direction_100m_brescia",
                  "wind_direction_10m_cremona", "wind_direction_100m_cremona",
                  "wind_direction_10m_parma", "wind_direction_100m_parma"]

def load() -> pd.DataFrame:
    df = pd.read_csv("data.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if df["date"].duplicated().any():
        print("Warning: duplicated timestamps in data.csv (two models in one file?). Keeping the first of each.")
        df = df.drop_duplicates("date", keep="first")
    df = df.set_index("date").sort_index().asfreq("h")
    return df


def build_dataset(df: pd.DataFrame):
    X = pd.DataFrame(index=df.index)
    for c in [c for c in LEVEL_VARS if c in df.columns]:
        X[c] = df[c]
        X[f"{c}_d3"] = df[c] - df[c].shift(3)
        X[f"{c}_d6"] = df[c] - df[c].shift(6)
    for c in [c for c in DIRECTION_VARS if c in df.columns]:
        X[f"{c}_sin"] = np.sin(np.deg2rad(df[c]))
        X[f"{c}_cos"] = np.cos(np.deg2rad(df[c]))
    for w in (1, 3, 6, 24):
        X[f"rain_{w}h"] = df["rain"].rolling(w).sum()
    hour, month = df.index.hour, df.index.month
    
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
 
    future_rain = df["rain"].rolling(HORIZON_H).sum().shift(-HORIZON_H)  # sum over t+1..t+6
    y = (future_rain >= RAIN_THRESHOLD_MM).astype(int)
    
    X_latest = X[X.notna().all(axis=1)].iloc[[-1]]  # most recent hour with complete features (no label needed)
    ok = X.notna().all(axis=1) & future_rain.notna()
    return X[ok], y[ok], X_latest

def report(name, y_true, y_pred, y_score=None):
    pr_auc = average_precision_score(y_true, y_score) if y_score is not None else float("nan")
    print(
        f"{name:<30} precision={precision_score(y_true, y_pred, zero_division=0):.3f}  "
        f"recall={recall_score(y_true, y_pred, zero_division=0):.3f}  "
        f"F1={f1_score(y_true, y_pred, zero_division=0):.3f}  PR-AUC={pr_auc:.3f}"
    )

def probability_forecast(X_tr, y_tr, X_te, y_te, X_latest):

    n_fit = int(len(X_tr) * 0.8)
    X_fit, y_fit = X_tr.iloc[: n_fit - HORIZON_H], y_tr.iloc[: n_fit - HORIZON_H]  # gap: no label overlap
    X_cal, y_cal = X_tr.iloc[n_fit:], y_tr.iloc[n_fit:]
 
    model = HistGradientBoostingClassifier(random_state=0).fit(X_fit, y_fit)
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X_cal)[:, 1], y_cal)
 
    p_raw = model.predict_proba(X_te)[:, 1]
    p_cal = np.clip(iso.predict(p_raw), 1e-4, 1 - 1e-4)
    p_clim = np.full(len(y_te), y_tr.mean())  # always predict the average rain frequency
 
    print("\n--- Probability forecast (lower is better) ---")
    for name, p in [("Climatology (constant)", p_clim), ("Gradient boosting", p_raw), ("GB + calibration", p_cal)]:
        print(f"{name:<24} Brier={brier_score_loss(y_te, p):.4f}  log-loss={log_loss(y_te, np.clip(p, 1e-4, 1 - 1e-4)):.4f}")
    skill = 1 - brier_score_loss(y_te, p_cal) / brier_score_loss(y_te, p_clim)
    print(f"Brier skill score vs climatology: {skill:.3f}  (0 = no better than the average, 1 = perfect)")
 
    # Reliability table: when the model says X%, how often does it really rain?
    bins = pd.cut(p_cal, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)
    table = pd.DataFrame({"p": p_cal, "rained": y_te.values}).groupby(bins, observed=True).agg(
        n=("rained", "size"), predicted=("p", "mean"), observed=("rained", "mean"))
    print("\nReliability (predicted vs observed frequency):")
    print(table.round(3).to_string())
 
    p_now = iso.predict(model.predict_proba(X_latest)[:, 1])[0]
    print(f"\nProbability of rain in the {HORIZON_H} h after {X_latest.index[0]:%Y-%m-%d %H:%M} UTC: {p_now:.0%}")

def main():
    X, y, X_latest = build_dataset(load())
    print(f"{len(X)} hourly samples from {X.index.min():%Y-%m-%d} to {X.index.max():%Y-%m-%d}, {X.shape[1]} features")
    print(f"Share of 'rain in next {HORIZON_H}h': {y.mean():.1%}\n")
 
    # Time-based split. The gap stops training labels from overlapping the test period.
    cut = X.index[int(len(X) * (1 - TEST_FRACTION))]
    train = X.index < cut - pd.Timedelta(hours=HORIZON_H)
    test = X.index >= cut
    X_tr, y_tr, X_te, y_te = X[train], y[train], X[test], y[test]
    print(f"Train: {len(X_tr)} rows   Test: {len(X_te)} rows (from {cut:%Y-%m-%d})")
    print(f"Always-'no rain' accuracy on test: {1 - y_te.mean():.1%}  <- why accuracy is not reported\n")
 
    # Baseline: "it rains soon if it rained in the last hour"
    report("Persistence baseline", y_te, (X_te["rain_1h"] >= RAIN_THRESHOLD_MM).astype(int))
 
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    logreg.fit(X_tr, y_tr)
    report("Logistic regression", y_te, logreg.predict(X_te), logreg.predict_proba(X_te)[:, 1])
 
    gb = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    gb.fit(X_tr, y_tr)
    report("Gradient boosting", y_te, gb.predict(X_te), gb.predict_proba(X_te)[:, 1])
 
        # Small experiment: how much does pressure help?
    no_p = [c for c in X.columns if not c.startswith("surface_pressure")]
    if len(no_p) < X.shape[1]:
        gb2 = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
        gb2.fit(X_tr[no_p], y_tr)
        report("Gradient boosting, no pressure", y_te, gb2.predict(X_te[no_p]), gb2.predict_proba(X_te[no_p])[:, 1])
 
    # Dimensionality reduction (PCA), step by step without a pipeline.
    # Each step is fitted on the training set only, then applied to the test set.
    scaler = StandardScaler().fit(X_tr)

    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    pca = PCA(n_components=0.95).fit(X_tr_s)  # keep enough components for 95% of the variance

    X_tr_p, X_te_p = pca.transform(X_tr_s), pca.transform(X_te_s)

    print(f"\nPCA keeps {pca.n_components_} of {X_tr.shape[1]} components (95% of the variance)")

    lr_pca = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr_p, y_tr)
    report("Logistic regression + PCA", y_te, lr_pca.predict(X_te_p), lr_pca.predict_proba(X_te_p)[:, 1])

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced").fit(X_tr_s, y_tr)
    report("SVM (RBF kernel)", y_te, svm.predict(X_te_s), svm.decision_function(X_te_s))
    
    probability_forecast(X_tr, y_tr, X_te, y_te, X_latest)
 
if __name__ == "__main__":
    main()