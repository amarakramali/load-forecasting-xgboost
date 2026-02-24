import pandas as pd
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor

CSV_RAW = r"data\AEP_hourly.csv"
FEATURES_PATH = r"data\features_aep.csv"

OUT_CSV = r"reports\forecast_next24h.csv"
OUT_PNG = r"reports\figures\forecast_next24h.png"
OUT_MODEL = r"models\aep_xgb.joblib"


# ---------- 1) Trainingsdaten (Features) laden ----------
feat = pd.read_csv(FEATURES_PATH)
feat["Datetime"] = pd.to_datetime(feat["Datetime"])
feat = feat.set_index("Datetime").sort_index()

feature_cols = [c for c in feat.columns if c != "y"]
X = feat[feature_cols]
y = feat["y"]

# ---------- 2) Modell trainieren (Finalmodell auf allen Daten) ----------
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)
model.fit(X, y)

joblib.dump({"model": model, "features": feature_cols}, OUT_MODEL)
print(f"Modell gespeichert: {OUT_MODEL}")

# ---------- 3) Historie laden (für Lags/Rolling) ----------
raw = pd.read_csv(CSV_RAW)
raw["Datetime"] = pd.to_datetime(raw["Datetime"])
raw = raw.set_index("Datetime").sort_index()

history = raw["AEP_MW"].copy()

# Duplikate (z.B. wegen DST) zusammenfassen und sortieren
history = history.groupby(history.index).mean().sort_index()

# Auf stündlich bringen (fehlende Stunden -> NaN) und dann auffüllen
history = history.resample("h").mean().ffill()

end = history.index.max()
future_index = pd.date_range(end + pd.Timedelta(hours=1), periods=24, freq="h")

def make_row(ts, hist: pd.Series) -> pd.DataFrame:
    row = pd.DataFrame(index=[ts])

    # Zeitfeatures
    row["hour"] = ts.hour
    row["dayofweek"] = ts.dayofweek
    row["month"] = ts.month
    row["is_weekend"] = int(ts.dayofweek >= 5)

    # Lags
    row["lag_1"] = hist.loc[ts - pd.Timedelta(hours=1)]
    row["lag_24"] = hist.loc[ts - pd.Timedelta(hours=24)]
    row["lag_168"] = hist.loc[ts - pd.Timedelta(hours=168)]

    # Rolling (nur Vergangenheit)
    row["roll_24_mean"] = hist.loc[: ts - pd.Timedelta(hours=1)].tail(24).mean()
    row["roll_168_mean"] = hist.loc[: ts - pd.Timedelta(hours=1)].tail(168).mean()

    return row

# ---------- 4) 24h rekursiv vorhersagen ----------
preds = []
baseline_blend = []

for ts in future_index:
    row = make_row(ts, history)

    # Baseline (Blend 50/50) zum Vergleich
    b = 0.5 * row["lag_24"].iloc[0] + 0.5 * row["lag_168"].iloc[0]
    baseline_blend.append(float(b))

    y_hat = model.predict(row[feature_cols])[0]
    preds.append(y_hat)

    # Prognose in Historie einfügen (für lag_1 bei nächstem Schritt)
    history.loc[ts] = y_hat

# ---------- 5) Export ----------
out = pd.DataFrame(
    {"forecast_xgb_MW": preds, "baseline_blend_MW": baseline_blend},
    index=future_index
)
out.to_csv(OUT_CSV)
print(f"Forecast gespeichert: {OUT_CSV}")

# ---------- 6) Plot (letzte 7 Tage + nächste 24h) ----------
last7 = history.loc[end - pd.Timedelta(days=7): end]

plt.figure()
plt.plot(last7.index, last7.values, label="Actual (last 7 days)")
plt.plot(out.index, out["baseline_blend_MW"], label="Baseline (blend)")
plt.plot(out.index, out["forecast_xgb_MW"], label="XGBoost (next 24h)")
plt.title("AEP Load Forecast: next 24 hours")
plt.xlabel("Time")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.show()

print(f"Plot gespeichert: {OUT_PNG}")