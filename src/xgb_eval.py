import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.evaluation import HOURS_PER_DAY, chronological_split

FEATURES_PATH = r"data\features_aep.csv"
EVALUATION_DAYS = 30

# 1) Features laden
df = pd.read_csv(FEATURES_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# 2) Zeitbasierter Split:
# Train: alles vor den letzten 60 Tagen
# Valid: vorletzte 30 Tage
# Test : letzte 30 Tage
end = df.index.max()
evaluation_hours = EVALUATION_DAYS * HOURS_PER_DAY
train, valid, test = chronological_split(
    df,
    validation_hours=evaluation_hours,
    test_hours=evaluation_hours,
)

feature_cols = [c for c in df.columns if c != "y"]

X_train, y_train = train[feature_cols], train["y"]
X_valid, y_valid = valid[feature_cols], valid["y"]
X_test, y_test = test[feature_cols], test["y"]

# 3) Baseline (Blend 50/50) auf Test
baseline = 0.5 * test["lag_24"] + 0.5 * test["lag_168"]

# 4) XGBoost Modell
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

pred = model.predict(X_test)

def report(name, y, p):
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    print(f"{name:18s}  MAE: {mae:8.2f}   RMSE: {rmse:8.2f}")
    return mae, rmse

print("Vergleich (Test: letzte 30 Tage)")
m_base = report("Baseline Blend", y_test, baseline)
m_xgb  = report("XGBoost", y_test, pred)

impr = (m_base[0] - m_xgb[0]) / m_base[0] * 100
print(f"\nMAE-Verbesserung vs Baseline: {impr:.1f}%")

# 5) Plot: letzte 7 Tage
plot_start = end - pd.Timedelta(days=7)
plot = test.loc[plot_start:end].copy()

plt.figure()
plt.plot(plot.index, plot["y"], label="Actual")
plt.plot(plot.index, (0.5*plot["lag_24"] + 0.5*plot["lag_168"]), label="Baseline Blend")
plt.plot(plot.index, pred[-len(plot):], label="XGBoost")
plt.title("Forecast Vergleich (letzte 7 Tage im Test)")
plt.xlabel("Time")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.show()
