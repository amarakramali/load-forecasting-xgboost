from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor

from src.evaluation import HOURS_PER_DAY, chronological_split
from src.reporting import (
    evaluate_predictions,
    format_result,
    mae_improvement_percent,
    save_results,
)

FEATURES_PATH = r"data\features_aep.csv"
EVALUATION_DAYS = 30
METRICS_PATH = Path("reports") / "xgb_evaluation_metrics.csv"
PLOT_PATH = Path("reports") / "figures" / "xgb_evaluation.png"

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

print("Vergleich (Test: letzte 30 Tage)")
baseline_result = evaluate_predictions("Baseline Blend", y_test, baseline)
xgb_result = evaluate_predictions("XGBoost", y_test, pred)
results = [baseline_result, xgb_result]
for result in results:
    print(format_result(result))
saved_metrics = save_results(results, METRICS_PATH)
print(f"Metriken gespeichert: {saved_metrics}")

impr = mae_improvement_percent(baseline_result, xgb_result)
print(f"\nMAE-Verbesserung vs Baseline: {impr:.1f}%")

# 5) Plot: letzte 7 Tage
plot_start = end - pd.Timedelta(days=7)
plot = test.loc[plot_start:end].copy()
predictions = pd.Series(pred, index=test.index)

figure, axis = plt.subplots()
axis.plot(plot.index, plot["y"], label="Actual")
axis.plot(
    plot.index,
    0.5 * plot["lag_24"] + 0.5 * plot["lag_168"],
    label="Baseline Blend",
)
axis.plot(plot.index, predictions.loc[plot.index], label="XGBoost")
axis.set_title("Forecast Vergleich (letzte 7 Tage im Test)")
axis.set_xlabel("Time")
axis.set_ylabel("MW")
axis.legend()
figure.tight_layout()
PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(PLOT_PATH, dpi=150)
plt.close(figure)
print(f"Diagramm gespeichert: {PLOT_PATH}")
