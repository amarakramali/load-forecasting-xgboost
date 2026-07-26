from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation import HOURS_PER_DAY, trailing_window
from src.reporting import evaluate_predictions, format_result, save_results

FEATURES_PATH = r"data\features_aep.csv"
EVALUATION_DAYS = 30
METRICS_PATH = Path("reports") / "baseline_metrics.csv"
PLOT_PATH = Path("reports") / "figures" / "baseline_evaluation.png"

df = pd.read_csv(FEATURES_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# Testzeitraum: exakt die letzten 30 Tage (stündlich)
end = df.index.max()
test = trailing_window(
    df,
    hours=EVALUATION_DAYS * HOURS_PER_DAY,
)

y_test = test["y"]

# Baselines:
# 1) "Gestern gleiche Stunde"
pred_yesterday = test["lag_24"]

# 2) "Letzte Woche gleiche Stunde"
pred_lastweek = test["lag_168"]

# 3) Mischung (oft überraschend gut)
pred_blend = 0.5 * pred_yesterday + 0.5 * pred_lastweek


print("Baseline-Auswertung (Test: letzte 30 Tage):")
results = [
    evaluate_predictions("Yesterday (lag_24)", y_test, pred_yesterday),
    evaluate_predictions("Last week (lag_168)", y_test, pred_lastweek),
    evaluate_predictions("Blend 50/50", y_test, pred_blend),
]
for result in results:
    print(format_result(result))
saved_metrics = save_results(results, METRICS_PATH)
print(f"Metriken gespeichert: {saved_metrics}")

# Plot: letzte 7 Tage im Test
plot_start = end - pd.Timedelta(days=7)
plot_df = test.loc[plot_start:end, ["y"]].copy()
plot_df["Yesterday"] = pred_yesterday.loc[plot_start:end]
plot_df["LastWeek"] = pred_lastweek.loc[plot_start:end]
plot_df["Blend"] = pred_blend.loc[plot_start:end]

figure, axis = plt.subplots()
axis.plot(plot_df.index, plot_df["y"], label="Actual")
axis.plot(plot_df.index, plot_df["Blend"], label="Blend 50/50")
axis.set_title("Baseline Forecast (last 7 days of test)")
axis.set_xlabel("Time")
axis.set_ylabel("MW")
axis.legend()
figure.tight_layout()
PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(PLOT_PATH, dpi=150)
plt.close(figure)
print(f"Diagramm gespeichert: {PLOT_PATH}")
