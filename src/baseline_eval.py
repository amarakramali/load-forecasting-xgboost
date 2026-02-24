import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_PATH = r"data\features_aep.csv"

df = pd.read_csv(FEATURES_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

y_true = df["y"]

# Testzeitraum: letzte 30 Tage (stündlich)
end = df.index.max()
start_test = end - pd.Timedelta(days=30)
test = df.loc[start_test:end].copy()

y_test = test["y"]

# Baselines:
# 1) "Gestern gleiche Stunde"
pred_yesterday = test["lag_24"]

# 2) "Letzte Woche gleiche Stunde"
pred_lastweek = test["lag_168"]

# 3) Mischung (oft überraschend gut)
pred_blend = 0.5 * pred_yesterday + 0.5 * pred_lastweek


def report(name, y, p):
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    print(f"{name:18s}  MAE: {mae:8.2f}   RMSE: {rmse:8.2f}")
    return mae, rmse


print("Baseline-Auswertung (Test: letzte 30 Tage):")
m1 = report("Yesterday (lag_24)", y_test, pred_yesterday)
m2 = report("Last week (lag_168)", y_test, pred_lastweek)
m3 = report("Blend 50/50", y_test, pred_blend)

# Plot: letzte 7 Tage im Test
plot_start = end - pd.Timedelta(days=7)
plot_df = test.loc[plot_start:end, ["y"]].copy()
plot_df["Yesterday"] = pred_yesterday.loc[plot_start:end]
plot_df["LastWeek"] = pred_lastweek.loc[plot_start:end]
plot_df["Blend"] = pred_blend.loc[plot_start:end]

plt.figure()
plt.plot(plot_df.index, plot_df["y"], label="Actual")
plt.plot(plot_df.index, plot_df["Blend"], label="Blend 50/50")
plt.title("Baseline Forecast (last 7 days of test)")
plt.xlabel("Time")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.show()