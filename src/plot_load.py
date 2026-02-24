import pandas as pd
import matplotlib.pyplot as plt

# 1) Pfad zur CSV anpassen (Dateiname ggf. ändern!)
CSV_PATH = r"data\AEP_hourly.csv"

# 2) CSV laden
df = pd.read_csv(CSV_PATH)

# 3) Spalten checken (damit wir wissen, wie sie heißen)
print("Spalten:", list(df.columns))

# 4) Standard für PJM: Datetime + Lastspalte
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# Bei PJME: Lastspalte heißt meistens PJME_MW
load_col = "AEP_MW"

# 5) Plot: letzte 14 Tage
end = df.index.max()
start = end - pd.Timedelta(days=14)
last = df.loc[start:end, load_col]

plt.figure()
plt.plot(last.index, last.values)
plt.title("Electric Load (last 14 days)")
plt.xlabel("Time")
plt.ylabel("MW")
plt.tight_layout()
plt.show()