import pandas as pd

CSV_PATH = r"data\AEP_hourly.csv"
TARGET = "AEP_MW"

# 1) Daten laden
df = pd.read_csv(CSV_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime").sort_index()

# 2) Zeit-Features
feat = pd.DataFrame(index=df.index)
feat["y"] = df[TARGET]

feat["hour"] = feat.index.hour
feat["dayofweek"] = feat.index.dayofweek
feat["month"] = feat.index.month
feat["is_weekend"] = (feat["dayofweek"] >= 5).astype(int)

# 3) Lag-Features (wichtig für Last)
feat["lag_1"] = feat["y"].shift(1)       # 1 Stunde vorher
feat["lag_24"] = feat["y"].shift(24)     # gleicher Zeitpunkt gestern
feat["lag_168"] = feat["y"].shift(168)   # gleicher Zeitpunkt letzte Woche

# 4) Rolling means (nur Vergangenheit!)
feat["roll_24_mean"] = feat["y"].shift(1).rolling(24).mean()
feat["roll_168_mean"] = feat["y"].shift(1).rolling(168).mean()

# 5) NA entfernen (entsteht durch shifts/rolling)
feat = feat.dropna()

print("Feature-Tabelle:")
print("Zeilen:", len(feat))
print("Spalten:", list(feat.columns))
print("\nBeispiel (erste 3 Zeilen):")
print(feat.head(3))

# 6) Speichern für später
feat.to_csv(r"data\features_aep.csv")
print("\nGespeichert: data\\features_aep.csv")