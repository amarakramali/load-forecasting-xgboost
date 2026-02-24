\# AEP Load Forecasting (XGBoost)



Kurzprojekt zur stündlichen Lastprognose (AEP) mit Feature Engineering und XGBoost.



\## Highlights

\- Zeitfeatures (hour, dayofweek, month, weekend)

\- Lag-Features (t-1h, t-24h, t-168h) + Rolling Means

\- Vergleich gegen Baseline (Blend aus gestern \& letzter Woche)

\- Export einer 24h-Vorhersage als CSV + Plot



\## Ergebnis (Test: letzte 30 Tage)

\- Baseline (Blend 50/50): MAE 921.10, RMSE 1215.43

\- XGBoost: MAE 142.44, RMSE 184.41  (\*\*~84.5% MAE Improvement\*\*)



\## Projektstruktur

\- `data/` Rohdaten + Feature-Tabelle (lokal)

\- `src/` Skripte (Plot, Feature Engineering, Baseline, XGBoost, 24h Forecast)

\- `reports/` Exporte (CSV) und Plots (`reports/figures/`)

\- `models/` gespeichertes Modell (`.joblib`)



\## Setup (Windows, PowerShell)

```powershell

pip install pandas numpy matplotlib scikit-learn xgboost joblib

