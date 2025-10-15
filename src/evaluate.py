import pickle
import pandas as pd
from prophet import Prophet
import os
import logging
import json
import numpy as np
import yaml

if not os.path.exists("models/"):
    logging.error("models directory not found")
    raise FileNotFoundError("models directory not found")

with open("models/prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

if not os.path.exists("data/processed"):
    logging.error("processed data directory not found")
    raise FileNotFoundError("processed data directory not found")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

df_test = pd.read_csv("data/processed/df_test.csv")
df_test['date'] = pd.to_datetime(df_test['date'])
df_test = df_test.sort_values(by="date", ascending=True).reset_index(drop=True)
df_test = df_test.rename(columns={"date": "ds"})

future = pd.DataFrame({"ds": pd.date_range(start=df_test['ds'].min(), end=df_test['ds'].max(), freq='D')})
forecast = model.predict(future)

forecast = forecast[['ds', 'yhat']]
forecast = forecast.merge(df_test, on="ds", how="inner")

forecast['yhat'] = forecast['yhat'].astype(float)
forecast['value'] = forecast['value'].astype(float)

if params['evaluate']['transform'].lower() == "log":
    if params['evaluate']['base'] == 10:
        forecast['yhat'] = np.power(10, forecast['yhat'])
    elif params['evaluate']['base'] == "e":
        forecast['yhat'] = np.exp(forecast['yhat'])
    elif params['evaluate']['base'] == 2:
        forecast['yhat'] = np.exp2(forecast['yhat'])
    
else:
    pass

rmse = np.sqrt(np.mean((forecast['yhat'] - forecast['value'])**2))
mae = np.mean(np.abs(forecast['yhat'] - forecast['value']))
mape = np.mean(np.abs(forecast['yhat'] - forecast['value']) / forecast['value'])
mpe = np.mean((forecast['yhat'] - forecast['value']) / forecast['value'])

metric_dict = {
    "RMSE": rmse,
    "MAE": mae,
    "MAPE": mape,
    "MPE": mpe
}

with open("metrics.json", "w") as f:
    json.dump(metric_dict, f)

print("\n" + "="*60)
print("📊 MODEL EVALUATION METRICS")
print("="*60)
for metric_name, metric_value in metric_dict.items():
    print(f"{metric_name:10s}: {metric_value:.6f}")
print("="*60)
print("\nMetrics saved to metrics.json\n")