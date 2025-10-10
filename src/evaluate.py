import pickle
import pandas as pd
from prophet import Prophet
import os
import logging
import numpy as np

if not os.path.exists("models/"):
    logging.error("models directory not found")
    raise FileNotFoundError("models directory not found")

with open("models/prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

if not os.path.exists("data/processed"):
    logging.error("processed data directory not found")
    raise FileNotFoundError("processed data directory not found")

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

rmse = np.sqrt(np.mean((forecast['yhat'] - forecast['value'])**2))
mae = np.mean(np.abs(forecast['yhat'] - forecast['value']))

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")