from prophet import Prophet
import pandas as pd
import logging
import numpy as np
import os
import pickle

if not os.path.exists("data/processed"):
    logging.error("processed data directory not found")
    raise FileNotFoundError("processed data directory not found")

df_train = pd.read_csv("data/processed/df_train.csv")
df_test = pd.read_csv("data/processed/df_test.csv")

df_train['date'] = pd.to_datetime(df_train['date'])
df_train = df_train.sort_values(by="date", ascending=True).reset_index(drop=True)

df_train = df_train.rename(columns={"date": "ds", "value": "y"})

model = Prophet()
model.fit(df_train)

if not os.path.exists("models/"):
    os.makedirs("models/")
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)
else:
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)

print("Model saved")


