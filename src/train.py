from prophet import Prophet
import pandas as pd
import logging
import numpy as np
import os
import pickle
import yaml

if not os.path.exists("data/processed"):
    logging.error("processed data directory not found")
    raise FileNotFoundError("processed data directory not found")

df_train = pd.read_csv("data/processed/df_train.csv")
df_test = pd.read_csv("data/processed/df_test.csv")

df_train['date'] = pd.to_datetime(df_train['date'])
df_train = df_train.sort_values(by="date", ascending=True).reset_index(drop=True)

df_train = df_train.rename(columns={"date": "ds", "value": "y"})

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

if params['train']['transform'].lower() == "log":
    logging.info("Log transformation applied")
    if params['train']['base'] == 10:
        logging.info("Log base 10 transformation applied")
        df_train['y'] = np.log10(df_train['y'])
    elif params['train']['base'] == "e":
        logging.info("Log base e transformation applied")
        df_train['y'] = np.log(df_train['y'])
else:
    logging.info("No transformation applied")
    pass

hyperparameters = params['train']['hyperparameters']
model = Prophet(**hyperparameters)
model.fit(df_train)

if not os.path.exists("models/"):
    os.makedirs("models/")
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)
else:
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)

print("Model saved")


