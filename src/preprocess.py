import os
from google.cloud import bigquery,storage
import pandas as pd
import logging
import numpy as np

if not os.path.exists("data/raw"):
    logging.error("raw data directory not found")
    raise FileNotFoundError("raw data directory not found")

df = pd.read_csv("data/raw/raw_data.csv")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
df_train = df[:int(len(df)*0.8)].reset_index(drop=True)
df_test = df[int(len(df)*0.8):].reset_index(drop=True)

# treat outliers by capping them at the lower and upper bounds and to prevent data leakage in the test set we are not using the test set to calculate the bounds
iqr = df_train['value'].quantile(0.75) - df_train['value'].quantile(0.25)
lower_bound = df_train['value'].quantile(0.25) - 1.5 * iqr
upper_bound = df_train['value'].quantile(0.75) + 1.5 * iqr

df_train['value'] = np.where(df_train['value'] < lower_bound, lower_bound, df_train['value'])
df_train['value'] = np.where(df_train['value'] > upper_bound, upper_bound, df_train['value'])

df_test['value'] = np.where(df_test['value'] < lower_bound, lower_bound, df_test['value'])
df_test['value'] = np.where(df_test['value'] > upper_bound, upper_bound, df_test['value'])

if not os.path.exists("data/processed"):
    os.makedirs("data/processed")
    df_train.to_csv("data/processed/df_train.csv", index=False)
    df_test.to_csv("data/processed/df_test.csv", index=False)
    logging.info("processed data saved")
else:
    df_train.to_csv("data/processed/df_train.csv", index=False)
    df_test.to_csv("data/processed/df_test.csv", index=False)
    logging.info("processed data saved")

