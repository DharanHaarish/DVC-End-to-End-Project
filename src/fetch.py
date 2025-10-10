import pandas as pd
from google.cloud import bigquery

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), "dvc_e2e.json")

bq_client = bigquery.Client()

query = """
select *
from `dvc_data.raw_data_noisy`
"""
query_job = bq_client.query(query)
df = query_job.to_dataframe()

if not os.path.exists("data/raw"):
    os.makedirs("data/raw")
    df.to_csv("data/raw/raw_data.csv", index=False)
