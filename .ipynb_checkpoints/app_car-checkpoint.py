#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from feature_engine.encoding import OrdinalEncoder
from google.cloud import bigquery
import os

# Load model
model = joblib.load('CAR_PRICE_PREDICTION_KUBERNETES.pkl')

# Define input class
app = FastAPI()

class MyInput(BaseModel):
    model: str
    year: int
    transmission: str
    mileage: int
    fuelType: str
    tax: int
    mpg: float
    engineSize: float

# Initialize BigQuery client
client = bigquery.Client()
dataset_id = os.getenv('car-price-prediction-427001.101231')
table_id = os.getenv('car-price-prediction-427001.101231.Car-prediction')

@app.post('/predict/')
async def predict(input: MyInput):
    data = input.dict()
    data_ = pd.DataFrame([data])

    # Preprocessing steps
    data_['age'] = 2024 - data_['year']
    
    bin_edges = [0, 1.0, 1.5, 2.0, data_['engineSize'].max()]
    bin_labels = ['small', 'medium', 'large', 'very_large']
    
    unique_bin_edges = pd.unique(bin_edges)
    if len(unique_bin_edges) != len(bin_edges):
        bin_edges = unique_bin_edges.tolist()
        bin_labels = bin_labels[:len(bin_edges) - 1]
    
    data_['engine_size_category'] = pd.cut(data_['engineSize'], bins=bin_edges, labels=bin_labels, duplicates='drop')
    data_['price_per_engine_size'] = data_['engineSize']

    def encode_categorical_variables(df):
        categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
        encoder = OrdinalEncoder(encoding_method='arbitrary', variables=categorical_columns)
        df_encoded = encoder.fit_transform(df)
        return df_encoded

    data_ = encode_categorical_variables(data_)
    data_.drop(['transmission', 'fuelType'], axis=1, inplace=True)

    prediction = model.predict(data_)[0]

    # Save to BigQuery
    table = client.get_table(f"{dataset_id}.{table_id}")
    rows_to_insert = [{**data, "prediction": prediction}]
    errors = client.insert_rows_json(table, rows_to_insert)
    if errors:
        print(f"Errors: {errors}")

    return {
        'prediction': prediction
    }











# In[ ]:




