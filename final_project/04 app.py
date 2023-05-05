# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# DBTITLE 0,YOUR APPLICATIONS CODE HERE...
import json
import pandas as pd

import mlflow
import mlflow.pyfunc
import datetime
import pytz
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.serialize import model_from_json


# Load the production model using MLflow
def load_production_model():
    client = mlflow.tracking.MlflowClient()
    production_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Production"])[0]
    model_uri = production_model_version.source
    return mlflow.prophet.load_model(model_uri)

def prepare_future_df(hours, df_last):
    future_df = df_last.copy()
    future_df = future_df.drop(columns=["y"])
    future_timestamps = [df_last["ds"].max() + datetime.timedelta(hours=i+1) for i in range(hours)]
    future_df = future_df.append(pd.DataFrame({"ds": future_timestamps}), ignore_index=True)
    return future_df

# Load the production model
GROUP_MODEL_NAME = "G09_model"
model = load_production_model()

# Get the current timestamp
now = datetime.datetime.now(pytz.utc)

# Calculate hours to forecast
hours_to_forecast = 4

# Read silver table as input data
trip_data = spark.read.format("delta").load("dbfs:/FileStore/tables/bronze_station_status.delta").toPandas()
filtered_dataframe = trip_data[trip_data['station_id'] == "61c82689-3f4c-495d-8f44-e71de8f04088"]

filtered_dataframe['last_reported'] = pd.to_datetime(filtered_dataframe['last_reported'], unit='s')

display(filtered_dataframe)





# COMMAND ----------

m = Prophet(changepoint_prior_scale=0.01).fit(trip_data)
future = m.make_future_dataframe(periods=4, freq='H')
fcst = m.predict(future)
fig = m.plot(fcst)

# COMMAND ----------

# Prepare future dataframe
future_df = prepare_future_df(hours_to_forecast, trip_data.tail(1))
future_df.fillna(True, inplace=True)
display(future_df)
print(future_df.isna().sum())


# COMMAND ----------

# Make predictions
forecast = model.predict(future_df)

# Display results
st.subheader("Forecasted bike sharing demand")
st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(hours_to_forecast))



# COMMAND ----------

# Plot results
fig, ax = plt.subplots()
ax.plot(forecast['ds'], forecast['yhat'], label='yhat')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', label='Confidence Interval')
ax.set_xlabel('Date')
ax.set_ylabel('Bike Sharing Demand')
ax.legend()
st.subheader("Bike sharing demand forecast plot")
st.pyplot(fig)


# COMMAND ----------

pip install streamlit


# COMMAND ----------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import mlflow.pyfunc
import json

st.set_page_config(page_title='Bike Share Forecasting', layout='wide')
st.title('Bike Share Forecasting')

# Load the MLflow model
model_path = dbutils.widgets.get("Staging uri")  # or "Production uri" if you prefer
model = mlflow.pyfunc.load_model(model_path)

# Input parameters
st.sidebar.header('Input Parameters')
start_date = st.sidebar.date_input('Start date for the forecast', value=pd.to_datetime('2023-03-01'))
end_date = st.sidebar.date_input('End date for the forecast', value=pd.to_datetime('2023-03-10'))
forecast_hours = st.sidebar.slider('Forecast hours', min_value=1, max_value=48, value=4, step=1)

# Load data for additional regressors
additional_regressors = "path/to/your/additional_regressors.csv"  # Replace with the path to your additional_regressors file
df_regressors = pd.read_csv(additional_regressors)
df_regressors['ds'] = pd.to_datetime(df_regressors['ds'])

# Prepare data for forecasting
future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='H'), columns=['ds'])
future = future.merge(df_regressors, on='ds', how='left')

# Forecast
forecast = model.predict(future)

# Visualize the forecast
st.header('Forecasted Bike Share Demand')
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Display forecast data
st.header('Forecast Data')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])


# COMMAND ----------

import mlflow 
import mlflow.pyfunc
from pyspark.sql.functions import col 
from pyspark.sql.types import StringType
# Get latest model
latest_model_detail = mlflow.tracking.MlflowClient().get_latest_versions("G09_model", stages= ['Production'])[0]
model_udf = mlflow.pyfunc.spark_udf(spark, latest_model_detail.source)

# COMMAND ----------


# Python

