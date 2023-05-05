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

# Read bronze streaming delta table
trip_data = spark.read.format("delta").load("dbfs:/FileStore/tables/bronze_station_status.delta").toPandas()
station_info = spark.read.format("delta").load("dbfs:/FileStore/tables/bronze_station_info.delta").toPandas()
weather_data = spark.read.format("delta").load("dbfs:/FileStore/tables/bronze_nyc_weather.delta").toPandas()

# COMMAND ----------

display(trip_data)
display(station_info)
display(weather_data)

# baseline_model.add_regressor("weekday_indicator")
#         baseline_model.add_regressor("temp")
#         baseline_model.add_regressor("pop")
#         baseline_model.add_regressor("snow_1h")

# COMMAND ----------

import datetime
# filtering and cleaning the streaming tables
filtered_trip_data = trip_data[trip_data['station_id'] == "61c82689-3f4c-495d-8f44-e71de8f04088"]
filtered_trip_data['last_reported'] = pd.to_datetime(filtered_trip_data['last_reported'], unit='s')

filtered_station_info = station_info[station_info['external_id'] == "61c82689-3f4c-495d-8f44-e71de8f04088"]

# Add weekday_indicator
weather_data['time'] = pd.to_datetime(weather_data['time'])
def is_weekday(day):
    return day.weekday() < 5
weather_data['weekday_indicator'] = weather_data['time'].apply(is_weekday)

filtered_weather_data = weather_data[['weekday_indicator', 'temp', 'pop', 'rain.1h', 'time']]


display(filtered_trip_data)
display(filtered_station_info)
display(weather_data)
display(filtered_weather_data)


# COMMAND ----------

# Load the production model using MLflow
def load_production_model():
    client = mlflow.tracking.MlflowClient()
    production_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Production"])[0]
    model_uri = f"models:/{GROUP_MODEL_NAME}/{production_model_version.version}"
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
# hours_to_forecast = 4

future = model.make_future_dataframe(periods=4, freq='H')
fcst = model.predict(future)
fig = model.plot(fcst)

# COMMAND ----------

import datetime
import pytz
import pandas as pd
import mlflow
import mlflow.prophet
from prophet import Prophet

# Load the production model using MLflow
def load_production_model():
    client = mlflow.tracking.MlflowClient()
    production_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Production"])[0]
    model_uri = f"models:/{GROUP_MODEL_NAME}/{production_model_version.version}"
    return mlflow.prophet.load_model(model_uri)

def prepare_future_df(hours, df_last):
    future_df = df_last.copy()
    future_df = future_df.drop(columns=["y"])
    future_timestamps = [df_last["timestamp"].max() + datetime.timedelta(hours=i+1) for i in range(hours)]
    future_df = future_df.append(pd.DataFrame({"timestamp": future_timestamps}), ignore_index=True)
    return future_df

# Load the production model
GROUP_MODEL_NAME = "G09_model"
model = load_production_model()

# Your DataFrame with columns "timestamp", "weekday_indicator", "pop", "rain_1h", "temp"
# Replace 'your_dataframe' with the actual DataFrame you have
filtered_weather_data = filtered_weather_data.rename(columns={"time": "ds", "": "y"})

# Prepare future dataframe
hours_to_forecast = 4
future_df = prepare_future_df(hours_to_forecast, your_dataframe.tail(1))

# Make predictions
forecast = model.predict(future_df)

# Plot results
fig = model.plot(forecast)



# COMMAND ----------




# COMMAND ----------




# COMMAND ----------




# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


