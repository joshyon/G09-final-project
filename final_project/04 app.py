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

filtered_weather_data = weather_data[['weekday_indicator', 'temp', 'pop', 'time']]


display(filtered_trip_data)
display(filtered_station_info)
display(weather_data)
display(filtered_weather_data)


# COMMAND ----------



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
    future_timestamps = [df_last["ds"].max() + datetime.timedelta(hours=i+1) for i in range(hours)]
    future_df = future_df.append(pd.DataFrame({"ds": future_timestamps}), ignore_index=True)
    return future_df

# Load the production model
GROUP_MODEL_NAME = "G09_model"
model = load_production_model()


filtered_weather_data = filtered_weather_data.rename(columns={"time": "ds"})

# Prepare future dataframe
hours_to_forecast = 4
future_df = prepare_future_df(hours_to_forecast, filtered_weather_data)
future_df = future_df.dropna()
# print(future_df)
# Make predictions
forecast = model.predict(future_df)

display(forecast)
# Plot results
fig = model.plot(forecast)
plt.show()



# COMMAND ----------

import datetime
import pytz

eastern_tz = pytz.timezone('US/Eastern')
current_time_eastern = datetime.datetime.now(eastern_tz)
formatted_datetime = current_time_eastern.strftime('%Y-%m-%d %H:%M:%S')

# Filter the forecast to keep only the data from the current time and future
forecast = forecast[forecast["ds"] > formatted_datetime]

# print(formatted_datetime)
# Print the filtered forecast
print(forecast)



# COMMAND ----------

# Get the last 5 rows of the forecast DataFrame (including the current hour and the next 4 hours)
forecast_tail = forecast.head(5)

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the line
ax.plot(forecast_tail["ds"], forecast_tail["yhat"], marker='o', linestyle='-', label='Predicted')

# Set the x-axis limits to show only the 4 hours in the future
ax.set_xlim(forecast_tail["ds"].min(), forecast_tail["ds"].max())

# Customize the plot
ax.set_xlabel("Time")
ax.set_ylabel("Bike Sharing Demand")
ax.set_title("Bike Sharing Demand Forecast for the Next 4 Hours")
ax.legend()

# Show the plot
plt.show()


# COMMAND ----------

# %pip install folium
import folium

latitude, longitude = 40.7128, -74.0060  # New York City coordinates
map = folium.Map(location=[latitude, longitude], zoom_start=12)
marker_latitude, marker_longitude = 40.74322681432173, -73.97449783980846 
folium.Marker([marker_latitude, marker_longitude], popup='Example Location').add_to(map)
display(map)



# COMMAND ----------




# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


