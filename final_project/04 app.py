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

# hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))

# Read bronze streaming delta table
trip_data = spark.read.format("delta").load("dbfs:/FileStore/tables/bronze_station_status.delta").toPandas()
station_info = spark.read.format("delta").load("dbfs:/FileStore/tables/bronze_station_info.delta").toPandas()
weather_data = spark.read.format("delta").load("dbfs:/FileStore/tables/bronze_nyc_weather.delta").toPandas()

# COMMAND ----------



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


# display(filtered_trip_data)
# display(filtered_station_info)
# display(weather_data)
# display(filtered_weather_data)


# COMMAND ----------

# Load streaming historical data (net change)

bronze_station_status_oneday = spark.read.format("delta").load("dbfs:/user/hive/warehouse/g09_db.db/bronze_station_status_oneday")

# Convert the Spark DataFrame to a Pandas DataFrame (optional)
bronze_station_status_oneday_pd = bronze_station_status_oneday.toPandas()
bronze_station_status_oneday_pd = bronze_station_status_oneday_pd.dropna()


# Display the DataFrame
display(bronze_station_status_oneday_pd)

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
hours_to_forecast = 1000
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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#plot residual
actual_values = bronze_station_status_oneday_pd[['rounded_last_reported', 'Net_Change']].set_index('rounded_last_reported')
predicted_values = forecast[['ds', 'yhat']].set_index('ds')

# Filter the actual_values and predicted_values to include only the common dates
common_dates = actual_values.index.intersection(predicted_values.index)
actual_values = actual_values.loc[common_dates]
predicted_values = predicted_values.loc[common_dates]

# Calculate residuals
residuals = actual_values['Net_Change'] - predicted_values['yhat']

# Create residual plot
plt.figure(figsize=(10, 6))
plt.scatter(predicted_values, residuals, alpha=0.5)

# Fit a linear regression model to the residuals
lr = LinearRegression()
lr.fit(predicted_values.values.reshape(-1, 1), residuals.values.reshape(-1, 1))
predicted_residuals = lr.predict(predicted_values.values.reshape(-1, 1))

# Add a line representing the trend of the residuals
plt.plot(predicted_values, predicted_residuals, color='r', linestyle='--', alpha=0.7, label='Trend Line')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Production Model')
plt.legend()

plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

residuals = actual_values['Net_Change'] - predicted_values['yhat']
# Create violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=residuals)
plt.xlabel('Residuals')
plt.title('Residuals for Production Model')

plt.show()




# COMMAND ----------

# MAGIC %md #Display Timestamp of When You Run the Code (Current Time)

# COMMAND ----------

import datetime
import pytz

eastern_tz = pytz.timezone('US/Eastern')
current_time_eastern = datetime.datetime.now(eastern_tz)
formatted_datetime = current_time_eastern.strftime('%Y-%m-%d %H:%M:%S')

# Filter the forecast to keep only the data from the current time and future
forecast = forecast[forecast["ds"] > formatted_datetime]

print(formatted_datetime)
# Print the filtered forecast
# print(forecast)

# COMMAND ----------

# MAGIC %md #Display the Current Staging and Production Model Version

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest production model version
production_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Production"])[0]

# Get the latest staging model version
staging_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Staging"])[0]

# Display the current production and staging model versions
print(f"Current Production Model: {production_model_version.name}, Version: {production_model_version.version}")
print(f"Current Staging Model: {staging_model_version.name}, Version: {staging_model_version.version}")


# COMMAND ----------

# MAGIC %md #Run the pip instal folium command if folium is not installed. (Simply Uncomment the next cell)

# COMMAND ----------

# %pip install folium

# COMMAND ----------

# MAGIC %md #Display Station Name and Map Location (Click on the Marker to See Station Name)

# COMMAND ----------

import folium

latitude, longitude = 40.7128, -74.0060  # New York City coordinates
map = folium.Map(location=[latitude, longitude], zoom_start=12)
marker_latitude, marker_longitude = 40.74322681432173, -73.97449783980846 
folium.Marker([marker_latitude, marker_longitude], popup='E 33 St & 1 Ave').add_to(map)
display(map)

# COMMAND ----------

# MAGIC %md #Display Current Weather

# COMMAND ----------

import pandas as pd

current_weather = weather_data[weather_data["time"] > formatted_datetime]
display(current_weather.head(1))

# COMMAND ----------

# Get the last 5 rows of the forecast DataFrame (including the current hour and the next 4 hours)
forecast_tail = forecast.head(5)

# COMMAND ----------

# MAGIC %md #Total Docks at Our Station is Represented by the Red Horizontal Line (y=83, 83 Docks)

# COMMAND ----------

# MAGIC %md #Total Bikes Available is Represented by the Blue Line

# COMMAND ----------

# MAGIC %md #If the Blue Line Drops Below 0 or Passed the Red Line, It Means Our Station is Out of Bike or Have Too Many Bikes

# COMMAND ----------

import pandas as pd

# Sort filtered_trip_data by 'last_reported' column
filtered_trip_data_sorted = filtered_trip_data.sort_values(by='last_reported')

# Merge filtered_trip_data_sorted with forecast_tail to find the closest match
merged_data = pd.merge_asof(
    forecast_tail, 
    filtered_trip_data_sorted[['last_reported', 'num_docks_available']], 
    left_on='ds', 
    right_on='last_reported', 
    direction='nearest'
)

# Subtract the net bike change from the num_docks_available value
merged_data["adjusted_demand"] = merged_data["num_docks_available"] - merged_data["yhat"]

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the line
ax.plot(merged_data["ds"], merged_data["adjusted_demand"], marker='o', linestyle='-', label='Adjusted Prediction')

# Set the x-axis limits to show only the 4 hours in the future
ax.set_xlim(merged_data["ds"].min(), merged_data["ds"].max())

# Customize the plot
ax.set_xlabel("Time")
ax.set_ylabel("Bikes Available")
ax.set_title("Station Forecast")
ax.legend()

# Add a horizontal line at y=83 and label it
station_capacity = 83
ax.axhline(y=station_capacity, color='r', linestyle='--', alpha=0.7, label='Station Capacity Line')

# Show the plot
plt.show()



# COMMAND ----------

# MAGIC %md #Repeat Everything for Staging Model

# COMMAND ----------

import datetime
import pytz
import pandas as pd
import mlflow
import mlflow.prophet
from prophet import Prophet

# Load the production model using MLflow
def load_staging_model():
    client = mlflow.tracking.MlflowClient()
    staging_model_version = client.get_latest_versions(GROUP_MODEL_NAME, stages=["Staging"])[0]
    model_uri = f"models:/{GROUP_MODEL_NAME}/{staging_model_version.version}"
    return mlflow.prophet.load_model(model_uri)


def prepare_future_df(hours, df_last):
    future_df = df_last.copy()
    future_timestamps = [df_last["ds"].max() + datetime.timedelta(hours=i+1) for i in range(hours)]
    future_df = future_df.append(pd.DataFrame({"ds": future_timestamps}), ignore_index=True)
    return future_df

# Load the production model
GROUP_MODEL_NAME = "G09_model"
model = load_staging_model()


filtered_weather_data = filtered_weather_data.rename(columns={"time": "ds"})

# Prepare future dataframe
hours_to_forecast = 1000
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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#plot residual
actual_values = bronze_station_status_oneday_pd[['rounded_last_reported', 'Net_Change']].set_index('rounded_last_reported')
predicted_values = forecast[['ds', 'yhat']].set_index('ds')

# Filter the actual_values and predicted_values to include only the common dates
common_dates = actual_values.index.intersection(predicted_values.index)
actual_values = actual_values.loc[common_dates]
predicted_values = predicted_values.loc[common_dates]

# Calculate residuals
residuals = actual_values['Net_Change'] - predicted_values['yhat']

# Create residual plot
plt.figure(figsize=(10, 6))
plt.scatter(predicted_values, residuals, alpha=0.5)

# Fit a linear regression model to the residuals
lr = LinearRegression()
lr.fit(predicted_values.values.reshape(-1, 1), residuals.values.reshape(-1, 1))
predicted_residuals = lr.predict(predicted_values.values.reshape(-1, 1))

# Add a line representing the trend of the residuals
plt.plot(predicted_values, predicted_residuals, color='r', linestyle='--', alpha=0.7, label='Trend Line')

plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Staging Model')
plt.legend()

plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

residuals = actual_values['Net_Change'] - predicted_values['yhat']
# Create violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=residuals)
plt.xlabel('Residuals')
plt.title('Residuals for Staging Model')

plt.show()

# COMMAND ----------

# MAGIC %md #Register Current Staging Model as Production Model
# MAGIC ## Apparently, the Staging Model Have a Much Better Performance Than the Production Model. We will Register the Staging Model as the New Production Model.

# COMMAND ----------

# import mlflow
# from mlflow.tracking import MlflowClient

# mlflow_client = MlflowClient()


# # Specify the model name
# GROUP_MODEL_NAME = "G09_model"

# # Fetch the latest staging model version
# staging_model_version = mlflow_client.get_latest_versions(GROUP_MODEL_NAME, stages=["Staging"])[0]

# # Register the staging model version as the new production model version
# mlflow_client.transition_model_version_stage(
#     name=GROUP_MODEL_NAME,
#     version=staging_model_version.version,
#     stage="Production"
# )

# print(f"Staging model version {staging_model_version.version} has been promoted to production.")

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
