# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

dbutils.widgets.removeAll()

dbutils.widgets.text('01.start_date', "2021-10-01")
dbutils.widgets.text('02.end_date', "2023-03-01")
dbutils.widgets.text('03.hours_to_forecast', '4')
dbutils.widgets.text('04.promote_model', 'No')

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
# print("YOUR CODE HERE...")

print(start_date,end_date,hours_to_forecast, promote_model)
print("YOUR CODE HERE...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore the Bronze Dataset

# COMMAND ----------

# read from BRONZE BIKE TRIP dataset
# dbutils.fs.ls("dbfs:/FileStore/tables/G09/") # TODO: debugging line to be deleted
trip_df = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/bronze_historic_bike_trip.delta/")
display(trip_df)

# COMMAND ----------

# read from BRONZE WHEATER dataset
weather_df = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/bronze_historic_weather.delta/")
weather_df = weather_df.orderBy("dt")
display(weather_df)

# COMMAND ----------

# read in STATION INFO
station_df = spark.read.format("delta").load(BRONZE_STATION_INFO_PATH)
station_df = station_df.filter(station_df.name=="E 33 St & 1 Ave")
display(station_df)

# COMMAND ----------

# read in STATION STATUS
status_df = spark.read.format("delta").load(BRONZE_STATION_STATUS_PATH)
status_df = status_df.filter(status_df.station_id=="61c82689-3f4c-495d-8f44-e71de8f04088")
status_df = status_df.orderBy("last_reported")
display(status_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment with Fake Training Data

# COMMAND ----------

# a fake training dataset was uploaded to g09_db, the table was named as "fake_training_data"
# read in fake data
from pyspark.sql.functions import *
trip_data = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/hourly_trip_data/").toPandas()
# fake_df = fake_df.rename(columns={"Time": "ds", "Net_Change": "y"})
display(trip_data)

# COMMAND ----------

# import useful package for ML
import json
import pandas as pd
import numpy as np
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.express as px
import itertools
np.random.seed(202)

ARTIFACT_NAME = "G09_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Baseline Model

# COMMAND ----------

# create and fit the baseline model
baseline_model = Prophet()
baseline_model.add_regressor("Temperature")
baseline_model.add_regressor("Probability_of_Precipitation")
baseline_model.add_regressor("Amount_of_Snowfall")
baseline_model.add_regressor("Weekday")
baseline_model.add_regressor("Holiday")
baseline_model.fit(fake_df)

# cross validation 
baseline_model_cv = cross_validation(model=baseline_model, horizon="10 hours", parallel="threads")
display(baseline_model_cv)

# model performance
baseline_model_p = performance_metrics(baseline_model_cv, rolling_window=1)
display(baseline_model_p)

print(f"MAPE of baseline model: {baseline_model_p['mape'].values[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Tuned Model

# COMMAND ----------

param_grid = {
    'yearly_seasonality': [True],
    'weekly_seasonality': [True],
    'daily_seasonality': [True],
    'changepoint_prior_scale': [0.001],  # , 0.05, 0.08, 0.5
    'seasonality_prior_scale': [0.01],  # , 1, 5, 10, 12
    'seasonality_mode': ['additive', 'multiplicative']
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

for kwargs in all_params:
    updated_model = Prophet(**kwargs)
    updated_model.add_regressor("Temperature")
    updated_model.add_regressor("Probability_of_Precipitation")
    updated_model.add_regressor("Amount_of_Snowfall")
    updated_model.add_regressor("Weekday")
    updated_model.add_regressor("Holiday")
    updated_model.fit(fake_df)

    updated_model_cv = cross_validation(model=updated_model, horizon="10 hours", parallel="threads")

    updated_model_p = performance_metrics(updated_model_cv, rolling_window=1)




# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


