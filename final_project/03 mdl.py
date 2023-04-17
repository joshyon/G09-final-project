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

# read from bronze bike trip dataset
trip_df = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/bronze_historic_bike_trip/")
display(trip_df)

# COMMAND ----------

# read from bronze weather dataset
weather_df = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/bronze_historic_weather_data/")
display(weather_df)

# COMMAND ----------

# read in station info
df = spark.read.format("delta").load(BRONZE_STATION_INFO_PATH)
df = df.filter(df.name=="E 33 St & 1 Ave")
display(df)

# COMMAND ----------

df = spark.read.format("delta").load(BRONZE_STATION_STATUS_PATH)
df = df.filter(df.station_id=="61c82689-3f4c-495d-8f44-e71de8f04088")
display(df)

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
