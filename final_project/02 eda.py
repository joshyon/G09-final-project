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
print("YOUR CODE HERE...")

# COMMAND ----------

# MAGIC %sql
# MAGIC USE G09_db;
# MAGIC SHOW Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze_historic_bike_trip;

# COMMAND ----------

# MAGIC %md
# MAGIC # Trip Data

# COMMAND ----------

# MAGIC %python
# MAGIC import holidays

# COMMAND ----------

print(holidays.USA())

# COMMAND ----------

display(dbutils.fs.ls(BIKE_TRIP_DATA_PATH))

# COMMAND ----------

trip_files = dbutils.fs.ls(BIKE_TRIP_DATA_PATH)
for file in trip_files:
  print(file.path)

# COMMAND ----------

trip_files[0]

# COMMAND ----------

temp_trip_df = spark.read.csv(trip_files[0].path, header=True, inferSchema=True)

# COMMAND ----------

print(temp_trip_df.head(5))

# COMMAND ----------

# Read CSV files as DataFrames
# dfs = []
# for file in trip_files:
#     if file.name.endswith(".csv"):
#         df = spark.read.csv(file.path, header=True, inferSchema=True)
#         dfs.append(df)

# COMMAND ----------

# merging DataFrames with Spark
# trip_df = dfs[0]
# for i in range(1, len(dfs)):
#     trip_df = trip_df.union(dfs[i])

# COMMAND ----------

trip_df=spark.read.format('csv').option("header","True").option("inferSchema","True").load(BIKE_TRIP_DATA_PATH+'202111_citibike_tripdata.csv')
display(trip_df)

# COMMAND ----------

type(trip_df)

# COMMAND ----------

# creating a table from merged DataFrame with Spark
trip_df.createOrReplaceTempView("trips")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM trips
# MAGIC ORDER BY started_at DESC
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC USE G09_db;
# MAGIC CREATE TABLE IF NOT EXISTS station_trips(
# MAGIC   SELECT * FROM trips
# MAGIC   WHERE start_station_name = 'E 33 St & 1 Ave' or end_station_name = 'E 33 St & 1 Ave'
# MAGIC )

# COMMAND ----------

trip_trend_df = spark.sql("SELECT concat(year,'-',month,'-',day) as date, concat(year,'-',month) as month_year, ride_id FROM(SELECT YEAR(started_at) AS year, MONTH(started_at) AS month, DAY(started_at) AS day, ride_id FROM bronze_historic_bike_trip) ORDER BY date")
display(trip_trend_df, truncate=False)

# COMMAND ----------

# MAGIC %python
# MAGIC import matplotlib 
# MAGIC month_year_df = trip_trend_df.groupBy('month_year').count()

# COMMAND ----------



# COMMAND ----------

display(spark.read.format('delta').load(BRONZE_STATION_INFO_PATH))

# COMMAND ----------

# E 33 St & 1 Ave
station_info = spark.read.format('delta').load(BRONZE_STATION_INFO_PATH)
display(station_info.filter(station_info["name"] == "E 33 St & 1 Ave"))

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


