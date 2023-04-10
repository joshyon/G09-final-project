# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
#print("YOUR CODE HERE...")

# COMMAND ----------

spark.sql("set spark.sql.streaming.schemaInference=true")
from pyspark.sql.functions import col
#change to streaming after figuring out how to process the data
historic_bike_df = spark.read.csv(BIKE_TRIP_DATA_PATH, header="true", inferSchema="true")

# COMMAND ----------

our_station_historic_df = historic_bike_df.select('*').filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT) | (col("end_station_name") == GROUP_STATION_ASSIGNMENT))

our_station_historic_df.createOrReplaceTempView("historic_bike_temp_view")

#batch data about bike trips where our station is at the start or end

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM historic_bike_temp_view

# COMMAND ----------

historic_weather_df = spark.read.csv(NYC_WEATHER_FILE_PATH, header="true", inferSchema="true")

# COMMAND ----------

historic_weather_df.createOrReplaceTempView("historic_weather_temp_view")

#batch data from historic weather

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM historic_weather_temp_view

# COMMAND ----------

spark.readStream.format("delta").load(BRONZE_STATION_INFO_PATH).createOrReplaceTempView("bronze_station_info_tmp_vw")

#streaming data from info about stations

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze_station_info_tmp_vw

# COMMAND ----------

spark.readStream.format("delta").load(BRONZE_STATION_STATUS_PATH).createOrReplaceTempView("bronze_station_status_tmp_vw")

#streaming station data

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze_station_status_tmp_vw

# COMMAND ----------

spark.readStream.format("delta").load(BRONZE_NYC_WEATHER_PATH).createOrReplaceTempView("bronze_nyc_weather_temp_vw")

#streaming data for weather

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze_nyc_weather_temp_vw

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
