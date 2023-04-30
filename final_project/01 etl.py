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

# MAGIC %md
# MAGIC The cell below defines the readStream for the historic bike trip data

# COMMAND ----------

#This cell defines the readStreaming for the historic_bike_data
historic_bike_df = (spark.readStream
 .csv(BIKE_TRIP_DATA_PATH, header="true", schema= 
     ("""ride_id string,
  rideable_type string,
  started_at timestamp,
  ended_at timestamp,
  start_station_name string,
  start_station_id string,
  end_station_name string,
  end_station_id string,
  start_lat double,
  start_lng double,
  end_lat double,
  end_lng double,
  member_casual string""")))


# COMMAND ----------

# MAGIC %md
# MAGIC This cell defines the writeStream for the historic bike trip data, creating a bronze delta table in the GROUP_DATA_PATH for the historic bike trip data.

# COMMAND ----------

#This cell completes the writeStream for the historic_bike_data
(historic_bike_df.writeStream
 .option("checkpointLocation", f"{GROUP_DATA_PATH}/bronze/historic_bike/checkpoints")
 .option("mergeSchema", "true")
 .outputMode("append")
 .trigger(availableNow=True)
 .format("delta")
 .start(f"{GROUP_DATA_PATH}/bronze_historic_bike_trip.delta")
)

#change the toTable command to .save(GROUP_DATA_PATH + delta_table_name.delta)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS bronze_historic_bike_trip
# MAGIC   USING delta
# MAGIC     LOCATION 'dbfs:/FileStore/tables/G09/bronze_historic_bike_trip.delta/'

# COMMAND ----------

# MAGIC %md
# MAGIC The cell below defines the readStream for the historic weather data

# COMMAND ----------

#This is the readStream for the historic_weather_data
historic_weather_df = (spark.readStream
 .csv(NYC_WEATHER_FILE_PATH, header="true", schema= 
     ("""dt integer,
  temp double,
  feels_like double,
  pressure integer,
  humidity integer,
  dew_point double,
  uvi double,
  clouds integer,
  visibility integer,
  wind_speed double,
  wind_deg integer,
  pop double,
  snow_1h double,
  id integer,
  main string,
  description string,
  icon string,
  loc string,
  lat double,
  lon double,
  timezone string,
  timezone_offset integer,
  rain_1h double""")))

# COMMAND ----------

# MAGIC %md
# MAGIC This cell defines the writeStream for the historic weather data, creating a bronze delta table in the GROUP_DATA_PATH for the historic weather data.

# COMMAND ----------

#This is the writeStream for the historic_weather_data
(historic_weather_df.writeStream
 .option("checkpointLocation", f"{GROUP_DATA_PATH}/bronze/historic_weather/checkpoints")
 .option("mergeSchema", "true")
 .outputMode("append")
 .trigger(availableNow=True)
 .format("delta")
 .start(f"{GROUP_DATA_PATH}/bronze_historic_weather.delta")
)



# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS bronze_historic_weather_data
# MAGIC  USING delta
# MAGIC     LOCATION 'dbfs:/FileStore/tables/G09/bronze_historic_weather.delta/'

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/G09

# COMMAND ----------

from pyspark.sql.functions import col, window, count, when, sum
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import from_unixtime

weather_data = spark.read.load("dbfs:/FileStore/tables/G09/bronze_historic_weather.delta")

weather_data = weather_data.withColumn('date_timestamp', from_unixtime('dt'))
# weather_data.show()
bike_trip_info = spark.read.load("dbfs:/FileStore/tables/G09/bronze_historic_bike_trip.delta").filter((col("start_station_name") == "E 33 St & 1 Ave") | (col("end_station_name") == "E 33 St & 1 Ave")).orderBy(col("started_at").asc())
# bike_trip_info.show(n=1000)

# COMMAND ----------

hourly_trip_data = bike_trip_info.groupBy(window(col("started_at"), "1 hour"), col("start_station_name") == "E 33 St & 1 Ave").agg(count("*").alias("count")).orderBy(col("window").asc())
hourly_trip_data = hourly_trip_data.withColumnRenamed("(start_station_name = E 33 St & 1 Ave)", "indicator")


success_count = sum(when(col("indicator") == True, col("count")))

# Define a conditional expression to count error statuses
error_count = sum(when(col("indicator") == False, col("count")))

# Group by hour and apply the conditional expressions
hourly_trip_data = hourly_trip_data.groupBy("window").agg(success_count.alias("true"), error_count.alias("false")).orderBy(col("window").asc()).na.fill(0)


hourly_trip_data = hourly_trip_data.withColumnRenamed("window", "window").withColumnRenamed("false", "bikes_returning").withColumnRenamed("true", "bikes_leaving")
hourly_trip_data = hourly_trip_data.withColumn("bikes_net_change", hourly_trip_data["bikes_returning"]-hourly_trip_data["bikes_leaving"])
hourly_trip_data = hourly_trip_data.withColumn("date_timestamp", hourly_trip_data["window"]["start"])
hourly_trip_data = hourly_trip_data.withColumn("weekday_indicator", (dayofweek(hourly_trip_data["date_timestamp"]) >= 2) & (dayofweek(hourly_trip_data["date_timestamp"]) <= 6))
hourly_trip_data = hourly_trip_data.drop("window")

# merge with weather dataframe
join_df = weather_data.select("date_timestamp", "temp", "snow_1h")
hourly_trip_data = hourly_trip_data.join(join_df, "date_timestamp", "left").orderBy(col("date_timestamp").asc())

hourly_trip_data.show(n=5000)


# COMMAND ----------

hourly_trip_data.write.mode("overwrite").option("header", "true").csv("dbfs:/FileStore/tables/G09/hourly_trip_data")


# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
