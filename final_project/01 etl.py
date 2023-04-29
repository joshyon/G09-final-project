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

#This cell completes the writeStream for the historic_bike_data
(historic_bike_df.writeStream
 .option("checkpointLocation", f"{GROUP_DATA_PATH}/bronze/historic_bike/checkpoints")
 .outputMode("append")
 .trigger(availableNow=True)
 .format("delta")
 .start(f"{GROUP_DATA_PATH}/bronze_historic_bike_trip.delta")
)

#change the toTable command to .save(GROUP_DATA_PATH + delta_table_name.delta)

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/G09

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS bronze_historic_bike_trip
# MAGIC   USING delta
# MAGIC     LOCATION 'dbfs:/FileStore/tables/G09/bronze_historic_bike_trip.delta/'

# COMMAND ----------

#our_station_historic_df = historic_bike_df.select('*').filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT) | (col("end_station_name") == GROUP_STATION_ASSIGNMENT))

#our_station_historic_df.createOrReplaceTempView("historic_bike_temp_view")

#batch data about bike trips where our station is at the start or end

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
  timezone string""")))

# COMMAND ----------

#This is the writeStream for the historic_weather_data
(historic_weather_df.writeStream
 .option("checkpointLocation", f"{GROUP_DATA_PATH}/bronze/historic_weather/checkpoints")
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

# MAGIC %sql
# MAGIC SELECT * FROM bronze_historic_weather_data

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/G09

# COMMAND ----------

for s in spark.streams.active:
    print("Stopping " + s.id)
    s.stop()
    s.awaitTermination()

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
