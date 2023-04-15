# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS ETL
# MAGIC LOCATION 'dbfs:/FileStore/tables/G09/'

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/G09

# COMMAND ----------

# MAGIC %sql
# MAGIC USE ETL

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS bronze_historic_bike_trip (
# MAGIC   ride_id string,
# MAGIC   rideable_type string,
# MAGIC   started_at timestamp,
# MAGIC   ended_at timestamp,
# MAGIC   start_station_name string,
# MAGIC   start_station_id string,
# MAGIC   end_station_name string,
# MAGIC   end_station_id string,
# MAGIC   start_lat double,
# MAGIC   start_lng double,
# MAGIC   end_lat double,
# MAGIC   end_lng double,
# MAGIC   member_casual string
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN ETL

# COMMAND ----------

start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
#print("YOUR CODE HERE...")

# COMMAND ----------

#This cell definites the readStreaming for the historic_bike_data
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
 .toTable("bronze_historic_bike_trip")
)

# COMMAND ----------

#our_station_historic_df = historic_bike_df.select('*').filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT) | (col("end_station_name") == GROUP_STATION_ASSIGNMENT))

#our_station_historic_df.createOrReplaceTempView("historic_bike_temp_view")

#batch data about bike trips where our station is at the start or end

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS bronze_historic_weather_data (
# MAGIC   dt integer,
# MAGIC   temp double,
# MAGIC   feels_like double,
# MAGIC   pressure integer,
# MAGIC   humidity integer,
# MAGIC   dew_point double,
# MAGIC   uvi double,
# MAGIC   clouds integer,
# MAGIC   visibility integer,
# MAGIC   wind_speed double,
# MAGIC   wind_deg integer,
# MAGIC   pop double,
# MAGIC   snow_1h double,
# MAGIC   id integer,
# MAGIC   main string,
# MAGIC   description string,
# MAGIC   icon string,
# MAGIC   loc string,
# MAGIC   lat double,
# MAGIC   lon double,
# MAGIC   timezone string
# MAGIC )

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
 .option("checkpointLocation", "f{GROUP_DATA_PATH}/bronze/historic_weather_checkpoint")
 .outputMode("append")
 .trigger(availableNow=True)
 .toTable("bronze_historic_weather_data")
)



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

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
