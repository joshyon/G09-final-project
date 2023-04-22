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

display(dbutils.fs.ls(GROUP_DATA_PATH))

# COMMAND ----------

bike_data = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/bronze_historic_bike_trip.delta/")
bike_data.write.format("delta").mode("overwrite").saveAsTable("G09_db.bronze_historic_bike_trip")

# COMMAND ----------

weather_data = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/bronze_historic_weather.delta/")
weather_data.write.format("delta").mode("overwrite").saveAsTable("G09_db.bronze_historic_weather_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC USE G09_db;
# MAGIC SHOW Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze_historic_bike_trip WHERE start_station_name = 'E 33 St & 1 Ave' or end_station_name = 'E 33 St & 1 Ave';

# COMMAND ----------

# MAGIC %md
# MAGIC # Trip Data

# COMMAND ----------

trip_trend_df = spark.sql("""
                          SELECT concat(year,'-',month,'-',day) as date, concat(year,'-',month) as month_year, ride_id 
                          FROM(
                              SELECT YEAR(started_at) AS year, MONTH(started_at) AS month, DAY(started_at) AS day, ride_id, if(start_station_name = 'E 33 St & 1 Ave', 'out', 'in') as trip_direction
                              FROM bronze_historic_bike_trip
                              WHERE start_station_name = 'E 33 St & 1 Ave' or end_station_name = 'E 33 St & 1 Ave') 
                          ORDER BY date
                          """)

# COMMAND ----------

# MAGIC %python
# MAGIC from datetime import datetime, timedelta
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC import holidays
# MAGIC import matplotlib.dates as mdates
# MAGIC import seaborn as sns

# COMMAND ----------

# MAGIC %python
# MAGIC # Define the start and end dates as strings in the format "YYYY-MM-DD"
# MAGIC start_date = trip_trend_df.head(1)[0]['date']
# MAGIC end_date = trip_trend_df.tail(1)[0]['date']
# MAGIC 
# MAGIC # Convert the start and end dates to datetime objects
# MAGIC start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
# MAGIC end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
# MAGIC 
# MAGIC # Create a timedelta object representing one day
# MAGIC one_day = timedelta(days=1)
# MAGIC 
# MAGIC # Initialize an empty list to store the datetime objects
# MAGIC date_list = []
# MAGIC counts = []
# MAGIC # Loop through the date range and append each date to the list
# MAGIC current_date = start_date_obj
# MAGIC while current_date <= end_date_obj:
# MAGIC     date_list.append([current_date.strftime('%Y-%m-%d'),0])
# MAGIC     counts.append(0)
# MAGIC     current_date += one_day

# COMMAND ----------

# MAGIC %python
# MAGIC # temp_df = spark.createDataFrame([current_date, counts], ["date", 'trip_count'])
# MAGIC temp_df = pd.DataFrame(date_list, columns = ["date", 'trip_count'])

# COMMAND ----------

temp_df = spark.createDataFrame(temp_df)
temp_df.createOrReplaceTempView("date_table")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT date_table.date, concat(YEAR(date_table.date),'-',LPAD(MONTH(date_table.date), 2, '0')) as month_year, COALESCE(t2.trip_count, 0) AS trip_count, t2.trip_direction
# MAGIC FROM date_table
# MAGIC LEFT JOIN (
# MAGIC   SELECT date, count(ride_id) as trip_count, t1.trip_direction
# MAGIC   FROM(
# MAGIC     SELECT concat(YEAR(started_at),'-',LPAD(MONTH(started_at), 2, '0'),'-',LPAD(DAY(started_at), 2, '0')) as date, ride_id, if(start_station_name = 'E 33 St & 1 Ave', 'out', 'in') as trip_direction
# MAGIC     FROM bronze_historic_bike_trip
# MAGIC     WHERE start_station_name = 'E 33 St & 1 Ave' or end_station_name = 'E 33 St & 1 Ave'
# MAGIC   ) as t1
# MAGIC   GROUP BY date, t1.trip_direction
# MAGIC ) as t2
# MAGIC on date_table.date = t2.date
# MAGIC ORDER BY date_table.date

# COMMAND ----------

sql_command = """
SELECT date_table.date, concat(YEAR(date_table.date),'-',LPAD(MONTH(date_table.date), 2, '0')) as month_year, WEEKDAY(date_table.date) as weekday, COALESCE(t2.trip_count, 0) AS trip_count, t2.trip_direction
FROM date_table
LEFT JOIN (
  SELECT date, count(ride_id) as trip_count, t1.trip_direction
  FROM(
    SELECT concat(YEAR(started_at),'-',LPAD(MONTH(started_at), 2, '0'),'-',LPAD(DAY(started_at), 2, '0')) as date, ride_id, if(start_station_name = 'E 33 St & 1 Ave', 'out', 'in') as trip_direction
    FROM bronze_historic_bike_trip
    WHERE start_station_name = 'E 33 St & 1 Ave' or end_station_name = 'E 33 St & 1 Ave'
  ) as t1
  GROUP BY date, t1.trip_direction
) as t2
on date_table.date = t2.date
ORDER BY date_table.date
"""

trip_trend_df_all = spark.sql(sql_command)
trip_trend_df = trip_trend_df_all.filter((trip_trend_df_all.trip_direction == 'out') | (trip_trend_df_all.trip_direction.isNull()))

# COMMAND ----------

# MAGIC %md
# MAGIC Trip trends plot on daily, monthly, and weekday-based resolution. From the daily trip trend we can see that incoming and outgoing trips' counts are roughly the same.

# COMMAND ----------

display(trip_trend_df_all)

# COMMAND ----------

display(trip_trend_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Trip with US Holidays

# COMMAND ----------

us_holidays = holidays.UnitedStates(years = [2021,2022,2023])

# COMMAND ----------

# MAGIC %python
# MAGIC # holiday_dates = []
# MAGIC holidays_lt = []
# MAGIC for date, name in sorted(us_holidays.items()):
# MAGIC #     holiday_dates.append(date)
# MAGIC     holidays_lt.append([date.strftime("%Y-%m-%d"), name])

# COMMAND ----------

holidays_lt[10]

# COMMAND ----------

# MAGIC %python
# MAGIC trip_df = trip_trend_df.toPandas()
# MAGIC holidays_df = pd.DataFrame(holidays_lt, columns = ['date', 'name'])
# MAGIC holidays_df.head(3)

# COMMAND ----------

holidays_df['trip_count'] = 0
for i in range(len(holidays_df)):
    date = holidays_df['date'].iloc[i]
    if date in trip_df['date'].values:
        value = trip_df[trip_df['date']==date]['trip_count'].values[0]
        holidays_df['trip_count'].iloc[i] = value
holidays_df = holidays_df[holidays_df['trip_count']>0]
holidays_df.head(5)

# COMMAND ----------

# MAGIC %python
# MAGIC holiday_colors = {
# MAGIC     "Veterans Day": "lightcoral",
# MAGIC     "Thanksgiving" : "darkgreen",
# MAGIC     "Christmas Day" : "navy",
# MAGIC     "Christmas Day (Observed)" : "navy",
# MAGIC     "New Year's Day" : "red",
# MAGIC     "New Year's Day (Observed)" : "red",
# MAGIC     "Martin Luther King Jr. Day" : "teal",
# MAGIC     "Washington's Birthday" : "lime",
# MAGIC     "Memorial Day" : "blueviolet",
# MAGIC     "Juneteenth National Independence Day" : "burlywood",
# MAGIC     "Juneteenth National Independence Day (Observed)" :  "burlywood",
# MAGIC     "Independence Day" : "aqua",
# MAGIC     "Labor Day" : "hotpink",
# MAGIC     "Columbus Day" : "silver"
# MAGIC }

# COMMAND ----------

# MAGIC %md
# MAGIC The daily trip trend plot bellow incorporated the holidays. The dots represent the trip counts on the specific holidays and the colors differentiate different holidays.

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC plt.figure(dpi=200, figsize = (20,10))
# MAGIC dates = [datetime.strptime(x, '%Y-%m-%d') for x in trip_df['date'].values]
# MAGIC plt.plot(dates, trip_df['trip_count'])
# MAGIC legends = ['Daily Trip Trend']
# MAGIC # plt.scatter(holidays_df['date'], holidays_df['trip_count'], color = 'r', linewidth=3)
# MAGIC for i in range(len(holidays_df)):
# MAGIC     date = holidays_df['date'].iloc[i]
# MAGIC     value = holidays_df['trip_count'].iloc[i]
# MAGIC     name = holidays_df['name'].iloc[i]
# MAGIC     plt.scatter(datetime.strptime(date, '%Y-%m-%d'), value, color = holiday_colors[name], linewidth=3)
# MAGIC     legends.append(name)
# MAGIC # plt.axhline(y = 30, color = 'r', linestyle = '-')
# MAGIC # plt.axvline(x = 30, color = 'r', linestyle = '-')
# MAGIC # plt.axvline(x = 0, color = 'r', linestyle = '-')
# MAGIC 
# MAGIC plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# MAGIC plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# MAGIC 
# MAGIC plt.legend(legends, loc='upper right', prop={'size': 6})
# MAGIC plt.xlabel('Date')
# MAGIC plt.ylabel('Trip Count')
# MAGIC plt.xticks(rotation=45)
# MAGIC plt.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC # Trip With Weather

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *, FROM_UNIXTIME(dt)
# MAGIC FROM bronze_historic_weather_data
# MAGIC ORDER BY dt
# MAGIC LIMIT 30;

# COMMAND ----------

trip_trend_df.createOrReplaceTempView("trip_trend_table")

# COMMAND ----------

# MAGIC %md
# MAGIC The following plots try to display the influence of different weather aspects on the trip count of the station

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM trip_trend_table as t1
# MAGIC LEFT JOIN(
# MAGIC   SELECT date, AVG(temp) as avg_temp, AVG(feels_like) as avg_feels_like, AVG(humidity) as avg_humidity, AVG(wind_speed) as avg_wind_speed, AVG(pop) as avg_pop, AVG(snow_1h) as avg_snow
# MAGIC   FROM(
# MAGIC     SELECT concat(YEAR(FROM_UNIXTIME(dt)),'-',LPAD(MONTH(FROM_UNIXTIME(dt)), 2, '0'),'-',LPAD(DAY(FROM_UNIXTIME(dt)), 2, '0')) as date, DATE_FORMAT(FROM_UNIXTIME(dt),'HH:mm:ss') as time, `temp`, feels_like, humidity, wind_speed, pop, snow_1h
# MAGIC     FROM bronze_historic_weather_data
# MAGIC   )
# MAGIC   GROUP BY date
# MAGIC ) as t2
# MAGIC ON t1.date = t2.date
# MAGIC WHERE t2.avg_temp IS NOT NULL
# MAGIC ORDER BY t1.date;

# COMMAND ----------

sql_command2 = """
SELECT *
FROM trip_trend_table as t1
LEFT JOIN(
  SELECT date, AVG(temp) as avg_temp, AVG(feels_like) as avg_feels_like, AVG(humidity) as avg_humidity, AVG(wind_speed) as avg_wind_speed, AVG(pop) as avg_pop, AVG(snow_1h) as avg_snow
  FROM(
    SELECT concat(YEAR(FROM_UNIXTIME(dt)),'-',LPAD(MONTH(FROM_UNIXTIME(dt)), 2, '0'),'-',LPAD(DAY(FROM_UNIXTIME(dt)), 2, '0')) as date, DATE_FORMAT(FROM_UNIXTIME(dt),'HH:mm:ss') as time, `temp`, feels_like, humidity, wind_speed, pop, snow_1h
    FROM bronze_historic_weather_data
  )
  GROUP BY date
) as t2
ON t1.date = t2.date
WHERE t2.avg_temp IS NOT NULL
ORDER BY t1.date
"""

weather_trip_trend_df = spark.sql(sql_command2)

# COMMAND ----------

# MAGIC %md
# MAGIC The correlation heat mat shows the correlation between different station and weather variables.

# COMMAND ----------

# MAGIC %python
# MAGIC weather_trip_df = weather_trip_trend_df.toPandas()
# MAGIC weather_trip_df.head()

# COMMAND ----------

# MAGIC %python
# MAGIC # Compute correlation matrix
# MAGIC corr_matrix = weather_trip_df[['trip_count', 'avg_temp', 'avg_feels_like', 'avg_humidity', 'avg_wind_speed', 'avg_pop', 'avg_snow']].corr()
# MAGIC 
# MAGIC # Plot correlation heatmap
# MAGIC sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# MAGIC plt.title('Correlation Heatmap')
# MAGIC plt.figure(dpi=200, figsize=(20,10)) 
# MAGIC plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The percipitation probability and snow value may not only be use as a numerical variable. The amount may not be that important. Having percipitation or snow and having a sunny or windy day may already be influential to the station's operation.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT t1.trip_count, IF(t2.avg_snow>0, 1, 0) as has_snow, IF(t2.avg_pop>0.4, 1, 0) as likely_percipitating
# MAGIC FROM trip_trend_table as t1
# MAGIC LEFT JOIN(
# MAGIC   SELECT date, AVG(temp) as avg_temp, AVG(feels_like) as avg_feels_like, AVG(humidity) as avg_humidity, AVG(wind_speed) as avg_wind_speed, AVG(pop) as avg_pop, AVG(snow_1h) as avg_snow
# MAGIC   FROM(
# MAGIC     SELECT concat(YEAR(FROM_UNIXTIME(dt)),'-',LPAD(MONTH(FROM_UNIXTIME(dt)), 2, '0'),'-',LPAD(DAY(FROM_UNIXTIME(dt)), 2, '0')) as date, DATE_FORMAT(FROM_UNIXTIME(dt),'HH:mm:ss') as time, `temp`, feels_like, humidity, wind_speed, pop, snow_1h
# MAGIC     FROM bronze_historic_weather_data
# MAGIC   )
# MAGIC   GROUP BY date
# MAGIC ) as t2
# MAGIC ON t1.date = t2.date
# MAGIC WHERE t2.avg_temp IS NOT NULL
# MAGIC ORDER BY t1.date;

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


