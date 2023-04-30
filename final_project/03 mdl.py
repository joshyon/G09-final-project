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
# MAGIC ### Import Packages

# COMMAND ----------

# import useful package for ML
import json
import mlflow
import itertools
import datetime
import plotly.express as px
import pandas as pd
import numpy as np
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
np.random.seed(202)

ARTIFACT_PATH = "G09_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Silver Table As Training Data

# COMMAND ----------

# read silver table
trip_data = spark.read.csv("dbfs:/FileStore/tables/G09/hourly_trip_data/", header=True, inferSchema=True).toPandas()
trip_data = trip_data.rename(columns={"date_timestamp": "ds", "bikes_net_change": "y"}) # rename columns to be automatically identified by Prophet model
display(trip_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML Model Construction

# COMMAND ----------

# check whether there is already a model in production. if not, create a baseline model
production_exist = False
client = mlflow.tracking.MlflowClient()
version_list = client.search_model_versions("name = '%s'" % GROUP_MODEL_NAME)
for version in version_list:
    if version.current_stage == "Production":
        production_exist = True
        break

print(f"Currently has a production model: {production_exist}")

# COMMAND ----------

# extract model parameters. reference and credit: prophet example notebook provided in the share folder.  
def extract_params(model):
    return {attr: getattr(model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

# COMMAND ----------

# if there is not a production model yet, create a baseline model and push it to production and staging at the same time
# TODO: LOG THE BASELINE MODEL, PUSH IT TO PRODUCTION, AND WORK ON MORE COMPLEX MODELS
if not production_exist:
    with mlflow.start_run():
        baseline_model = Prophet()
        # add additional multivariate regressors
        baseline_model.add_regressor("weekday_indicator")
        baseline_model.add_regressor("temp")
        baseline_model.add_regressor("pop")
        baseline_model.add_regressor("snow_1h")
        baseline_model.fit(trip_data) # fit the model
        # cross validation
        baseline_model_cv = cross_validation(model=baseline_model, horizon="91.25 days", parallel="threads")
        # model performance
        baseline_model_p = performance_metrics(baseline_model_cv, rolling_window=1)
        # display(baseline_model_p)
        # record the performance metric
        metric_dict = {}
        metric_list = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
        for m in metric_list:
            metric_dict[m] = baseline_model_p[m].mean()
        # get the model parameter
        param = extract_params(baseline_model)

        # log the original model
        mlflow.prophet.log_model(baseline_model, artifact_path=ARTIFACT_PATH)
        mlflow.log_params(param)
        mlflow.log_metrics(metric_dict)
        model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
        # register model and push to staging
        baseline_model_detail = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
        client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=baseline_model_detail.version, stage="Staging")
        # register a copy of the original model to push to production
        baseline_model_detail2 = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
        client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=baseline_model_detail2.version, stage="Production")
        dbutils.widgets.text('Staging uri', model_uri)
        dbutils.widgets.text('Production uri', model_uri)


# COMMAND ----------

# if there is already a production model, construct a fine tuned model and push it only to staging. 
# param_grid = {
#     'yearly_seasonality': [True],
#     'weekly_seasonality': [True],
#     'daily_seasonality': [True],
#     'changepoint_prior_scale': [0.001, 0.05, 0.08, 0.1, 0.25, 0.5]
#     'seasonality_prior_scale': [0.1, 1, 5, 10, 15, 20, 50, 80, 100]
#     'seasonality_mode': ['additive', 'multiplicative']
# }

# # reference and credit: prophet example notebook provided in the share folder.  
# all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


