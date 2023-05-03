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
# TODO: NEED TO READ FROM DELTA TABLES
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

# specify the holidays that we want to consider
holiday_to_consider = pd.DataFrame({
  'holiday': 'holiday_to_consider',
  'ds': pd.to_datetime(['2021-01-01', '2021-11-25', '2021-12-25',
                        '2022-01-01', '2022-11-24', '2022-12-25',
                        '2023-01-01', '2023-11-23', '2023-12-25']),
  'lower_window': 0,
  'upper_window': 1,
})

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
        # create widget so it could be used by other notebook
        dbutils.widgets.text('Staging uri', model_uri)
        dbutils.widgets.text('Production uri', model_uri)
        staging_uri = None
        production_uri = None
else:
    print("Already has a production model, skip building the baseline model. ")
    version_list = client.search_model_versions("name = '%s'" % GROUP_MODEL_NAME)
    production_uri = [v.source for v in version_list if v.current_stage == "Production"][0]
    dbutils.widgets.text('Production uri', production_uri)
    latest_staging_version = max([v.version for v in version_list])
    staging_uri = [v.source for v in version_list if v.version == latest_staging_version][0]
    dbutils.widgets.text('Staging uri', staging_uri)

# COMMAND ----------

# if there is already a production model, construct a fine tuned model and push it only to staging. 
# helper function: define the objective function for hyperopt
def objective(search_space):
    updated_model = Prophet(yearly_seasonality=True, 
                             weekly_seasonality=True, 
                             daily_seasonality=True, 
                             changepoint_prior_scale=search_space["changepoint_prior_scale"], 
                             seasonality_prior_scale=search_space["seasonality_prior_scale"],
                             seasonality_mode="additive",
                             holidays=holiday_to_consider)
    # add additional multivariate regressors and holidays
    updated_model.add_regressor("weekday_indicator")
    updated_model.add_regressor("temp")
    updated_model.add_regressor("pop")
    updated_model.add_regressor("snow_1h")
    # updated_model.add_country_holidays(country_name='US')
    updated_model.fit(trip_data)
    # cross validation
    updated_model_cv = cross_validation(model=updated_model, horizon="91.25 days", parallel="threads")
    # model performance
    updated_model_p = performance_metrics(updated_model_cv, rolling_window=1)
    mse = updated_model_p["mse"].mean()
    return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

# define the search space for the hyperparameter
if production_exist:
    search_space = {
        'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.5),
        'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 100)
    }
    # set up other hyperparameter tuning arguments
    algo=tpe.suggest
    spark_trials = SparkTrials()

    with mlflow.start_run():
        argmin = fmin(
        fn=objective,
        space=search_space,
        algo=algo,
        max_evals=40,
        trials=spark_trials)

        # set up, register, and stage the best model found
        selected_model = Prophet(yearly_seasonality=True, 
                                weekly_seasonality=True, 
                                daily_seasonality=True, 
                                changepoint_prior_scale=argmin["changepoint_prior_scale"], 
                                seasonality_prior_scale=argmin["seasonality_prior_scale"],
                                seasonality_mode="additive",
                                holidays=holiday_to_consider)
        selected_model.add_regressor("weekday_indicator")
        selected_model.add_regressor("temp")
        selected_model.add_regressor("pop")
        selected_model.add_regressor("snow_1h")                        
        # selected_model.add_country_holidays(country_name='US')
        selected_model.fit(trip_data) # fit the model
        # cross validation
        selected_model_cv = cross_validation(model=selected_model, horizon="91.25 days", parallel="threads")
        # model performance
        selected_model_p = performance_metrics(selected_model_cv, rolling_window=1)
        # record the performance metric
        metric_dict = {}
        metric_list = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
        for m in metric_list:
            metric_dict[m] = selected_model_p[m].mean()
        # get the model parameter
        param = extract_params(selected_model)

        # log the original model 
        mlflow.prophet.log_model(selected_model, artifact_path=ARTIFACT_PATH) # store model artifact to be retrieved by app notebook
        mlflow.log_params(param)
        mlflow.log_metrics(metric_dict)
        model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
        print(model_uri)
        # register model and push to staging
        selected_model_detail = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
        client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=selected_model_detail.version, stage="Staging")

# remove the original widget for staging uri
if production_exist:
    dbutils.widgets.remove("Staging uri")

# COMMAND ----------

# updating staging uri widget
if production_exist:
    dbutils.widgets.text('Staging uri', model_uri)

# COMMAND ----------

# remove the registered model from mlflow
# for i in range(1, 9):
#     client.transition_model_version_stage(
#         name=GROUP_MODEL_NAME,
#         version=i,
#         stage="Archived"
#     )

# client.delete_registered_model(name=GROUP_MODEL_NAME)

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
