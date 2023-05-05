# Databricks notebook source
# MAGIC %run ./includes/includes
# MAGIC dbutils.widgets.removeAll()
# MAGIC
# MAGIC dbutils.widgets.text('01.start_date', "2021-10-01")
# MAGIC dbutils.widgets.text('02.end_date', "2023-03-01")
# MAGIC dbutils.widgets.text('03.hours_to_forecast', '4')
# MAGIC dbutils.widgets.text('04.promote_model', 'No')
# MAGIC
# MAGIC start_date = str(dbutils.widgets.get('01.start_date'))
# MAGIC end_date = str(dbutils.widgets.get('02.end_date'))
# MAGIC hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
# MAGIC promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)
# MAGIC
# MAGIC print(start_date,end_date,hours_to_forecast, promote_model)
# MAGIC # print("YOUR CODE HERE...")
# MAGIC
# MAGIC print(start_date,end_date,hours_to_forecast, promote_model)
# MAGIC print("YOUR CODE HERE...")
# MAGIC %md
# MAGIC ### Import Packages
# MAGIC # import useful package for ML
# MAGIC import json
# MAGIC import mlflow
# MAGIC import itertools
# MAGIC import datetime
# MAGIC import plotly.express as px
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC from prophet import Prophet, serialize
# MAGIC from prophet.diagnostics import cross_validation, performance_metrics
# MAGIC from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
# MAGIC np.random.seed(202)
# MAGIC
# MAGIC ARTIFACT_PATH = "G09_model"
# MAGIC %md
# MAGIC ### Read Silver Table As Training Data
# MAGIC # read silver table
# MAGIC # TODO: NEED TO READ FROM DELTA TABLES
# MAGIC trip_data = spark.read.format("delta").load("dbfs:/FileStore/tables/G09/silver_hourly_trip_info.delta/").toPandas()
# MAGIC trip_data = trip_data.rename(columns={"date_timestamp": "ds", "bikes_net_change": "y"}) # rename columns to be automatically identified by Prophet model
# MAGIC trip_data = trip_data.sort_values(["ds"])
# MAGIC display(trip_data)
# MAGIC %md
# MAGIC ### ML Model Construction
# MAGIC # check whether there is already a model in production. if not, create a baseline model
# MAGIC production_exist = False
# MAGIC client = mlflow.tracking.MlflowClient()
# MAGIC version_list = client.search_model_versions("name = '%s'" % GROUP_MODEL_NAME)
# MAGIC for version in version_list:
# MAGIC     if version.current_stage == "Production":
# MAGIC         production_exist = True
# MAGIC         break
# MAGIC
# MAGIC print(f"Currently has a production model: {production_exist}")
# MAGIC # extract model parameters. reference and credit: prophet example notebook provided in the share folder.  
# MAGIC def extract_params(model):
# MAGIC     return {attr: getattr(model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
# MAGIC # specify the holidays that we want to consider
# MAGIC holiday_to_consider = pd.DataFrame({
# MAGIC   'holiday': 'holiday_to_consider',
# MAGIC   'ds': pd.to_datetime(['2021-01-01', '2021-11-25', '2021-12-25',
# MAGIC                         '2022-01-01', '2022-11-24', '2022-12-25',
# MAGIC                         '2023-01-01', '2023-11-23', '2023-12-25']),
# MAGIC   'lower_window': 0,
# MAGIC   'upper_window': 1,
# MAGIC })
# MAGIC # if there is not a production model yet, create a baseline model and push it to production and staging at the same time
# MAGIC # TODO: LOG THE BASELINE MODEL, PUSH IT TO PRODUCTION, AND WORK ON MORE COMPLEX MODELS
# MAGIC if not production_exist:
# MAGIC     with mlflow.start_run():
# MAGIC         baseline_model = Prophet()
# MAGIC         # add additional multivariate regressors
# MAGIC         baseline_model.add_regressor("weekday_indicator")
# MAGIC         baseline_model.add_regressor("temp")
# MAGIC         baseline_model.add_regressor("pop")
# MAGIC         baseline_model.add_regressor("snow_1h")
# MAGIC         baseline_model.fit(trip_data) # fit the model
# MAGIC         # cross validation
# MAGIC         baseline_model_cv = cross_validation(model=baseline_model, horizon="91.25 days", parallel="threads")
# MAGIC         # model performance
# MAGIC         baseline_model_p = performance_metrics(baseline_model_cv, rolling_window=1)
# MAGIC         # display(baseline_model_p)
# MAGIC         # record the performance metric
# MAGIC         metric_dict = {}
# MAGIC         metric_list = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
# MAGIC         for m in metric_list:
# MAGIC             metric_dict[m] = baseline_model_p[m].mean()
# MAGIC         # get the model parameter
# MAGIC         param = extract_params(baseline_model)
# MAGIC
# MAGIC         # log the original model
# MAGIC         mlflow.prophet.log_model(baseline_model, artifact_path=ARTIFACT_PATH)
# MAGIC         mlflow.log_params(param)
# MAGIC         mlflow.log_metrics(metric_dict)
# MAGIC         model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
# MAGIC         # register model and push to staging
# MAGIC         baseline_model_detail = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
# MAGIC         client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=baseline_model_detail.version, stage="Staging")
# MAGIC         # register a copy of the original model to push to production
# MAGIC         baseline_model_detail2 = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
# MAGIC         client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=baseline_model_detail2.version, stage="Production")
# MAGIC         # create widget so it could be used by other notebook
# MAGIC         dbutils.widgets.text('Staging uri', model_uri)
# MAGIC         dbutils.widgets.text('Production uri', model_uri)
# MAGIC         staging_uri = None
# MAGIC         production_uri = None
# MAGIC else:
# MAGIC     print("Already has a production model, skip building the baseline model. ")
# MAGIC     version_list = client.search_model_versions("name = '%s'" % GROUP_MODEL_NAME)
# MAGIC     production_uri = [v.source for v in version_list if v.current_stage == "Production"][0]
# MAGIC     dbutils.widgets.text('Production uri', production_uri)
# MAGIC     latest_staging_version = max([v.version for v in version_list])
# MAGIC     staging_uri = [v.source for v in version_list if v.version == latest_staging_version][0]
# MAGIC     dbutils.widgets.text('Staging uri', staging_uri)
# MAGIC # if there is already a production model, construct a fine tuned model and push it only to staging. 
# MAGIC # helper function: define the objective function for hyperopt
# MAGIC def objective(search_space):
# MAGIC     updated_model = Prophet(yearly_seasonality=True, 
# MAGIC                              weekly_seasonality=True, 
# MAGIC                              daily_seasonality=True, 
# MAGIC                              changepoint_prior_scale=search_space["changepoint_prior_scale"], 
# MAGIC                              seasonality_prior_scale=search_space["seasonality_prior_scale"],
# MAGIC                              seasonality_mode="additive",
# MAGIC                              holidays=holiday_to_consider)
# MAGIC     # add additional multivariate regressors and holidays
# MAGIC     updated_model.add_regressor("weekday_indicator")
# MAGIC     updated_model.add_regressor("temp")
# MAGIC     updated_model.add_regressor("pop")
# MAGIC     updated_model.add_regressor("snow_1h")
# MAGIC     # updated_model.add_country_holidays(country_name='US')
# MAGIC     updated_model.fit(trip_data)
# MAGIC     # cross validation
# MAGIC     updated_model_cv = cross_validation(model=updated_model, horizon="91.25 days", parallel="threads")
# MAGIC     # model performance
# MAGIC     updated_model_p = performance_metrics(updated_model_cv, rolling_window=1)
# MAGIC     mse = updated_model_p["mse"].mean()
# MAGIC     return {'loss': mse, 'status': STATUS_OK}
# MAGIC # define the search space for the hyperparameter
# MAGIC if production_exist:
# MAGIC     search_space = {
# MAGIC         'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.5),
# MAGIC         'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 100)
# MAGIC     }
# MAGIC     # set up other hyperparameter tuning arguments
# MAGIC     algo=tpe.suggest
# MAGIC     spark_trials = SparkTrials()
# MAGIC
# MAGIC     with mlflow.start_run():
# MAGIC         argmin = fmin(
# MAGIC         fn=objective,
# MAGIC         space=search_space,
# MAGIC         algo=algo,
# MAGIC         max_evals=40,
# MAGIC         trials=spark_trials)
# MAGIC
# MAGIC         # set up, register, and stage the best model found
# MAGIC         selected_model = Prophet(yearly_seasonality=True, 
# MAGIC                                 weekly_seasonality=True, 
# MAGIC                                 daily_seasonality=True, 
# MAGIC                                 changepoint_prior_scale=argmin["changepoint_prior_scale"], 
# MAGIC                                 seasonality_prior_scale=argmin["seasonality_prior_scale"],
# MAGIC                                 seasonality_mode="additive",
# MAGIC                                 holidays=holiday_to_consider)
# MAGIC         selected_model.add_regressor("weekday_indicator")
# MAGIC         selected_model.add_regressor("temp")
# MAGIC         selected_model.add_regressor("pop")
# MAGIC         selected_model.add_regressor("snow_1h")                        
# MAGIC         # selected_model.add_country_holidays(country_name='US')
# MAGIC         selected_model.fit(trip_data) # fit the model
# MAGIC         # cross validation
# MAGIC         selected_model_cv = cross_validation(model=selected_model, horizon="91.25 days", parallel="threads")
# MAGIC         # model performance
# MAGIC         selected_model_p = performance_metrics(selected_model_cv, rolling_window=1)
# MAGIC         # record the performance metric
# MAGIC         metric_dict = {}
# MAGIC         metric_list = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
# MAGIC         for m in metric_list:
# MAGIC             metric_dict[m] = selected_model_p[m].mean()
# MAGIC         # get the model parameter
# MAGIC         param = extract_params(selected_model)
# MAGIC
# MAGIC         # log the original model 
# MAGIC         mlflow.prophet.log_model(selected_model, artifact_path=ARTIFACT_PATH) # store model artifact to be retrieved by app notebook
# MAGIC         mlflow.log_params(param)
# MAGIC         mlflow.log_metrics(metric_dict)
# MAGIC         model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
# MAGIC         print(model_uri)
# MAGIC         # register model and push to staging
# MAGIC         selected_model_detail = mlflow.register_model(model_uri=model_uri, name=GROUP_MODEL_NAME)
# MAGIC         client.transition_model_version_stage(name=GROUP_MODEL_NAME, version=selected_model_detail.version, stage="Staging")
# MAGIC
# MAGIC # remove the original widget for staging uri
# MAGIC if production_exist:
# MAGIC     dbutils.widgets.remove("Staging uri")
# MAGIC # updating staging uri widget
# MAGIC if production_exist:
# MAGIC     dbutils.widgets.text('Staging uri', model_uri)
# MAGIC # remove the registered model from mlflow
# MAGIC # for i in range(1, 9):
# MAGIC #     client.transition_model_version_stage(
# MAGIC #         name=GROUP_MODEL_NAME,
# MAGIC #         version=i,
# MAGIC #         stage="Archived"
# MAGIC #     )
# MAGIC
# MAGIC # client.delete_registered_model(name=GROUP_MODEL_NAME)
# MAGIC import json
# MAGIC
# MAGIC # Return Success
# MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------



# COMMAND ----------


