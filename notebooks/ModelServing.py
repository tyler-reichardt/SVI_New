# Databricks notebook source
# MAGIC %run ../configs/configs

# COMMAND ----------

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

if workspace_url == 'adb-7739692825668627.5.azuredatabricks.net':
    dbutils.notebook.exit("Exiting notebook as Serving Endpoint should not be created in Model Build")

# COMMAND ----------

import sys

notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)
sys.path.append(sys_path)

from functions.serving import *

import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.deployments import get_deploy_client
from requests.exceptions import HTTPError

with open(f'{config_path}', 'r') as file:
    config = yaml.safe_load(file)
    
# Extract the congig lists
extract_column_transformation_lists("/config_files/serving.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
catalog = workspaces.get(workspace_url) + catalog_prefix

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
serving_client = get_deploy_client("databricks")
client = MlflowClient()

registered_model_name = f"{catalog}.{schema}.{model_name}"
endpoint_name = f"{schema}_endpoint"

# COMMAND ----------

# Create a new endpoint or update existing endpoint with new champion model version
try:
    deploy_champion_model(
                    serving_client,
                    client,
                    registered_model_name,
                    endpoint_name,
                    model_name,
                    catalog,
                    schema,
                    cpu_size
                    )
    
except HTTPError as e:
    if e.response.status_code == 400:
        update_endpoint_with_new_model_version(
                    serving_client,
                    client,
                    registered_model_name,
                    endpoint_name,
                    model_name,
                    catalog,
                    schema,
                    cpu_size
                    )
    else:
        raise

except Exception as e:
    print(f"Error deploying model: {e}")
