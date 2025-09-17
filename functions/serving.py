import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.deployments import get_deploy_client
from requests.exceptions import HTTPError

def deploy_champion_model(serving_client: mlflow.deployments.get_deploy_client,
                          client: MlflowClient,
                          registered_model_name: str,
                          endpoint_name: str,
                          model_name: str, # This model_name seems to be an alias for the served model
                          inference_table_catalog: str,
                          inference_table_schema: str,
                          cpu_size: str,
                          inference_table_name: str = None) -> str:
    """
    Deploy the Champion model to the specified endpoint.

    Args:
        serving_client (mlflow.deployments.get_deploy_client): The MLflow deployment client.
        client (MlflowClient): The MLflow client.
        registered_model_name (str): The fully qualified name of the registered model.
        endpoint_name (str): The name of the endpoint to create.
        model_name (str): The name of the model alias for the served entity.
        inference_table_catalog (str): Unity Catalog catalog for inference logging.
        inference_table_schema (str): Unity Catalog schema for inference logging.
        cpu_size (str): Workload size for the served model (e.g., "Small", "Medium").
        inference_table_name (str, optional): Base name for the inference logging table.
                                              Defaults to endpoint_name.

    Returns:
        str: A success message indicating the endpoint deployment.
    """
    model_version = int(client.get_model_version_by_alias(registered_model_name, "Champion").version)

    # Determine the actual table name for inference logging
    actual_inference_table_name = (inference_table_name if inference_table_name else endpoint_name) + '_payload'

    inference_logging_config = {
       "catalog_name": inference_table_catalog,
       "schema_name": inference_table_schema,
       "table_name_prefix": actual_inference_table_name.replace('_payload', ''), # Prefix should not contain _payload
       "enabled": True, # Ensure logging is enabled
    }

    # Ensure model_name (alias) is unique within the served_entities if multiple models are served
    served_model_alias = f"{model_name}-{model_version}" # Using alias in served_model_name is typical

    endpoint = serving_client.create_endpoint(
        name=f"{endpoint_name}",
        config={
            "served_models": [ # Use "served_models" for Databricks Model Serving
                {
                    "model_name": registered_model_name, # The actual registered model name
                    "model_version": f"{model_version}",
                    "workload_size": cpu_size,
                    "scale_to_zero_enabled": True,
                    "name": served_model_alias, # This is the name used in traffic_config
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_model_name": served_model_alias, # Must match "name" in served_models
                        "traffic_percentage": 100
                    }
                ]
            },
            "auto_capture_config": inference_logging_config # Use auto_capture_config for logging
        }
    )
    
    # Corrected: Separate print from return, use the derived table name
    success_message = f"Endpoint '{endpoint_name}' deployed with inference logging to {inference_table_catalog}.{inference_table_schema}.{actual_inference_table_name}"
    print(success_message)
    return success_message


def update_endpoint_with_new_model_version(
                                            serving_client: mlflow.deployments.get_deploy_client,
                                            client: MlflowClient, 
                                            registered_model_name: str, 
                                            endpoint_name: str, 
                                            model_name: str,
                                            inference_table_catalog: str,
                                            inference_table_schema: str,
                                            cpu_size: str,
                                            inference_table_name: str = None) -> str:
    """
    Updates the specified endpoint with a new version of the model.

    This function retrieves the version of the model registered under the "Champion" alias, 
    and updates the endpoint configuration with the new model version. It sets up traffic 
    routing and scaling options for the updated model.

    Args:
        serving_client (Any): The client used for interacting with the serving infrastructure.
        client (Any): The client used for fetching model information.
        registered_model_name (str): The name of the registered model to update.
        endpoint_name (str): The name of the endpoint to update with the new model version.
        model_name (str): The name of the model to be served at the endpoint.

    Returns:
        None: This function does not return any value, it performs an action to update the endpoint.
    """
    
    # Get the version of the model with the "Champion" alias
    model_version = int(client.get_model_version_by_alias(registered_model_name, "Champion").version)

    inference_logging_config = {
       "catalog_name": inference_table_catalog,
       "schema_name": inference_table_schema,
    }
    if inference_table_name:
       inference_logging_config["table_name"] = inference_table_name + '_payload'
    else:
       inference_logging_config["table_name"] = endpoint_name + '_payload'
       pass

    # Update the endpoint with the new model version and traffic configuration
    serving_client.update_endpoint(
        endpoint=f"{endpoint_name}",
        config={
            "served_entities": [
                {
                    "entity_name": registered_model_name,
                    "entity_version": model_version,  # No need for `int()` as `model_version` is already int
                    "workload_size": cpu_size,
                    "scale_to_zero_enabled": True,
                    "served_model_name": model_name
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_model_name": f"{model_name}-{model_version}",
                        "traffic_percentage": 100
                    }
                ]
            },
            "inference_logging": inference_logging_config
        }
    )
    return print(f"Endpoint '{endpoint_name}' deployed with inference logging to {inference_table_catalog}.{inference_table_schema}.{inference_logging_config['table_name']}")
