from mlflow.spark import load_model
import mlflow
from pyspark.sql import SparkSession, DataFrame, functions as F
import pandas as pd
from datetime import datetime
from mlflow.tracking import MlflowClient
from typing import List, Tuple
from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank, col, when, struct, lit, expr, udf, ceil
from builtins import abs as python_abs
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import Evaluator
import numpy as np
import xgboost as xgb

spark = SparkSession.builder.getOrCreate()

def get_latest_model_version(client: MlflowClient,
                             model_name: str,
                             ) -> int:
    """
    Get the latest version of a model.

    Parameters:
    client (mlflow.tracking.MlflowClient): The MLflow client.
    model_name (str): The name of the model.

    Returns:
    int: The latest version of the model.
    """
    latest_version = 1
    for mv in client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def get_model_uri(registered_model_name: str,
                  alias: str,
                  ) -> Tuple:
    """
    Get the model URI and the loaded model.

    Parameters:
    model_name (str): The name of the model.
    alias (str): The alias of the model.

    Returns:
    Tuple[Any, str]: The loaded model and the model URI.
    """
    model_version_uri = f"models:/{registered_model_name}@{alias}"
    return model_version_uri


def batch_inference(registered_model_name: str,
                    alias: str,
                    data: DataFrame,
                    ) -> DataFrame:
    """
    Perform batch inference using a PyFunc model.

    Parameters:
    model_uri (str): The URI of the model.
    df (pd.DataFrame): The input DataFrame.
    cols (List[str]): The columns to use for inference.

    Returns:
    pd.DataFrame: The DataFrame with predictions.
    """
    model_version_uri = get_model_uri(registered_model_name, alias)
    model = load_model(model_version_uri)
    predictions_df = model.transform(data)

    return predictions_df


def batch_inference_xgb(registered_model_name: str,
                    alias: str,
                    data: xgb.DMatrix,
                    ) -> np.ndarray:
    """
    Perform batch inference using a PyFunc model.

    Parameters:
    model_uri (str): The URI of the model.
    df (pd.DataFrame): The input DataFrame.
    cols (List[str]): The columns to use for inference.

    Returns:
    pd.DataFrame: The DataFrame with predictions.
    """
    model_version_uri = get_model_uri(registered_model_name, alias)
    model = mlflow.xgboost.load_model(model_version_uri)
    predictions = model.predict(data)

    return predictions


def swap_alias(client: MlflowClient,
               model_name: str,
               ) -> None:
    """
    Swap the aliases of the model versions.

    Parameters:
    model_name (str): The name of the model.
    """
    champion_version = int(client.get_model_version_by_alias(model_name, "Champion").version)
    challenger_version = int(client.get_model_version_by_alias(model_name, "Challenger").version)

    client.delete_registered_model_alias(model_name,
                                         "Champion",
                                         )
    client.delete_registered_model_alias(model_name,
                                         "Challenger",
                                         )

    client.set_registered_model_alias(model_name,
                                      "Champion",
                                      challenger_version,
                                      )
    client.set_registered_model_alias(model_name,
                                      "Challenger",
                                      champion_version,
                                      )


def promote_challenger(client: MlflowClient,
                       model_name: str,
                       ) -> None:
    """
    Promote the challenger model to champion.

    Parameters:
    model_name (str): The name of the model.
    """
    challenger_version = int(client.get_model_version_by_alias(model_name, "Challenger").version)

    client.delete_registered_model_alias(model_name,
                                         "Champion",
                                         )
    client.delete_registered_model_alias(model_name,
                                         "Challenger",
                                         )

    client.set_registered_model_alias(model_name,
                                      "Champion",
                                      challenger_version,
                                      )