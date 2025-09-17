# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training Module
# MAGIC
# MAGIC Production-ready model training pipeline for Single Vehicle Incident (SVI) fraud detection.
# MAGIC This module handles:
# MAGIC - Two-stage model training (Desk Check + Interview models)
# MAGIC - Multi-environment support (dsexp, modelbuild, modelpromotion, modeldeployment)
# MAGIC - Hyperparameter tuning using LightGBM
# MAGIC - Model registration to MLflow with environment-aware naming
# MAGIC - Comprehensive model evaluation and metrics tracking

# COMMAND ----------

# MAGIC %run ../../configs/configs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports and Configuration Setup

# COMMAND ----------

# Import required libraries
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from pyspark.sql.functions import array, when, lit, expr, col
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, List, Any
import warnings
warnings.filterwarnings('ignore')

# MLflow setup
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)
sys_path = sys_path.replace("/notebooks", "")

sys.path.append(sys_path)
from functions.scoring import SVIModelScoring

# Load configuration
extract_column_transformation_lists("/config_files/training.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# Get current environment
current_env = get_current_environment()
env_config = get_environment_config()

print(f"Running model training in environment: {current_env}")

# Define model names based on environment
desk_check_model_name = f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_desk_check_lgbm"
interview_model_name = f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_interview_lgbm"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Model Training

# COMMAND ----------

# Initialize trainer
scorer = SVIModelScoring(spark, env_config)

# Run training pipeline
df_scoring = scorer.load_training_data()

df_scoring['referred_to_tbg'] = df_scoring['referred_to_tbg'].fillna(0).astype(int)

print("\n" + "="*60)
print("STAGE 1: DESK CHECK MODEL SCORING")
print("="*60)

desk_check_features = scorer.prepare_desk_check_features(df_scoring)
df_scoring = scorer.desk_check_scoring(
    df_scoring,
    desk_check_features
)

print("\n" + "="*60)
print("STAGE 2: INTERVIEW MODEL SCORING")
print("="*60)

interview_features = scorer.prepare_interview_features(df_scoring)
df_scoring = scorer.interview_scoring(
    df_scoring,
    interview_features
)

# Check if file path exists, if not create it
scorer.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {env_config['mlstore_catalog']}.single_vehicle_incident_checks")

# Convert Pandas DataFrame to Spark DataFrame
df_scoring_spark = spark.createDataFrame(df_scoring)

# Delete the table if it exists
scorer.spark.sql(f"DROP TABLE IF EXISTS {env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_svi_predictions")

# Write with partitioning
df_scoring_spark.write.mode("overwrite").option("overwriteSchema", "true").partitionBy("dataset").saveAsTable(f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_svi_predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run SVIModelEvaluation notebook to evaluate model performance
# MAGIC 2. Use SVIModelServing for real-time inference
# MAGIC 3. Monitor model performance and retrain as needed
