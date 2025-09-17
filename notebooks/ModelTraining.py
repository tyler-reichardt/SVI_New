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

# MAGIC %run ../configs/configs

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
from pyspark.sql.functions import col
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

notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)

sys.path.append(sys_path)
from functions.training import SVIModelTraining

# MLflow setup
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

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
# MAGIC ## Run Model Training

# COMMAND ----------

print("Starting SVI model training pipeline")

# Initialize trainer
trainer = SVIModelTraining(spark, env_config)

train_df, test_df = trainer.load_training_data()

train_df['tbg_risk'] = train_df['tbg_risk'].fillna(0).astype(int)
test_df['tbg_risk'] = test_df['tbg_risk'].fillna(0).astype(int)

y_train_desk_check = train_df['referred_to_tbg'].fillna(0).astype(int)
y_test_desk_check = test_df['referred_to_tbg'].fillna(0).astype(int)

y_train_interview = train_df['svi_risk'].fillna(0).astype(int)
y_test_interview = test_df['svi_risk'].fillna(0).astype(int)

print("\n" + "="*60)
print("STAGE 1: DESK CHECK MODEL TRAINING")
print("="*60)

desk_check_features, desk_check_preprocessor = trainer.prepare_desk_check_features(train_df)
desk_check_model, desk_check_pred_test, desk_check_actual_test = trainer.train_desk_check_model(
    train_df, y_train_desk_check,
    test_df, y_test_desk_check,
    desk_check_features, desk_check_preprocessor
)

desk_check_pred_train = desk_check_model.predict_proba(train_df[desk_check_features])[:, 1]

print("\n" + "="*60)
print("STAGE 2: INTERVIEW MODEL TRAINING")
print("="*60)

interview_features, interview_preprocessor = trainer.prepare_interview_features(train_df)
interview_model, interview_model_pred_test, interview_model_actual_test = trainer.train_interview_model(
    train_df, y_train_interview,
    test_df, y_test_interview,
    desk_check_pred_train, desk_check_pred_test,
    interview_features, interview_preprocessor
)

print("\n" + "="*60)
print("TRAINING PIPELINE COMPLETED")
print("="*60)

print("\n" + "="*60)
print("MODEL METRICS")
print("="*60)
print(f"Desk Check Model Accuracy: {trainer.simple_classification_report(desk_check_pred_test, desk_check_actual_test)}")
print(f"Interview Model Accuracy: {trainer.simple_classification_report(interview_model_pred_test, interview_model_actual_test)}")

print(f"Desk Check Model: {trainer.desk_check_model_name}")
print(f"Interview Model: {trainer.interview_model_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## The End
