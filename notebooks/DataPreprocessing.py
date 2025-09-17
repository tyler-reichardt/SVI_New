# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing Module
# MAGIC
# MAGIC Production-ready data preprocessing pipeline for Single Vehicle Incident (SVI) fraud detection.
# MAGIC This module handles:
# MAGIC - Multi-environment data source management
# MAGIC - Claim and policy data extraction from ADP certified catalog
# MAGIC - Fraud risk indicators from referral logs
# MAGIC - Target variable creation for model training
# MAGIC - Data quality checks and validation
# MAGIC - Train/test split with stratification

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# Import required libraries
import sys
import os
from pathlib import Path
from pyspark.sql.functions import (
    col, row_number, greatest, least, collect_list, lower, mean, mode, when, 
    regexp_replace, min, max, datediff, to_date, concat, lit, round, 
    date_format, hour, udf
)
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql import Window
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pyspark.sql import functions as F

notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)

sys.path.append(sys_path)
from functions.data_processing import DataPreprocessing

# Get environment configuration
current_env = get_current_environment()
env_config = get_environment_config()

print(f"Running in environment: {current_env}")
print(f"MLStore catalog: {env_config['mlstore_catalog']}")
print(f"Auxiliary catalog: {env_config['auxiliary_catalog']}")
print(f"ADP catalog: {env_config['adp_catalog']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load data preprocessing configuration
extract_column_transformation_lists("/config_files/data_preprocessing.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Preprocessing Pipeline

# COMMAND ----------

# Initialize preprocessor
preprocessor = DataPreprocessing(spark, env_config = env_config)

# Load claim referral log
clm_log = preprocessor.load_claim_referral_log()

# Load SVI performance data and create target
target_df = preprocessor.load_svi_performance_data(clm_log)

# Get latest claim version
latest_claim_version = preprocessor.get_latest_claim_version(target_df)

# Load claim data
check_df = preprocessor.load_claim_data(latest_claim_version)

# Load policy data
policy_svi = preprocessor.load_policy_data(check_df)

# Deduplicate driver data
check_df = preprocessor.deduplicate_driver_data(check_df)

# Join claim and policy data
check_df = preprocessor.join_claim_and_policy_data(check_df, policy_svi)

# Calculate damage scores
check_df = preprocessor.calculate_damage_scores(check_df)

# Calculate vehicle and driver features
check_df = preprocessor.calculate_vehicle_and_driver_features(check_df)

# Create check variables (C1-C14)
check_df = preprocessor.create_check_variables(check_df)

# Clean data types
check_df = preprocessor.clean_data_types(check_df)

# Remove nan in column causing model training issue
check_df = check_df.dropna(subset=['vehicle_overnight_location_id'])

# Fill missing data
check_df = preprocessor.fill_missing_values(check_df)

# Create train/test split
check_df = preprocessor.create_train_test_split(check_df)

# Save if requested
preprocessor.save_processed_data(check_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

# Show data quality summary
print(f"Number of columns: {len(check_df.columns)}")
print(f"Total records: {check_df.count()}")
print(f"Train records: {check_df.filter(col('dataset') == 'train').count()}")
print(f"Test records: {check_df.filter(col('dataset') == 'test').count()}")
print(f"All columns: {check_df.columns}")

# Target distribution
check_df.groupBy("dataset", "svi_risk").count().orderBy("dataset", "svi_risk").show()

# Missing values summary
missing_counts = {}
for column in check_df.columns:
    missing = check_df.filter(col(column).isNull()).count()
    if missing > 0:
        missing_counts[column] = missing

if missing_counts:
    print("\nColumns with missing values:")
    for col_name, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col_name}: {count} ({count/check_df.count()*100:.2f}%)")
