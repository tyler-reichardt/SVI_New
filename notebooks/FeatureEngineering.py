# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Module
# MAGIC
# MAGIC Production-ready feature engineering pipeline for Single Vehicle Incident (SVI) fraud detection.
# MAGIC This module handles:
# MAGIC - Damage score calculations and severity aggregations
# MAGIC - Business rule check variables (C1-C14) for fraud detection
# MAGIC - Time-based and demographic feature engineering
# MAGIC - Driver feature aggregations
# MAGIC - Missing value imputation strategies

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# Import required libraries
import sys
import os
from pathlib import Path
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, FloatType
from pyspark.sql import Window
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pyspark.sql.functions import col, when, greatest, sum as spark_sum
from functools import reduce

notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)

sys.path.append(sys_path)
from functions.feature_engineering import FeatureEngineering

# Get current environment configuration
current_env = get_current_environment()
env_config = get_environment_config()

print(f"Running feature engineering in environment: {current_env}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load feature engineering configuration
extract_column_transformation_lists("/config_files/feature_engineering.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Feature Engineering Pipeline

# COMMAND ----------

# Load preprocessed data
table_path = get_table_path("single_vehicle_incident_checks", "claims_pol_svi", "mlstore")
try:
    preprocessed_df = spark.table(table_path)
    print(f"\nLoaded preprocessed data from: {table_path}")
except:
    print(f"\nCould not load data from {table_path}. Run DataPreprocessing first.")
    preprocessed_df = None

if preprocessed_df:
    # Initialize feature engineering
    feature_engineer = FeatureEngineering(spark, env_config = env_config)
    
    # Apply damage score calculations
    df = feature_engineer.apply_damage_score_calculation(preprocessed_df)
    
    # Create time-based features
    df = feature_engineer.create_time_based_features(df)
    
    # Calculate delays
    df = feature_engineer.calculate_delays(df)
    
    # Create check variables
    df = feature_engineer.create_check_variables(df)
    
    # Aggregate driver features (verify they exist)
    df = feature_engineer.aggregate_driver_features(df)

    # Remove nan in column causing model training issue
    df = df.dropna(subset=['vehicle_overnight_location_id'])
    
    # Handle missing values
    df = feature_engineer.handle_missing_values(df)

    # Create target variable
    df = feature_engineer.create_target_variable(df)
    
    # Select final features
    df = feature_engineer.select_final_features(df)
    
    # Save data
    feature_engineer.save_feature_data(df)
    
    print("\nFeature engineering pipeline completed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary

# COMMAND ----------

if df:
    # Show feature statistics
    print(f"Total features: {len(df.columns)}")
    print(f"\nFeature groups:")
    print(f"- Check variables: {len([c for c in df.columns if c.startswith('C') and '_' in c])}")
    print(f"- Numeric features: {len([c for c in df.columns if df.schema[c].dataType in ['IntegerType', 'FloatType', 'DoubleType']])}")
    print(f"All columns: {df.columns}")
    
    # Check variable summary
    check_cols = [c for c in df.columns if c.startswith('C') and '_' in c]
    if check_cols:
        print("\nCheck variable activation rates:")
        for check in sorted(check_cols):
            activation_rate = df.filter(col(check) == 1).count() / df.count() * 100
            print(f"  {check}: {activation_rate:.2f}%")
