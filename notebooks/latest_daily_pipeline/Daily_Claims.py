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

# MAGIC %run ../../configs/configs
# MAGIC

# COMMAND ----------

# Import required libraries
import sys
import os
from pathlib import Path
from pyspark.sql.functions import *
from pyspark.sql import functions as F
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
sys_path = sys_path.replace("/notebooks", "")

sys.path.append(sys_path)
from functions.feature_engineering import FeatureEngineering
from functions.data_processing import DataPreprocessing

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

# In another notebook, access the global temp view as follows:
spark.table("global_temp.this_day")

this_day = spark.table("global_temp.this_day").collect()[0][0]

display(this_day)

# COMMAND ----------

spark.sql('USE CATALOG prod_adp_certified')

policy_svi = spark.table(f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_policy_svi").filter(
    F.to_date(F.col('reported_date').substr(1, 10), 'yyyy-MM-dd') > F.to_date(F.lit(this_day), 'yyyy-MM-dd')
)

policy_svi = spark.table(f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_policy_svi").filter(f"DATE(reported_date)>='{this_day}'")

policy_svi.createOrReplaceTempView("policy_svi")

latest_claim_version = policy_svi.selectExpr('claim_id', 'claim_version_id')

latest_claim_version.createOrReplaceTempView("latest_claim_version")

# COMMAND ----------


check_df = spark.sql(
"""
SELECT DISTINCT claim.claim_version.claim_number,
claim.claim_version.policy_number, 
claim.claim_version.claim_version_id,
claim.claim_version_item.claim_version_item_index, 
claim.claim_version.policy_cover_type,
claim.claim_version_item.claim_item_type, 
claim.claim_version_item.not_on_mid, 
claim.claim_version_item.vehicle_unattended,
claim.claim_version_item.excesses_applied,
claim.claim_version_item.total_loss_date, 
claim.claim_version_item.total_loss_flag,
claim.claim_version_item.first_party as cvi_first_party,
claim.claim_version_item.event_enqueued_utc_time AS latest_event_time,
claim.claimant.is_first_party,
incident.event_identity as incident_event_identity,
-- lcv.latest_event_time,
claim.incident.start_date,
claim.incident.reported_date,
claim.incident.multiple_parties_involved,
claim.incident.notification_method,
claim.incident.impact_speed,
claim.incident.impact_speed_unit,
claim.incident.impact_speed_range,
hour(claim.incident.start_date) as incidentHourC,
dayofweek(claim.incident.start_date) as incidentDayOfWeekC,
month(claim.incident.start_date) as incidentMonthC,
claim.incident.incident_location_longitude,
claim.incident.incident_type,
claim.incident.incident_cause,
claim.incident.incident_sub_cause,
claim.incident.circumstances, 
claim.vehicle.year_of_manufacture as manufacture_yr_claim,
claim.vehicle.outstanding_finance_amount,
claim.driver.driver_id,
YEAR(claim.incident.start_date) - YEAR(claim.driver.date_of_birth) as claim_driver_age,
claim.claim.first_party_confirmed_tp_notified_claim,
claim.claim_version.claim_id,
claim.emergency_services.is_air_ambulance_attendance, 
claim.emergency_services.is_ambulance_attendance, 
claim.emergency_services.is_crime_reference_provided, 
claim.emergency_services.is_fire_service_attendance, 
claim.emergency_services.is_police_attendance,  
claim.emergency_services.police_considering_actions, 
claim.damage_details.assessment_category,
claim.damage_details.boot_opens,
claim.damage_details.doors_open,
claim.damage_details.engine_damage,
claim.damage_details.front_severity, claim.damage_details.front_bonnet_severity, claim.damage_details.front_left_severity, claim.damage_details.front_right_severity, claim.damage_details.left_severity, claim.damage_details.left_back_seat_severity, claim.damage_details.left_front_wheel_severity, claim.damage_details.left_mirror_severity, claim.damage_details.left_rear_wheel_severity, claim.damage_details.left_underside_severity, claim.damage_details.rear_severity, claim.damage_details.rear_left_severity, claim.damage_details.rear_right_severity, claim.damage_details.rear_window_damage_severity, claim.damage_details.right_severity, claim.damage_details.right_back_seat_severity, claim.damage_details.right_front_wheel_severity, claim.damage_details.right_mirror_severity, claim.damage_details.right_rear_wheel_severity, claim.damage_details.right_roof_severity, claim.damage_details.right_underside_severity, claim.damage_details.roof_damage_severity, claim.damage_details.underbody_damage_severity, claim.damage_details.windscreen_damage_severity
FROM latest_claim_version lcv
INNER JOIN claim.claim_version
ON lcv.claim_id = claim_version.claim_id 
INNER JOIN claim.claim_version_item
ON lcv.claim_version_id = claim_version.claim_version_id
AND claim_version.claim_version_id = claim_version_item.claim_version_id
AND lcv.claim_id = claim_version_item.claim_id
INNER JOIN claim.claim
ON claim.claim_id = claim_version.claim_id
AND claim.claim_id = claim_version_item.claim_id
LEFT JOIN claim.damage_details
ON damage_details.event_identity = claim_version.event_identity
AND damage_details.claim_version_item_index = claim_version_item.claim_version_item_index
LEFT JOIN claim.incident
ON claim_version.event_identity = incident.event_identity
LEFT JOIN claim.vehicle
ON claim_version.event_identity = vehicle.event_identity
AND claim_version_item.claim_version_item_index = vehicle.claim_version_item_index
LEFT JOIN claim.claimant
ON claimant.claim_version_id = claim_version_item.claim_version_id
AND claimant.claim_version_item_index = claim_version_item.claim_version_item_index
AND claimant.event_identity = claim_version_item.event_identity
LEFT JOIN claim.emergency_services
ON claim.claim_version.event_identity = emergency_services.event_identity
LEFT JOIN claim.driver
ON claim.driver.claim_version_item_index = claim_version_item.claim_version_item_index
AND claim.driver.event_identity = claim_version_item.event_identity
AND claim_version.event_identity = claim.driver.event_identity
WHERE claim_version.claim_number IS NOT NULL
AND claim.claimant.is_first_party = true
AND claim_version_item.claim_version_item_index=0
"""
)

# COMMAND ----------

#add vehicle use etc from quotes
quote_iteration_df = spark.table("prod_adp_certified.quote_motor.quote_iteration")
vehicle_df = spark.table("prod_adp_certified.quote_motor.vehicle").selectExpr("quote_iteration_id", "vehicle_use_code AS vehicle_use_quote")

policy_svi = policy_svi.drop("claim_id", 'claim_version_id')

policy_svi = policy_svi.join(
    quote_iteration_df.select("session_id", "quote_iteration_id"),
    policy_svi.quote_session_id == quote_iteration_df.session_id, "left"
    ).join(vehicle_df, "quote_iteration_id", "left"
    ).select("*")


# Specify window for max transaction id per policy
window_spec = Window.partitionBy(col("policy_number")).orderBy(col("policy_transaction_id").desc())

#filter for the latest (max) transaction id 
policy_svi = policy_svi.withColumn("row_num", row_number().over(window_spec)).filter(col("row_num") == 1).drop("row_num")

policy_svi = policy_svi.drop("reported_date")

check_df = check_df.join(policy_svi, on="policy_number", how="left").filter(col("policy_transaction_id").isNotNull()).dropDuplicates()


# COMMAND ----------

driver_cols = ['additional_vehicles_owned', 'age_at_policy_start_date', 'cars_in_household', 'licence_length_years', 'years_resident_in_uk']

all_cols = list(policy_svi.columns)

for col_name in driver_cols:
    if f"{col_name}_5" not in all_cols:
        policy_svi = policy_svi.withColumn(f"{col_name}_5", lit(None))
    policy_svi = policy_svi.withColumn(
        f"max_{col_name}", 
        greatest(
            col(f"{col_name}_1"), 
            col(f"{col_name}_2"), 
            col(f"{col_name}_3"), 
            col(f"{col_name}_4"), 
            col(f"{col_name}_5")
        )
    )    
    policy_svi = policy_svi.withColumn(
        f"min_{col_name}", 
        least(
            col(f"{col_name}_1"), 
            col(f"{col_name}_2"), 
            col(f"{col_name}_3"), 
            col(f"{col_name}_4"), 
            col(f"{col_name}_5")
        )
    )

#if fifth driver present
drop_cols = ['claim_id', 'claim_version_id', 'additional_vehicles_owned_2', 'additional_vehicles_owned_3', 'additional_vehicles_owned_4', 'additional_vehicles_owned_5', 'age_at_policy_start_date_2', 'age_at_policy_start_date_3', 'age_at_policy_start_date_4', 'age_at_policy_start_date_5', 'cars_in_household_2', 'cars_in_household_3', 'cars_in_household_4', 'cars_in_household_5', 'licence_length_years_2', 'licence_length_years_3', 'licence_length_years_4', 'licence_length_years_5', 'years_resident_in_uk_2', 'years_resident_in_uk_3', 'years_resident_in_uk_4', 'years_resident_in_uk_5', 'reported_date']

policy_svi = policy_svi.drop(*drop_cols)

check_df = check_df.join(policy_svi, on="policy_number", how="left")
# filter for claims with only matched policies
check_df = check_df.filter(col("policy_transaction_id").isNotNull()).dropDuplicates()

# COMMAND ----------

# Initialize preprocessor
preprocessor = DataPreprocessing(spark, env_config = env_config)

# Deduplicate driver data
check_df = preprocessor.deduplicate_driver_data(check_df)

# Calculate damage scores
check_df = preprocessor.calculate_damage_scores(check_df)

# Calculate vehicle and driver features
check_df = preprocessor.calculate_vehicle_and_driver_features(check_df)

# Clean data types
check_df = preprocessor.clean_data_types(check_df)

# Remove nan in column causing model training issue
check_df = check_df.dropna(subset=['vehicle_overnight_location_id'])

# Fill missing data
check_df = preprocessor.fill_missing_values(check_df)

# COMMAND ----------

try:
    preprocessed_df = check_df
except:
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

    print("\nFeature engineering pipeline completed successfully")

# COMMAND ----------

df.write \
    .mode("overwrite") \
    .format("delta").option("overwriteSchema", "true") \
    .saveAsTable(f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_claims_svi")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The End
