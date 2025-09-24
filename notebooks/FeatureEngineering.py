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

# Get current environment configuration
current_env = get_current_environment()
env_config = get_environment_config()

print(f"Running feature engineering in environment: {current_env}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Feature Engineering Functions

# COMMAND ----------

def apply_damage_score_calculation(df):
    """Apply damage score calculations to the dataframe"""
    # Create relative damage score
    df = df.withColumn("areasDamagedRelative", 
                      col("areasDamagedMinimal") + 
                      2*col("areasDamagedMedium") + 
                      3*col("areasDamagedSevere") + 
                      4*col("areasDamagedHeavy"))
    return df

def create_time_based_features(df):
    """Create time-based features from incident date and other temporal columns"""
    # Extract time-based features
    if "start_date" in df.columns:
        df = df.withColumn("incident_hour", hour(col("start_date")))
        df = df.withColumn("incident_day_of_week", date_format(col("start_date"), "E"))
        df = df.withColumn("incident_month", month(col("start_date")))
        df = df.withColumn("incident_year", year(col("start_date")))
        
    if "reported_date" in df.columns:
        df = df.withColumn("reported_day_of_week", date_format(col("reported_date"), "E"))
        
    return df

def calculate_delays(df):
    """Calculate various delay metrics"""
    if "reported_date" in df.columns and "start_date" in df.columns:
        df = df.withColumn("delay_in_reporting", 
                          datediff(col("reported_date"), col("start_date")))
                          
    if "policy_start_date" in df.columns and "start_date" in df.columns:
        df = df.withColumn("inception_to_claim", 
                          datediff(col("start_date"), col("policy_start_date")))
                          
    if "policy_renewal_date" in df.columns and "start_date" in df.columns:
        df = df.withColumn("claim_to_policy_end", 
                          datediff(col("policy_renewal_date"), col("start_date")))
    
    return df

def create_check_variables(df):
    """Create business rule check variables for fraud detection"""
    
    check_cols = []
    
    # C1: Friday/Saturday night incident
    if "incident_day_of_week" in df.columns and "start_date" in df.columns:
        fri_sat_night = ((col("incident_day_of_week").isin("Fri", "Sat") & 
                         (hour(col("start_date")).between(20, 23))) | 
                        (col("incident_day_of_week").isin("Sat", "Sun") & 
                         (hour(col("start_date")).between(0, 4))))
        df = df.withColumn("C1_fri_sat_night",
                          when(fri_sat_night, 1)
                          .when(fri_sat_night.isNull(), 1)
                          .otherwise(0))
        check_cols.append("C1_fri_sat_night")
    
    # C2: Reporting delay
    if "delay_in_reporting" in df.columns:
        df = df.withColumn("C2_reporting_delay", 
                          when(col("delay_in_reporting") >= 3, 1)
                          .when(col("delay_in_reporting").isNull(), 1)
                          .otherwise(0))
        check_cols.append("C2_reporting_delay")
    
    # C3: Weekend incident reported Monday
    if "start_date" in df.columns and "reported_date" in df.columns:
        df = df.withColumn("is_incident_weekend",
                          when(date_format(col("start_date"), "E").isin("Fri", "Sat", "Sun"), 1)
                          .otherwise(0))
        df = df.withColumn("is_reported_monday",
                          when(date_format(col("reported_date"), "E") == "Mon", 1)
                          .otherwise(0))
        df = df.withColumn("C3_weekend_incident_reported_monday",
                          when((col("is_incident_weekend") == 1) & 
                               (col("is_reported_monday") == 1), 1)
                          .otherwise(0))
        check_cols.append("C3_weekend_incident_reported_monday")
    
    # C5: Night incident (11pm-5am)
    if "start_date" in df.columns:
        df = df.withColumn("C5_is_night_incident",
                          when((hour(col("start_date")) >= 23) | 
                               (hour(col("start_date")) <= 5) | 
                               hour(col("start_date")).isNull(), 1)
                          .otherwise(0))
        check_cols.append("C5_is_night_incident")
    
    # C6: No commuting but rush hour
    if "vehicle_use_quote" in df.columns and "start_date" in df.columns:
        not_commuting_rush = ((col("vehicle_use_quote") == 1) & 
                              ((hour(col("start_date")).between(6, 10)) | 
                               (hour(col("start_date")).between(15, 18))))
        df = df.withColumn("C6_no_commuting_but_rush_hour",
                          when(not_commuting_rush, 1)
                          .when(not_commuting_rush.isNull(), 1)
                          .otherwise(0))
        check_cols.append("C6_no_commuting_but_rush_hour")
    
    # C7: Police attended or crime reference
    if "is_police_attendance" in df.columns or "is_crime_reference_provided" in df.columns:
        df = df.withColumn("C7_police_attended_or_crime_reference",
                          when((col("is_police_attendance") == True) | 
                               (col("is_crime_reference_provided") == True), 1)
                          .otherwise(0))
        check_cols.append("C7_police_attended_or_crime_reference")
    
    # C9: Policy within 30 days
    if "inception_to_claim" in df.columns:
        df = df.withColumn("C9_policy_within_30_days",
                          when(col("inception_to_claim").between(0, 30), 1)
                          .when(col("inception_to_claim").isNull(), 1)
                          .otherwise(0))
        check_cols.append("C9_policy_within_30_days")
    
    # C10: Claim near policy end
    if "claim_to_policy_end" in df.columns:
        df = df.withColumn("C10_claim_to_policy_end",
                          when(col("claim_to_policy_end") < 60, 1)
                          .when(col("claim_to_policy_end").isNull(), 1)
                          .otherwise(0))
        check_cols.append("C10_claim_to_policy_end")
    
    # C11: Young or inexperienced driver
    if "age_at_policy_start_date_1" in df.columns or "licence_length_years_1" in df.columns:
        df = df.withColumn("driver_age_low_1",
                          when(col("age_at_policy_start_date_1") < 25, 1)
                          .when(col("age_at_policy_start_date_1").isNull(), 1)
                          .otherwise(0))
        df = df.withColumn("licence_low_1",
                          when(col("licence_length_years_1") <= 3, 1)
                          .otherwise(0))
        condition_inexperienced = (col("driver_age_low_1") == 1) | (col("licence_low_1") == 1)
        df = df.withColumn("C11_young_or_inexperienced",
                          when(condition_inexperienced, 1)
                          .when(condition_inexperienced.isNull(), 1)
                          .otherwise(0))
        check_cols.append("C11_young_or_inexperienced")
    
    # C12: Expensive car for driver age
    if "age_at_policy_start_date_1" in df.columns and "vehicle_value" in df.columns:
        condition_expensive = ((col("age_at_policy_start_date_1") < 25) & (col("vehicle_value") >= 20000)) | \
                             ((col("age_at_policy_start_date_1") >= 25) & (col("vehicle_value") >= 30000))
        df = df.withColumn("C12_expensive_for_driver_age",
                          when(condition_expensive, 1)
                          .when(condition_expensive.isNull(), 1)
                          .otherwise(0))
        check_cols.append("C12_expensive_for_driver_age")
    
    # C14: Contains watchwords
    if "Circumstances" in df.columns:
        watch_words = "|".join(["commut", "deliver", "parcel", "drink", "police", "custody", 
                               "arrest", "alcohol", "drug", "station", "custody"])
        df = df.withColumn("C14_contains_watchwords",
                          when(lower(col("Circumstances")).rlike(watch_words), 1)
                          .when(col("Circumstances").isNull(), 1)
                          .otherwise(0))
        check_cols.append("C14_contains_watchwords")
    
    # Create checks_max column if check columns exist
    if check_cols:
        df = df.withColumn("checks_max", greatest(*[col(c) for c in check_cols]))
        
        # Create num_failed_checks
        df = df.withColumn(
            "checks_list",
            array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in check_cols])
        )
        df = df.withColumn(
            "checks_list",
            expr("filter(checks_list, x -> x is not null)")
        ).withColumn("num_failed_checks", size(col("checks_list")))
    
    return df

def aggregate_driver_features(df):
    """Aggregate driver features across multiple drivers"""
    driver_cols = ['additional_vehicles_owned', 'age_at_policy_start_date', 
                   'cars_in_household', 'licence_length_years', 'years_resident_in_uk']
    
    for col_base in driver_cols:
        # Check if columns exist
        cols_to_check = [f"{col_base}_{i}" for i in range(1, 6)]
        existing_cols = [c for c in cols_to_check if c in df.columns]
        
        if existing_cols:
            # Create max and min aggregations
            df = df.withColumn(f"max_{col_base}", 
                              greatest(*[col(c) for c in existing_cols]))
            df = df.withColumn(f"min_{col_base}", 
                              least(*[col(c) for c in existing_cols]))
    
    # Handle min_claim_driver_age if it exists
    if "claim_driver_age" in df.columns:
        df = df.withColumn("claim_driver_age_low",
                          when(col("claim_driver_age") < 25, 1)
                          .when(col("claim_driver_age").isNull(), 1)
                          .otherwise(0))
    
    return df

def handle_missing_values(df):
    """Handle missing values with appropriate strategies"""
    
    # Define column groups for different fill strategies
    mean_fills = ["policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", 
                  "veh_age", "business_mileage", "annual_mileage", "incidentHourC", 
                  "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", 
                  "outstanding_finance_amount", "claim_to_policy_end", "num_failed_checks",
                  "incidentDayOfWeekC", "additional_vehicles_owned_1", "age_at_policy_start_date_1",
                  "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1",
                  "max_additional_vehicles_owned", "min_additional_vehicles_owned",
                  "max_age_at_policy_start_date", "min_age_at_policy_start_date",
                  "max_cars_in_household", "min_cars_in_household", "max_licence_length_years",
                  "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk"]
    
    # Boolean or damage columns with -1 fills
    neg_fills = ["vehicle_unattended", "excesses_applied", "is_first_party", 
                 "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance",
                 "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance",
                 "veh_age_more_than_10", "damageScore", "areasDamagedMinimal", "areasDamagedMedium",
                 "areasDamagedHeavy", "areasDamagedSevere", "areasDamagedTotal", "areasDamagedRelative",
                 "police_considering_actions", "is_crime_reference_provided", "ncd_protected_flag",
                 "boot_opens", "doors_open", "multiple_parties_involved", "is_incident_weekend",
                 "is_reported_monday", "driver_age_low_1", "claim_driver_age_low", "licence_low_1",
                 "total_loss_flag"]
    
    # Check variables fill with 1
    check_fills = [c for c in df.columns if c.startswith('C') and '_' in c]
    
    # String columns fill with 'missing'
    string_fills = ["assessment_category", "engine_damage", "policy_cover_type", 
                    "notification_method", "impact_speed_unit", "impact_speed_range",
                    "incident_type", "incident_cause", "incident_sub_cause"]
    
    # Apply fills based on column existence
    fills_dict = {}
    
    for col_name in mean_fills:
        if col_name in df.columns:
            fills_dict[col_name] = 0  # Will be replaced with actual mean in practice
            
    for col_name in neg_fills:
        if col_name in df.columns:
            fills_dict[col_name] = -1
            
    for col_name in check_fills:
        if col_name in df.columns:
            fills_dict[col_name] = 1
            
    for col_name in string_fills:
        if col_name in df.columns:
            fills_dict[col_name] = 'missing'
    
    if fills_dict:
        df = df.fillna(fills_dict)
    
    return df

def create_target_variable(df):
    """Create target variable for training"""
    if "svi_risk" in df.columns:
        # Ensure target is properly formatted
        df = df.withColumn("svi_risk", 
                          when(col("svi_risk") == -1, 0)
                          .otherwise(col("svi_risk")))
    
    if "tbg_risk" in df.columns:
        df = df.withColumn("referred_to_tbg", 
                          when(col("tbg_risk").isin([0, 1]), 1)
                          .otherwise(0))
    
    return df

def select_final_features(df):
    """Select final features for model training"""
    # Define feature groups
    numeric_features = [
        'policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age',
        'veh_age', 'business_mileage', 'annual_mileage', 'incidentHourC',
        'impact_speed', 'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim',
        'outstanding_finance_amount', 'claim_to_policy_end', 'incidentDayOfWeekC',
        'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy',
        'areasDamagedSevere', 'areasDamagedTotal', 'areasDamagedRelative',
        'num_failed_checks', 'damageScore'
    ]
    
    categorical_features = [
        'vehicle_unattended', 'excesses_applied', 'is_first_party',
        'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance',
        'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance',
        'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided',
        'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved',
        'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1',
        'claim_driver_age_low', 'licence_low_1', 'total_loss_flag',
        'checks_max'
    ]
    
    check_features = [c for c in df.columns if c.startswith('C') and '_' in c]
    
    id_features = ['claim_number', 'policy_number', 'dataset']
    target_features = ['svi_risk', 'tbg_risk', 'fa_risk', 'referred_to_tbg']
    
    # Get all available features
    all_features = []
    for feature_list in [numeric_features, categorical_features, check_features, 
                         id_features, target_features]:
        all_features.extend([f for f in feature_list if f in df.columns])
    
    # Add any additional columns that should be kept
    for col_name in df.columns:
        if col_name not in all_features and any(x in col_name for x in 
            ['age_at_policy_start_date', 'licence_length_years', 'years_resident_in_uk',
             'additional_vehicles_owned', 'cars_in_household', 'employment_type_abi_code',
             'severity', 'vehicle_overnight_location', 'incident', 'sales_channel',
             'policy_type', 'assessment_category', 'start_date', 'reported_date']):
            all_features.append(col_name)
    
    # Select only existing columns
    final_features = [f for f in all_features if f in df.columns]
    
    return df.select(final_features)

def save_feature_data(df, table_name=None):
    """Save processed feature data to table"""
    if table_name is None:
        table_name = f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.features_engineered"
    
    df.write \
        .mode("overwrite") \
        .format("delta") \
        .option("mergeSchema", "true") \
        .saveAsTable(table_name)
    
    print(f"Feature data saved to: {table_name}")

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
    # Apply feature engineering functions in sequence
    df = preprocessed_df
    
    # Apply damage score calculations
    df = apply_damage_score_calculation(df)
    
    # Create time-based features
    df = create_time_based_features(df)
    
    # Calculate delays
    df = calculate_delays(df)
    
    # Create check variables
    df = create_check_variables(df)
    
    # Aggregate driver features
    df = aggregate_driver_features(df)

    # Remove nan in column causing model training issue
    if 'vehicle_overnight_location_id' in df.columns:
        df = df.dropna(subset=['vehicle_overnight_location_id'])
    
    # Handle missing values
    df = handle_missing_values(df)

    # Create target variable
    df = create_target_variable(df)
    
    # Select final features
    df = select_final_features(df)
    
    # Save data
    save_feature_data(df)
    
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
    numeric_types = ['IntegerType', 'FloatType', 'DoubleType', 'LongType', 'DecimalType']
    print(f"- Numeric features: {len([c for c in df.columns if str(df.schema[c].dataType) in str(numeric_types)])}")
    print(f"All columns: {df.columns}")
    
    # Check variable summary
    check_cols = [c for c in df.columns if c.startswith('C') and '_' in c]
    if check_cols:
        print("\nCheck variable activation rates:")
        for check in sorted(check_cols):
            activation_rate = df.filter(col(check) == 1).count() / df.count() * 100
            print(f"  {check}: {activation_rate:.2f}%")