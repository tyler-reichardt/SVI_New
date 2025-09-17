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

# MAGIC %run ../../configs/configs

# COMMAND ----------

# Import required libraries
import sys
import os
from pathlib import Path
from pyspark.sql.functions import col, row_number, greatest, least, collect_list, lower, mean, when, regexp_replace, min, max, datediff, to_date, concat, lit, round, date_format, hour, udf, current_date
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
sys_path = sys_path.replace("/notebooks", "")

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

from pyspark.sql.utils import AnalysisException

try:
    df = spark.table(f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_policy_svi")
    if df.rdd.isEmpty():
        this_day = "1900-01-01"
    else:
        latest_date = df.agg(F.max("reported_date").alias("latest_date")).collect()[0]["latest_date"]
        this_day = latest_date.strftime('%Y-%m-%d')
except AnalysisException:
    this_day = "1900-01-01"

display(this_day)

# COMMAND ----------

spark.createDataFrame([(this_day,)], ["this_day"]).createOrReplaceGlobalTempView("this_day")

# COMMAND ----------

policy_transaction = spark.sql("""
SELECT 
    -- Columns from policy_transaction table
    pt.policy_transaction_id,
    pt.sales_channel, 
    pt.quote_session_id,
    pt.customer_focus_id,
    pt.customer_focus_version_id,
    pt.policy_number
    FROM prod_adp_certified.policy_motor.policy_transaction pt """)

# COMMAND ----------

policy = spark.sql(""" 
SELECT
    p.policy_number,
    p.policy_start_date,
    p.policy_renewal_date,
    p.policy_type,
    p.policyholder_ncd_years,
    p.ncd_protected_flag,
    p.policy_number FROM prod_adp_certified.policy_motor.policy p""")

# COMMAND ----------

vehicle = spark.sql(""" SELECT 
    v.policy_transaction_id,
    v.vehicle_overnight_location_code as overnight_location_abi_code,
    vo.vehicle_overnight_location_id, 
    vo.vehicle_overnight_location_name, 
    v.business_mileage, 
    v.annual_mileage, 
    v.year_of_manufacture, 
    v.registration_date, 
    v.car_group, 
    v.vehicle_value, 
    v.vehicle_registration,
    v.purchase_date from prod_adp_certified.policy_motor.vehicle v LEFT JOIN prod_adp_certified.reference_motor.vehicle_overnight_location vo ON v.vehicle_overnight_location_code = vo.vehicle_overnight_location_code""")

# COMMAND ----------

excess = spark.sql(""" select 
                   policy_transaction_id,
                   voluntary_amount
                   from prod_adp_certified.policy_motor.excess WHERE excess_index = 0""")

# COMMAND ----------

driver = spark.sql(""" select
    pd.policy_transaction_id,
    pd.first_name ,
    pd.last_name, 
    pd.date_of_birth,
    --pd.driving_licence_number,
    pd.additional_vehicles_owned, 
    pd.age_at_policy_start_date, 
    pd.cars_in_household, 
    pd.licence_length_years, 
    pd.years_resident_in_uk,
    do.occupation_code as employment_type_abi_code,
    ms.marital_status_code,
    ms.marital_status_name
    from prod_adp_certified.policy_motor.driver pd
    LEFT JOIN prod_adp_certified.policy_motor.driver_occupation do
    ON pd.policy_transaction_id = do.policy_transaction_id
    AND pd.driver_index = do.driver_index
    LEFT JOIN prod_adp_certified.reference_motor.marital_status ms ON pd.marital_status_code = ms.marital_status_id 
    WHERE do.occupation_index = 1
    ORDER BY pd.policy_transaction_id,pd.driver_index"""
    ).dropDuplicates()

driver_transformed = driver.groupBy("policy_transaction_id").agg(
    F.collect_list("first_name").alias("first_name"),
    F.collect_list("last_name").alias("last_name"),
    F.collect_list("date_of_birth").alias("date_of_birth"),
    F.collect_list("marital_status_code").alias("marital_status_code"),
    F.collect_list("marital_status_name").alias("marital_status_name"),
    F.collect_list("additional_vehicles_owned").alias("additional_vehicles_owned"),
    F.collect_list("age_at_policy_start_date").alias("age_at_policy_start_date"),
    F.collect_list("cars_in_household").alias("cars_in_household"),
    F.collect_list("licence_length_years").alias("licence_length_years"),
    F.collect_list("years_resident_in_uk").alias("years_resident_in_uk"),
    F.collect_list("employment_type_abi_code").alias("employment_type_abi_code")
)


max_list_size = driver_transformed.select(
    *[F.size(F.col(col)).alias(col) for col in driver_transformed.columns if col != "policy_transaction_id"]
).agg(F.max(F.greatest(*[F.col(col) for col in driver_transformed.columns if col != "policy_transaction_id"]))).collect()[0][0]

# Dynamically explode each list into individual columns
columns_to_explode = [col for col in driver_transformed.columns if col != "policy_transaction_id"]
for col in columns_to_explode:
    for i in range(max_list_size):
        driver_transformed = driver_transformed.withColumn(
            f"{col}_{i+1}",
            F.col(col)[i]
        )
# Drop the original list columns
driver_transformed = driver_transformed.drop(*columns_to_explode)

# COMMAND ----------

customer = spark.sql(""" select c.customer_focus_id,c.customer_focus_version_id,c.home_email, 
    c.postcode from
    prod_adp_certified.customer_360.single_customer_view c
""")

# COMMAND ----------

policy_transaction.join(customer, (customer.customer_focus_id == policy_transaction.customer_focus_id) & (customer.customer_focus_version_id == policy_transaction.customer_focus_version_id), "left").createOrReplaceTempView("policy_transaction_customer")

# COMMAND ----------

svi_claims = spark.sql("""
        SELECT 
            c.claim_id,
            c.policy_number, i.reported_date
        FROM prod_adp_certified.claim.claim_version cv
        LEFT JOIN
            prod_adp_certified.claim.incident i
        ON i.event_identity = cv.event_identity
        LEFT JOIN
            prod_adp_certified.claim.claim c
        ON cv.claim_id = c.claim_id
        WHERE
        incident_cause IN ('Animal', 'Attempted To Avoid Collision', 'Debris/Object', 'Immobile Object', 'Lost Control - No Third Party Involved')
""" ).filter(
    F.to_date(F.col('reported_date').substr(1, 10), 'yyyy-MM-dd') > F.to_date(F.lit(this_day), 'yyyy-MM-dd')
)

# COMMAND ----------

latest_claims = spark.sql("""
  select 
    max(claim_version_id) claim_version_id,
    claim_id
  from prod_adp_certified.claim.claim_version
  group by claim_id
""")

svi_claims = svi_claims.join(latest_claims, ["claim_id"], "left")

# COMMAND ----------

policy_svi = (
    svi_claims
    .join(policy, ['policy_number'], "left")
    .join(policy_transaction, ['policy_number'], "left")
    .join(vehicle, ['policy_transaction_id'], "left")
    .join(excess, ['policy_transaction_id'], "left")
    .join(driver_transformed, ['policy_transaction_id'], "left")
    .join(
        customer,
        ['customer_focus_id', 'customer_focus_version_id'],
        "left"
    )
    .drop_duplicates()
)


# COMMAND ----------

policy_svi.write \
    .mode("overwrite") \
    .format("delta").option("overwriteSchema", "true") \
    .saveAsTable(f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_policy_svi")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The End
