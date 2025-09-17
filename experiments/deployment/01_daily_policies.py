# Databricks notebook source
# MAGIC  %md
# MAGIC ## SVI: Extract policy features for single-vehicle claims

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import collect_list

# COMMAND ----------



# COMMAND ----------

this_day = dbutils.widgets.get("date_range")

#this_day = '2025-03-01'

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
    #F.collect_list("driving_licence_number").alias("driving_licence_number"),
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

# Show the transformed DataFrame
#driver_transformed.display()

# COMMAND ----------

'''
customer = spark.sql(""" select c.customer_focus_id,c.customer_focus_version_id,c.home_email, 
    ca.postcode from
    prod_adp_certified.customer.customer c
LEFT JOIN 
    prod_adp_certified.customer.customer_address ca
    ON c.customer_focus_id = ca.customer_focus_id
    AND c.customer_focus_version_id = ca.customer_focus_version_id""")
'''

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
""" ).filter(f"DATE(reported_date)='{this_day}'")

#display(svi_claims.select(sorted(svi_claims.columns)).limit(10))

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

#display(policy_svi.limit(10))


# COMMAND ----------

'''
latest_event_enqueued = spark.sql("""
    SELECT MAX(transaction_timestamp) AS latest_event_enqueued
    FROM prod_adp_certified.policy_motor.policy_transaction
""")

display(latest_event_enqueued)
'''

# COMMAND ----------


policy_svi.write \
    .mode("overwrite") \
    .format("delta").option("mergeSchema", "true") \
    .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_policy_svi")

