# Databricks notebook source
# MAGIC %md
# MAGIC Script to read claim variables from source tables (ADP)
# MAGIC
# MAGIC - Data from 2023 onwards is used to match ADP/DSP data range
# MAGIC - Not all policies are in policy.policy table (hence no start date etc for such policies)
# MAGIC - First party field does not seem properly populated: so using claim_version_item_index=0 in addtion
# MAGIC #SVI Data Build & Quality Checks: Main Notebook
# MAGIC
# MAGIC This notebook is split into parts: 
# MAGIC 1. Reading data from relevant sources
# MAGIC 2. Row filtering
# MAGIC 3. Deduplicating
# MAGIC 4. Joining datasets
# MAGIC 5. Initial Feature Engineering
# MAGIC 6. Quality checks
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###1. Reading Data
# MAGIC
# MAGIC There are three main data sources currently: SAS, the datalake & the azure data platform (ADP)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Get claim referral log data

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql import Window

# COMMAND ----------


def get_referral_vertices(df): 
  #process claim referral log
  df = df.withColumn("Claim Ref", regexp_replace("Claim Ref", "\\*", "")) \
    .withColumn("siu_investigated", when(col("Source of referral") == "SIU",1)
                                 .otherwise(0))
  
  #create indicator fraud investigation risk
  risk_cols = {"Final Outcome of Claim": ["Withdrawn whilst investigation ongoing", "Repudiated – Litigated – Claim then discontinued", "Repudiated – Litigated – Success at trial", "Repudiated – Not challenged"]
  }
  risk_cols["Outcome of Referral"] = ["Accepted"]
  risk_cols["Outcome of investigation"] = ["Repudiated", "Repudiated in part", "Under Investigation", "Withdrawn whilst investigation ongoing"]

  for this_col in risk_cols:
    df = df.withColumn(f'{this_col}_risk', 
                                      col(this_col).isin(*risk_cols[this_col]).cast('integer')) 

  df = df.fillna({"Final Outcome of Claim_risk": 0, 
                  "Outcome of Referral_risk": 0, 
                  "Outcome of investigation_risk": 0})
                                   
  df = df.withColumn("fraud_risk", greatest("Final Outcome of Claim_risk", "Outcome of Referral_risk", "Outcome of investigation_risk"))

  referral_vertices = df.select(
    col("Claim Ref").alias("id"), 
    "siu_investigated", 
    "fraud_risk", "Final Outcome of Claim_risk", 
    "Outcome of Referral_risk", "Outcome of investigation_risk",
    col("Concerns").alias("referral_concerns"),
    col("Date received").alias("transact_time"),
    col("Date received").alias("referral_log_date"),
    col("Date of Outcome").alias("referral_outcome_date")
  )
  return referral_vertices

start_clm_log = "2023-01-01"

#claim referral log
clm_log_df =  spark.sql("""
    SELECT DISTINCT * 
    FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claim_referral_log
    """).filter( col("Date received") >= start_clm_log)
clm_log = get_referral_vertices(clm_log_df).filter(lower(col("id")).contains("fc/")).select("id", "fraud_risk")
clm_log.createOrReplaceTempView("clm_log")

display(clm_log)


# COMMAND ----------

clm_log.createOrReplaceGlobalTempView("global_temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. Row Filtering
# MAGIC
# MAGIC Examples include filtering for a specific date range, for certain policy or claim statuses, for renewals/new business, etc

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get target variable (Claims Interview **Outcome**)

# COMMAND ----------

# filter for only cases with known outcomes (SVI reports or claim referral log)
'''

'''
# fa_risk: risk outcome after Fraud Analysis Team's review
# tbg_risk: risk outcome after external contactor (Brownsword)'s review
# fraud_risk: risk outcome from claim-referral log (Sharepoint)

df = spark.sql(
"""
SELECT DISTINCT

svi.`Claim Number` as claim_number, 
svi.`Result of Outsourcing` as TBG_Outcome, 
svi.`FA Outcome` as FA_Outcome,
log.fraud_risk,

CASE WHEN lower(svi.`Result of Outsourcing`) = 'settled' THEN 0 
    WHEN  lower(svi.`Result of Outsourcing`) IN ('withdrawn', 'repudiated', 'managed away', 'cancelled') THEN 1
END AS tbg_risk,
CASE WHEN  lower(svi.`FA Outcome`) IN ('claim closed', "claim to review", 'not comprehensive cover') THEN 1 ELSE 0 
END AS fa_risk
FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_performance svi
LEFT JOIN clm_log log
ON lower(svi.`Claim Number`) = lower(log.id)
WHERE svi.`Notification Date` >= '2023-01-01'
AND (lower(svi.`Result of Outsourcing`) IS NULL OR lower(svi.`Result of Outsourcing`) NOT IN ('ongoing - client', 'ongoing - tbg', 'pending closure'))
AND lower(svi.`FA Outcome`) != 'not comprehensive cover'
""")

#claim is high risk if flagged at either stages
target_df = df.withColumn(
    "svi_risk", greatest(col("fraud_risk"), col("tbg_risk"))
).fillna({"svi_risk": -1})

display(target_df.groupBy("fa_risk").count().orderBy("fa_risk"))
display(target_df.groupBy("TBG_Outcome").count().orderBy("TBG_Outcome"))
display(target_df.groupBy("svi_risk").count().orderBy("svi_risk"))
# CASE WHEN  lower(svi.`FA Outcome`) IN ('claim closed', "claim to review") THEN 1 ELSE 0 
#END AS fa_risk


# COMMAND ----------

# MAGIC %md
# MAGIC #### Get latest claim version

# COMMAND ----------

#get the latest claim_version_id for each claim

spark.sql('USE CATALOG prod_adp_certified')

target_df.createOrReplaceTempView("target_df")

latest_claim_version = spark.sql(
    """
    SELECT DISTINCT
        MAX(cv.claim_number) AS claim_number,
        MAX(svi.svi_risk) AS svi_risk, 
        MAX(svi.tbg_risk) AS tbg_risk, 
        MAX(svi.FA_Outcome) AS FA_Outcome, 
        MAX(svi.fa_risk) AS fa_risk,
        MAX(svi.fraud_risk) AS fraud_risk, 
        MAX(cv.claim_version_id) AS claim_version_id,
        cv.claim_id,
        MAX(cv.event_enqueued_utc_time) AS latest_event_time
    FROM target_df svi
    LEFT JOIN prod_adp_certified.claim.claim_version cv
    ON LOWER(cv.claim_number) = LOWER(svi.claim_number)
    GROUP BY cv.claim_id
    HAVING claim_number IS NOT NULL
    """
)
latest_claim_version.createOrReplaceTempView("latest_claim_version")

#display(latest_claim_version) #.limit(10))
#AND month(latest_event_time) = 3
#AND year(latest_event_time) = 2023


# COMMAND ----------

damage_list=['claim.damage_details.front_severity', 'claim.damage_details.front_bonnet_severity', 'claim.damage_details.front_left_severity', 'claim.damage_details.front_right_severity', 'claim.damage_details.left_severity', 'claim.damage_details.left_back_seat_severity', 'claim.damage_details.left_front_wheel_severity', 'claim.damage_details.left_mirror_severity', 'claim.damage_details.left_rear_wheel_severity', 'claim.damage_details.left_underside_severity', 'claim.damage_details.rear_severity', 'claim.damage_details.rear_left_severity', 'claim.damage_details.rear_right_severity', 'claim.damage_details.rear_window_damage_severity', 'claim.damage_details.right_severity', 'claim.damage_details.right_back_seat_severity', 'claim.damage_details.right_front_wheel_severity', 'claim.damage_details.right_mirror_severity', 'claim.damage_details.right_rear_wheel_severity', 'claim.damage_details.right_roof_severity', 'claim.damage_details.right_underside_severity', 'claim.damage_details.roof_damage_severity', 'claim.damage_details.underbody_damage_severity', 'claim.damage_details.windscreen_damage_severity']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Join claim tables and get variables

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
claim.claimant.is_first_party,
incident.event_identity as incident_event_identity,
lcv.latest_event_time,
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
claim.driver.date_of_birth as claim_driver_dob,
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
claim.damage_details.front_severity, claim.damage_details.front_bonnet_severity, claim.damage_details.front_left_severity, claim.damage_details.front_right_severity, claim.damage_details.left_severity, claim.damage_details.left_back_seat_severity, claim.damage_details.left_front_wheel_severity, claim.damage_details.left_mirror_severity, claim.damage_details.left_rear_wheel_severity, claim.damage_details.left_underside_severity, claim.damage_details.rear_severity, claim.damage_details.rear_left_severity, claim.damage_details.rear_right_severity, claim.damage_details.rear_window_damage_severity, claim.damage_details.right_severity, claim.damage_details.right_back_seat_severity, claim.damage_details.right_front_wheel_severity, claim.damage_details.right_mirror_severity, claim.damage_details.right_rear_wheel_severity, claim.damage_details.right_roof_severity, claim.damage_details.right_underside_severity, claim.damage_details.roof_damage_severity, claim.damage_details.underbody_damage_severity, claim.damage_details.windscreen_damage_severity,
lcv.tbg_risk, lcv.fraud_risk, lcv.svi_risk, lcv.FA_Outcome, lcv.fa_risk
FROM latest_claim_version lcv
INNER JOIN claim.claim_version
ON lcv.claim_number = claim_version.claim_number 
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
AND claim_version_item.claim_item_type='CarMotorVehicleClaimItem'
AND claim_version_item.claim_version_item_index=0
AND year(incident.start_date)>=2023
"""
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate damaged areas

# COMMAND ----------



# function to calculate the damage score
def calculate_damage_score(*args):
    damageScore = 1
    areasDamagedMinimal = 0
    areasDamagedMedium = 0
    areasDamagedHeavy = 0
    areasDamagedSevere = 0
    
    for damage in args:
        if damage == 'Minimal':
            damageScore *= 2
            areasDamagedMinimal += 1
        elif damage == 'Medium':
            damageScore *= 3
            areasDamagedMedium += 1
        elif damage == 'Heavy':
            damageScore *= 4
            areasDamagedHeavy += 1
        elif damage == 'Severe':
            damageScore *= 5
            areasDamagedSevere += 1
    
    return damageScore, areasDamagedMinimal, areasDamagedMedium, areasDamagedHeavy, areasDamagedSevere

# Register the UDF
calculate_damage_score_udf = udf(calculate_damage_score, StructType([
    StructField("damageScore", IntegerType(), False),
    StructField("areasDamagedMinimal", IntegerType(), False),
    StructField("areasDamagedMedium", IntegerType(), False),
    StructField("areasDamagedHeavy", IntegerType(), False),
    StructField("areasDamagedSevere", IntegerType(), False)
]))


# List of damage columns
damage_columns = [
    'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 
    'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 
    'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 
    'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
    'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 
    'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity'
]


# Apply the UDF to the DataFrame
check_df = check_df.withColumn(
    "damage_scores",
    calculate_damage_score_udf(*[check_df[col] for col in damage_columns])
)

# Split the struct column into separate columns
check_df = check_df.select(
    "*",
    "damage_scores.damageScore",
    "damage_scores.areasDamagedMinimal",
    "damage_scores.areasDamagedMedium",
    "damage_scores.areasDamagedHeavy",
    "damage_scores.areasDamagedSevere"
).withColumn("areasDamagedTotal", col("areasDamagedMinimal") + col("areasDamagedMedium") + col("areasDamagedSevere") + col("areasDamagedHeavy"))\
.withColumn("veh_age", round(datediff(col("start_date"), to_date(concat(col("manufacture_yr_claim"), lit('-01-01')))) / 365.25, 0))\
.withColumn("veh_age_more_than_10", (col("veh_age") > 10).cast("int"))\
.withColumn("claim_driver_age",
    round(datediff(col("start_date"), to_date(col("claim_driver_dob"))) / 365.25))\
.drop("damage_scores")


#display(check_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Deduplicating
# MAGIC
# MAGIC Duplicates can exist either across all columns (where a complete row is duplicated) or across a subset of column (e.g. where a Policy number has multiple rows but with different values associated in other column(s)). 
# MAGIC
# MAGIC The former can simply be removed by calling df.dropDuplicates(), while the latter requires more complex deduplicating. <br>
# MAGIC The functions below help you find those duplicates and offer some ideas on how to deal with them. Examples include deduplicating by taking minimum, maximum or sum of the columns causing the duplicates, or by taking the latest record based on a subset of id columns. You can also add custom code to deal with more complicated duplicates.
# MAGIC

# COMMAND ----------

#dedup driver features
# Get the minimum claim_driver_age for each claim_number
min_drv_age = check_df.groupBy("claim_number").agg(
    min(col("claim_driver_age")).alias("min_claim_driver_age")
)

# Join the min_drv_age DataFrame back to the original check_df
check_df = check_df.drop("claim_driver_age").join(min_drv_age, on="claim_number", how="left").drop("driver_id","claim_driver_dob").dropDuplicates()


# COMMAND ----------

# MAGIC %md
# MAGIC ###4. Joining Datasets
# MAGIC
# MAGIC Ideally, all your datasets share one or more primary keys to join on (e.g. policy number, claim number, VRN, etc), which you can use for straight forward join. <br>
# MAGIC
# MAGIC The names of the columns may not always be the same between datasets, in which case I've found it easier to rename the columns before joining rather than specifying the left and right join columns separately. You can rename columns in a spark dataframe by using df.withColumnRenamed('old_colname', 'new_colname').
# MAGIC
# MAGIC You may want to check for duplicates again after the join using the functions above.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get policy variables

# COMMAND ----------

#join check_df to prod_dsexp_auxiliarydata.single_vehicle_incident_checks.policy_svi using policy_number
# note: marital status not populated, so not used

pol_cols = ['policy_transaction_id', 'policy_number', 'quote_session_id', 'policy_start_date', 'policy_renewal_date', 'policy_type', 'policyholder_ncd_years', 'ncd_protected_flag', 'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_id', 'vehicle_overnight_location_name', 'business_mileage', 'annual_mileage', 'year_of_manufacture', 'registration_date', 'car_group', 'vehicle_value', 'purchase_date', 'voluntary_amount', 'date_of_birth_1', 'additional_vehicles_owned_1', 'additional_vehicles_owned_2', 'additional_vehicles_owned_3', 'additional_vehicles_owned_4', 'additional_vehicles_owned_5', 'age_at_policy_start_date_1', 'age_at_policy_start_date_2', 'age_at_policy_start_date_3', 'age_at_policy_start_date_4', 'age_at_policy_start_date_5', 'cars_in_household_1', 'cars_in_household_2', 'cars_in_household_3', 'cars_in_household_4', 'cars_in_household_5', 'licence_length_years_1', 'licence_length_years_2', 'licence_length_years_3', 'licence_length_years_4', 'licence_length_years_5', 'years_resident_in_uk_1', 'years_resident_in_uk_2', 'years_resident_in_uk_3', 'years_resident_in_uk_4', 'years_resident_in_uk_5', 'employment_type_abi_code_1', 'employment_type_abi_code_2', 'employment_type_abi_code_3', 'employment_type_abi_code_4', 'employment_type_abi_code_5', 'postcode']

policy_svi = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.policy_svi")\
                    .select(pol_cols)
                    
policy_svi.createOrReplaceTempView("policy_svi")
#add vehicle use etc from quotes
quote_iteration_df = spark.table("prod_adp_certified.quote_motor.quote_iteration")
vehicle_df = spark.table("prod_adp_certified.quote_motor.vehicle")

policy_svi = policy_svi.join(
    quote_iteration_df, policy_svi.quote_session_id == quote_iteration_df.session_id, "left"
    ).join(vehicle_df, "quote_iteration_id", "left"
    ).select(
        "policy_svi.*",
        quote_iteration_df.session_id,
        (vehicle_df.vehicle_use_code).alias("vehicle_use_quote"),
        quote_iteration_df.quote_iteration_id
    )

# Specify window for max transaction id per policy
window_spec = Window.partitionBy(col("policy_number")).orderBy(col("policy_transaction_id").desc())

#filter for the latest (max) transaction id 
policy_svi = policy_svi.withColumn("row_num", row_number().over(window_spec)).filter(col("row_num") == 1).drop("row_num")

policy_svi.createOrReplaceTempView("policy_svi")

driver_cols = ['additional_vehicles_owned', 'age_at_policy_start_date', 'cars_in_household', 'licence_length_years', 'years_resident_in_uk']

for col_name in driver_cols:
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

#display(check_df)
drop_cols = ['additional_vehicles_owned_2', 'additional_vehicles_owned_3', 'additional_vehicles_owned_4', 'additional_vehicles_owned_5', 'age_at_policy_start_date_2', 'age_at_policy_start_date_3', 'age_at_policy_start_date_4', 'age_at_policy_start_date_5', 'cars_in_household_2', 'cars_in_household_3', 'cars_in_household_4', 'cars_in_household_5', 'licence_length_years_2', 'licence_length_years_3', 'licence_length_years_4', 'licence_length_years_5', 'years_resident_in_uk_2', 'years_resident_in_uk_3', 'years_resident_in_uk_4', 'years_resident_in_uk_5']

policy_svi = policy_svi.drop(*drop_cols)

check_df = check_df.join(policy_svi, on="policy_number", how="left")
# filter for claims with only matched policies
check_df = check_df.filter(col("policy_transaction_id").isNotNull()).dropDuplicates()

#display(check_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###5. Feature Engineering
# MAGIC
# MAGIC Here you can create new features from your joined dataset and also define your target variable if not already in the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate check variables

# COMMAND ----------


#C1: was the incident on a Friday/Saturday *NIGHT*?
check_df = check_df.withColumn("incident_day_of_week", date_format(col("latest_event_time"), "E"))

fri_sat_night = ((col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4))))
                                                                                                                    
check_df = check_df.withColumn(
    "C1_fri_sat_night",
    when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0))

check_df = check_df.withColumn("reported_day_of_week", date_format(col("latest_event_time"), "E"))

#C2: Was there a delay in notifying us of the incident without reason?
check_df = check_df.withColumn("delay_in_reporting", datediff(col("reported_date"), col("start_date")))
check_df = check_df.withColumn("C2_reporting_delay", when(col("delay_in_reporting")>=3, 1).when(col("delay_in_reporting").isNull(), 1).otherwise(0))

# Add a column to check if the incident date is on a weekend
check_df = check_df.withColumn(
    "is_incident_weekend",
    when(date_format(col("start_date"), "E").isin("Fri", "Sat", "Sun"), 1).otherwise(0)
)

# Add a column to check if the reported date is on a Monday
check_df = check_df.withColumn(
    "is_reported_monday",
    when(date_format(col("reported_date"), "E") == "Mon", 1).otherwise(0)
)

#C3: Cases taking place over a weekend but not being reported until Monday
check_df = check_df.withColumn(
    "C3_weekend_incident_reported_monday",
    when((col("is_incident_weekend") == True) & (col("is_reported_monday") == True), 1).otherwise(0)
)

#c5: Incident between 11pm and 5am
check_df = check_df.withColumn(
    "C5_is_night_incident",
    when((hour(col("start_date")) >= 23) | (hour(col("start_date")) <= 5) | (hour(col("start_date"))).isNull(), 1).otherwise(0)
)

# C6 No commuting on policy and customer travelling between the hours of 6am and 10am or 3pm and 6pm?
not_commuting_rush = (lower(col("vehicle_use_quote")) == 1) & ((hour(col("start_date")).between(6, 10)) | (hour(col("start_date")).between(15, 18)))
check_df = check_df.withColumn(
    "C6_no_commuting_but_rush_hour",
    when(not_commuting_rush, 1
    )
    .when(not_commuting_rush.isNull(), 1
    ).otherwise(0)
)

#C7	Notified of a incident/CRN from the PH relating to the police attending the scene? (low risk)
check_df = check_df.withColumn(
    "C7_police_attended_or_crime_reference",
    when((col("is_police_attendance") == 1) | (col("is_crime_reference_provided") == True), 1).otherwise(0)
)

#C9: Was the policy incepted within 30 days of the incident date?
check_df = check_df.withColumn("inception_to_claim", datediff(to_date(col("start_date")), to_date(col("policy_start_date"))))

check_df = check_df.withColumn(
    "C9_policy_within_30_days",
    when(col("inception_to_claim").between(0, 30),1).when(col("inception_to_claim").isNull(), 1).otherwise(0)
)

#C10: Does the policy end within 1 or 2 months of the incident date?
check_df = check_df.withColumn("claim_to_policy_end", datediff(to_date(col("policy_renewal_date")), to_date(col("start_date"))))

check_df = check_df.withColumn(
    "C10_claim_to_policy_end",
        when(col("claim_to_policy_end")<60, 1). when(col("claim_to_policy_end").isNull(), 1).otherwise(0)
        )

check_df = check_df.withColumn( "driver_age_low_1", when(col("age_at_policy_start_date_1")<25, 1)
                               .when(col("age_at_policy_start_date_1").isNull(), 1).otherwise(0)
                               )
check_df = check_df.withColumn( "claim_driver_age_low", when(col("min_claim_driver_age")<25, 1)
                               .when(col("min_claim_driver_age").isNull(), 1).otherwise(0))

#TODO: check licence low threshold
check_df = check_df.withColumn( "licence_low_1", when(col("licence_length_years_1")<=3, 1).otherwise(0))

#C11	Are they classed as young/inexperienced ie under 25 or new to driving
condition_inexperienced = (col("driver_age_low_1") == 1) | (col("licence_low_1") == 1) 
check_df = check_df.withColumn( "C11_young_or_inexperienced", when(condition_inexperienced, 1)
                               .when(condition_inexperienced.isNull(), 1)
                               .otherwise(0))

# C12	Age in comparison to the type of vehicle (Value wise). Thresholds by business unit
# col("vehicle_value")/col("age_at_policy_start_date_1")
#>20k car for <25 years old, >30K car for >=25 years old (low priority flag)

condition_expensive_car =  ((col("age_at_policy_start_date_1") < 25) & (col("vehicle_value") >= 20000)) | ( (col("age_at_policy_start_date_1") >= 25) &(col("vehicle_value") >= 30000))

check_df = check_df.withColumn( "C12_expensive_for_driver_age", when(condition_expensive_car, 1)
                    .when(condition_expensive_car.isNull(), 1)
                    .otherwise(0))

# Create a regex pattern from the watch words
watch_words = "|".join(["commut", "deliver", "parcel", "drink", "police", "custody", "arrest", 
                        "alcohol", "drug", "station", "custody"])

# Add a column to check if Circumstances contains any of the items in list_a
check_df = check_df.withColumn(
    "C14_contains_watchwords",
    when(lower(col("Circumstances")).rlike(watch_words), 1)
    .when(col("Circumstances").isNull(), 1).otherwise(0)
)

boolean_columns = [ "vehicle_unattended", "excesses_applied", "is_first_party",     "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", "is_ambulance_attendance",     "is_fire_service_attendance", "is_police_attendance" ]

for col_name in boolean_columns:
    check_df = check_df.withColumn(col_name, col(col_name).cast("integer"))

#drop more columns
more_drops = ['driver_id', 'incident_location_longitude', 'purchase_date',
'registration_date', 'not_on_mid']
check_df = check_df.drop(*more_drops)

#fix issue with decimal type
decimal_cols = ['outstanding_finance_amount', 'vehicle_value', 'voluntary_amount']
for col_name in decimal_cols:
    check_df = check_df.withColumn(col_name, col(col_name).cast("float"))

# COMMAND ----------

check_df

# COMMAND ----------

filtered_df = check_df[check_df.claim_number == "FC/901136611"][sorted(check_df.columns)]
display(filtered_df)

# COMMAND ----------

melted_df = (
    filtered_df
    .selectExpr("stack({}, {}) as (column, value)".format(
        len(filtered_df.columns),
        ", ".join([f"'{c}', CAST(`{c}` AS STRING)" for c in filtered_df.columns])
    ))
)

display(melted_df)

# COMMAND ----------

check_df.createOrReplaceGlobalTempView("global_temp_view")

# COMMAND ----------

# columns to fill using mean
mean_fills = [ "policyholder_ncd_years", "inception_to_claim", "claim_driver_age", "veh_age", "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end"]

#boolean or damage columns with neg fills
neg_fills = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance","is_police_attendance","veh_age_more_than_10","damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal","police_considering_actions","is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved",  "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low","licence_low_1", ]

# fills with ones (rules variables, to trigger manual check)
one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_not_comprehensive_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords",]

#fill with word 'missing' (categoricals) 
string_cols = [
    'car_group', 'vehicle_overnight_location_id', 'incidentDayOfWeekC', 'incidentMonthC', 
    'employment_type_abi_code_5', 'employment_type_abi_code_4', 'employment_type_abi_code_3', 
    'employment_type_abi_code_2', 'policy_type', 'postcode', 'assessment_category', 'engine_damage', 
    'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 
    'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 'incident_cause', 
    'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 
    'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 
    'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 
    'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
    'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 
    'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 
    'employment_type_abi_code_1', 'incident_day_of_week', 'reported_day_of_week'
]

other_cols = ['claim_number', 'svi_risk', 'policy_number', 'policy_transaction_id',  'start_date', 'policy_start_date', 'fa_risk', 'fraud_risk', 'tbg_risk']

useful_cols = other_cols + mean_fills + neg_fills + one_fills + string_cols

print(useful_cols)

##fillna damage columns with 0
#damage_fills = {x:-1 for x in damage_columns}
#check_df = check_df.fillna(damage_fills)

# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

#split to train/test and tag accordingly
df_risk_pd = check_df.coalesce(1).toPandas()
train_df, test_df = train_test_split(df_risk_pd, test_size=0.3, random_state=42, stratify=df_risk_pd.svi_risk)
train_df['dataset']= 'train'
test_df['dataset']= 'test'

combined_df_pd = pd.concat([test_df, train_df])
check_df = spark.createDataFrame(combined_df_pd)


# COMMAND ----------

# Show data quality summary
print(f"Total records: {check_df.count()}")
print(f"Train records: {check_df.filter(col('dataset') == 'train').count()}")
print(f"Test records: {check_df.filter(col('dataset') == 'test').count()}")

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

# COMMAND ----------

#uncomment to write to ADP aux catalog

check_df.write \
    .mode("overwrite") \
    .format("delta").option("mergeSchema", "true") \
    .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi")

#.option("mergeSchema", "true")#overwriteSchema

# COMMAND ----------

dbutils.notebook.exit("Notebook stopped.")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ###6. Quality Checks
# MAGIC

# COMMAND ----------

# Check datatypes
# These will be looked at in more detail in the EDA but a sense check here is probably a good idea
check_df.dtypes

# COMMAND ----------

# Volumes over time
display(check_df\
    .groupby(year(col('start_date')).alias('Year'), month(col('start_date')).alias('Month'))\
    .count()\
    .orderBy('Year', 'Month'))

# COMMAND ----------

# Missing Values of time
display(check_df\
    .filter(col('vehicle_value').isNull())\
    .groupby(year(col('start_date')).alias('Year'), month(col('start_date')).alias('Month'))\
    .count()\
    .orderBy('Year', 'Month'))

# COMMAND ----------

display(check_df.select(sorted(check_df.columns)))

# COMMAND ----------


# Calculate the percentage of nulls and empty columns
null_counts = check_df.select(
    [(count(when(col(c).isNull() | (col(c) == ""), c)) / count("*")).alias(c) for c in check_df.columns]
)

# Display the result
null_counts_pd = null_counts.toPandas()*100
display(null_counts_pd.T.reset_index())


# COMMAND ----------

#define functions
def get_info(df): 
  return pd.DataFrame({"name": df.columns,\
    "% nulls": 100*df.isnull().sum().values/len(df),\
    "type": df.dtypes.astype(str),
    "nulls": df.isnull().sum().values,     
    "non-nulls": len(df)-df.isnull().sum().values, \
    "min": df.apply(pd.to_numeric, args=['coerce']).min(axis=0).astype(float),\
    "max": df.apply(pd.to_numeric, args=['coerce']).max(axis=0).astype(float),
    "median": df.apply(pd.to_numeric, args=['coerce']).median(axis=0),        
    "mean": df.apply(pd.to_numeric, args=['coerce']).mean(axis=0),        
    "std": df.apply(pd.to_numeric, args=['coerce']).std(axis=0),        
    "mode": df.mode(axis=0).iloc[0,:].astype(str),  
    "unique": df.nunique(axis=0)      
  })

check_pd = check_df.toPandas()
cdf = get_info(check_pd)
display(cdf)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
