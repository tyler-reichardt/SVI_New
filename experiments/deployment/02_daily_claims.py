# Databricks notebook source
# MAGIC %md
# MAGIC #Daily Claims Fetching
# MAGIC This notebook is split into parts: 
# MAGIC 1. Reading data from relevant sources
# MAGIC 2. Row filtering
# MAGIC 3. Deduplicating
# MAGIC 4. Joining datasets with policy table
# MAGIC 5. Saving result to table
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###1. Reading Data
# MAGIC
# MAGIC Read relevant policies

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. Row Filtering
# MAGIC
# MAGIC Examples include filtering for a specific date range, for certain policy or claim statuses, for renewals/new business, etc

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get latest claim version

# COMMAND ----------

#get the latest claim_version_id for each claim

this_day = dbutils.widgets.get("date_range")

#this_day = '2025-03-01'

spark.sql('USE CATALOG prod_adp_certified')

policy_svi = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_policy_svi")\
                    .filter(f"DATE(reported_date)='{this_day}'")
policy_svi.createOrReplaceTempView("policy_svi")

latest_claim_version = policy_svi.selectExpr('claim_id', 'claim_version_id')

latest_claim_version.createOrReplaceTempView("latest_claim_version")

#print(sorted(policy_svi.columns))
#print(policy_svi.select("*"))

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
claim.claim_version.position_status,
claim.claim_version_item.claim_item_type, 
claim.claim_version_item.not_on_mid, 
claim.claim_version_item.vehicle_unattended,
claim.claim_version_item.excesses_applied,
claim.claim_version_item.total_loss_date, 
claim.claim_version_item.total_loss_flag,
claim.claim_version_item.first_party as cvi_first_party,
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

#AND claim_version_item.claim_item_type='CarMotorVehicleClaimItem'

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

#display(check_df.limit(10))

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

#add vehicle use etc from quotes
quote_iteration_df = spark.table("prod_adp_certified.quote_motor.quote_iteration")
vehicle_df = spark.table("prod_adp_certified.quote_motor.vehicle").selectExpr("quote_iteration_id", "vehicle_use_code AS vehicle_use_quote")

policy_svi = policy_svi.join(
    quote_iteration_df.select("session_id", "quote_iteration_id"),
    policy_svi.quote_session_id == quote_iteration_df.session_id, "left"
    ).join(vehicle_df, "quote_iteration_id", "left"
    ).select("*")

# Specify window for max transaction id per policy
window_spec = Window.partitionBy(col("policy_number")).orderBy(col("policy_transaction_id").desc())

#filter for the latest (max) transaction id 
policy_svi = policy_svi.withColumn("row_num", row_number().over(window_spec)).filter(col("row_num") == 1).drop("row_num")

policy_svi.createOrReplaceTempView("policy_svi")

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



# COMMAND ----------

# MAGIC %md
# MAGIC ###5. Feature Engineering
# MAGIC
# MAGIC Here you can create new features from your joined dataset and also define your target variable if not already in the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate check variables

# COMMAND ----------



# COMMAND ----------


#C1: was the incident on a Friday/Saturday *NIGHT*?
check_df = check_df.withColumn("incident_day_of_week", date_format(col("reported_date"), "E"))

fri_sat_night = ((col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4))))
                                                                                                                    
check_df = check_df.withColumn(
    "C1_fri_sat_night",
    when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0))

check_df = check_df.withColumn("reported_day_of_week", date_format(col("reported_date"), "E"))

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
    when((col("is_police_attendance") == True) | (col("is_crime_reference_provided") == True), 1).otherwise(0)
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

other_cols = ['claim_number', 'policy_number', 'policy_transaction_id',  'start_date', 'policy_start_date', 'Circumstances', 'position_status']

useful_cols = other_cols + mean_fills + neg_fills + one_fills + string_cols

#print(useful_cols)

# COMMAND ----------

#uncomment to write to ADP aux catalog

check_df.write \
    .mode("append") \
    .format("delta").option("overwriteSchema", "true") \
    .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_claims_svi")

#.option("mergeSchema", "true")#overwriteSchema, .mode("overwrite") \


# COMMAND ----------

# MAGIC %md
# MAGIC
