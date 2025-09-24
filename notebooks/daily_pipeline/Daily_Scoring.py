# Databricks notebook source
# MAGIC %md
# MAGIC # Model Scoring
# MAGIC 
# MAGIC This notebook template is used for scoring new claims using the trained SVI fraud detection models

# COMMAND ----------

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.inspection import partial_dependence
from collections import Counter
import mlflow
import datetime

# COMMAND ----------

# Get date parameter from widget
this_day = dbutils.widgets.get("date_range")

# Alternative for testing
#this_day = '2025-03-01'

MODEL_VERSION = 0.1  # Change with model update

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Feature Lists and Helper Functions

# COMMAND ----------

# Define feature lists
numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 'incidentDayOfWeekC', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal']

categorical_features = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 'reported_day_of_week', 'checks_max']

check_cols = [
    'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
    'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
    'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
    'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
]

# Columns to fill using mean
mean_fills = ["policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", "incidentDayOfWeekC", "num_failed_checks"]

# Boolean or damage columns with neg fills
damage_cols = ["damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal"]
bool_cols = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance","is_police_attendance","veh_age_more_than_10","police_considering_actions","is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved", "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low","licence_low_1", "total_loss_flag"]

# Fills with ones (rules variables, to trigger manual check)
one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords"]

# Fill with word 'missing' (categoricals) 
string_cols = list(set([
    'car_group', 'vehicle_overnight_location_id', 'incidentMonthC', 
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
]) - set(['car_group', 'employment_type_abi_code_5', 'employment_type_abi_code_4', 'employment_type_abi_code_3', 'employment_type_abi_code_2', 'postcode', 'employment_type_abi_code_1']))

# COMMAND ----------

def set_types(raw_df):
    """Recast data types & check schema"""
    cols_groups = {
        "float": numeric_features,
        "string": categorical_features 
    }
    
    for dtype, column_list in cols_groups.items():
        if dtype == "float":
            raw_df[column_list] = raw_df[column_list].astype(float)
        elif dtype == "integer":
            raw_df[column_list] = raw_df[column_list].astype(int)
        elif dtype == "string":
            raw_df[column_list] = raw_df[column_list].astype('str')    
        elif dtype == "bool":
            raw_df[column_list] = raw_df[column_list].astype(int).astype('str')

    return raw_df

def do_fills_pd(raw_df, mean_dict):
    """Apply missing value imputation"""
    # Recast data types & check schema
    cols_groups = {
        "float": mean_fills + damage_cols,
        "string": string_cols,
        "bool": bool_cols + one_fills 
    }
    
    # Fillna columns
    neg_fills_dict = {x: -1 for x in bool_cols + damage_cols}
    one_fills_dict = {x: 1 for x in one_fills}
    string_fills_dict = {x: 'missing' for x in string_cols}
    combined_fills = {**one_fills_dict, **neg_fills_dict, **string_fills_dict, **mean_dict}
    raw_df = raw_df.fillna(combined_fills)

    for dtype, column_list in cols_groups.items():
        if dtype == "float":
            raw_df[column_list] = raw_df[column_list].astype(float)
        elif dtype == "integer":
            raw_df[column_list] = raw_df[column_list].astype(int)
        elif dtype == "string":
            raw_df[column_list] = raw_df[column_list].astype('str')        
        elif dtype == "bool":
            raw_df[column_list] = raw_df[column_list].astype(int).astype('str')

    return raw_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Interview Model Features

# COMMAND ----------

# Input features for interview model (aligned with experiments/notebooks/04b_Interview_Model.py)
num_interview = ['voluntary_amount', 'policyholder_ncd_years',
       'max_years_resident_in_uk', 'annual_mileage', 'min_claim_driver_age',
       'incidentHourC', 'areasDamagedHeavy', 'impact_speed',
       'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim',
       'claim_to_policy_end', 'veh_age', 'licence_length_years_1',
       'num_failed_checks', 'areasDamagedMedium', 'incidentDayOfWeekC',
       'age_at_policy_start_date_1']

cat_interview = ['assessment_category', 'left_severity', 'C9_policy_within_30_days',
       'incident_sub_cause', 'rear_right_severity', 'front_left_severity',
       'rear_window_damage_severity', 'incident_cause', 'checks_max',
       'total_loss_flag']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Models

# COMMAND ----------

# Load model run IDs (these should be replaced with actual model run IDs)
fa_model_run_id = "56ced60a6d464b07835f5a237425b33b"
interview_run_id = 'b4b7c18076b84880aeea3ff5389c3999'

# Load models from MLflow
fa_pipeline = mlflow.sklearn.load_model(f'runs:/{fa_model_run_id}/model')
interview_pipeline = mlflow.sklearn.load_model(f'runs:/{interview_run_id}/model')

print(f"Loaded FA model from run: {fa_model_run_id}")
print(f"Loaded Interview model from run: {interview_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Read and Scoring

# COMMAND ----------

def load_claims_data(this_day):
    """Load claims data for scoring"""
    table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_claims_svi"
    
    # Read in dataset
    other_cols = ['claim_number', 'reported_date', 'start_date', 'num_failed_checks',
                  'total_loss_flag', 'checks_list', 'delay_in_reporting', 'claim_id', 
                  'position_status', 'vehicle_use_quote', 'Circumstances'] + \
                  [x for x in check_cols if x not in categorical_features]
    
    raw_df = spark.table(table_path).withColumn('checks_max', greatest(*[col(c) for c in check_cols]).cast('string'))\
                    .withColumn("underbody_damage_severity", lit(None))\
                    .filter(f"DATE(reported_date)='{this_day}'")
    
    # Remove cases with delay in reporting > 30 days and non-comprehensive claims
    raw_df = raw_df.filter((col('delay_in_reporting') < 30) & (col('policy_cover_type') != 'TPFT'))
    
    # Fix issue with type of some boolean columns
    raw_df = raw_df.withColumn('police_considering_actions', col('police_considering_actions').cast('boolean'))
    raw_df = raw_df.withColumn('is_crime_reference_provided', col('is_crime_reference_provided').cast('boolean'))
    raw_df = raw_df.withColumn('multiple_parties_involved', col('multiple_parties_involved').cast('boolean'))
    raw_df = raw_df.withColumn('total_loss_flag', col('total_loss_flag').cast('boolean'))
    
    # Create checks_list and num_failed_checks
    checks_columns = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday",
                      "C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference",
                      "C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
                      "C12_expensive_for_driver_age", "C14_contains_watchwords"]
    
    raw_df = raw_df.withColumn(
        "checks_list",
        array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in checks_columns])
    )
    
    raw_df = raw_df.withColumn(
        "checks_list",
        expr("filter(checks_list, x -> x is not null)")
    ).withColumn("num_failed_checks", size(col("checks_list")))
    
    # Convert to pandas
    raw_df = raw_df.select(numeric_features + categorical_features + other_cols).toPandas()
    
    raw_df['reported_month'] = pd.to_datetime(raw_df['reported_date']).dt.month
    raw_df['reported_year'] = pd.to_datetime(raw_df['reported_date']).dt.year
    
    raw_df['num_failed_checks'] = raw_df['num_failed_checks'].astype('float64')
    raw_df['score_card'] = ((raw_df['delay_in_reporting'] > 3) | (raw_df['policyholder_ncd_years'] < 2)).astype(int)
    
    return raw_df

# Load claims data
raw_df = load_claims_data(this_day)

print(f"Loaded {len(raw_df)} claims for {this_day}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Imputation and Score Models

# COMMAND ----------

# Use pre-calculated mean dictionary (or calculate from training data)
mean_dict = {'policyholder_ncd_years': 6.7899, 'inception_to_claim': 141.2893, 'min_claim_driver_age': 37.5581, 
             'veh_age': 11.3038, 'business_mileage': 306.2093, 'annual_mileage': 7372.2649, 'incidentHourC': 12.8702, 
             'additional_vehicles_owned_1': 0.0022, 'age_at_policy_start_date_1': 39.4507, 'cars_in_household_1': 1.8289, 
             'licence_length_years_1': 15.3764, 'years_resident_in_uk_1': 34.6192, 'max_additional_vehicles_owned': 0.003, 
             'min_additional_vehicles_owned': 0.0013, 'max_age_at_policy_start_date': 43.1786, 
             'min_age_at_policy_start_date': 35.4692, 'max_cars_in_household': 1.8861, 'min_cars_in_household': 1.7626, 
             'max_licence_length_years': 18.3106, 'min_licence_length_years': 12.2208, 'max_years_resident_in_uk': 38.5058, 
             'min_years_resident_in_uk': 30.5888, 'impact_speed': 27.1128, 'voluntary_amount': 241.3595, 
             'vehicle_value': 7861.6867, 'manufacture_yr_claim': 2011.9375, 'outstanding_finance_amount': 0.0, 
             'claim_to_policy_end': 83.4337, 'incidentDayOfWeekC': 4.0115, 'num_failed_checks': 0}

# Apply fills
raw_df = do_fills_pd(raw_df, mean_dict)

# Score with FA (Desk Check) model
raw_df['fa_pred'] = fa_pipeline.predict_proba(raw_df[numeric_features + categorical_features])[:,1].round(4)

# Score with Interview model
raw_df['y_prob2'] = interview_pipeline.predict_proba(raw_df[num_interview + cat_interview])[:,1].round(4)

print(f"Scored {len(raw_df)} claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Business Rules and Thresholds

# COMMAND ----------

def apply_business_rules(raw_df):
    """Apply business rules and overrides to model predictions"""
    
    fa_threshold = 0.5
    interview_threshold = 0.5
    
    raw_df['y_pred'] = ((raw_df['fa_pred'] >= fa_threshold)).astype(int)
    raw_df['y_pred2'] = ((raw_df['y_prob2'] >= interview_threshold)).astype(int)
    
    # Combined prediction: both models must flag and at least one check must fail
    raw_df['y_cmb'] = ((raw_df['y_pred'] == 1) & (raw_df['y_pred2'] == 1) & (raw_df['num_failed_checks'] >= 1)).astype(int)
    raw_df['flagged_by_model'] = np.where(raw_df['y_cmb'] == 1, 1, 0)
    
    # Override model predictions where desired
    
    # No commuting on policy and customer travelling between the hours of 12am and 4am
    default_vehicle_use = 1  # fill value
    late_night_no_commuting_condition = (raw_df['vehicle_use_quote'].fillna(default_vehicle_use)\
                                            .astype('int') == 1) & (raw_df['incidentHourC'].between(1, 4))
    raw_df['late_night_no_commuting'] = np.where(late_night_no_commuting_condition, 1, 0)
    raw_df['y_cmb'] = np.where(raw_df['late_night_no_commuting'] == 1, 1, raw_df['y_cmb'])
    
    # Add a column to check if Circumstances contains words to indicate unconsciousness
    watch_words = "|".join(["pass out", "passed out", 'passing out', 'blackout', 'black out',
                            'blacked out', 'blacking out', "unconscious", 'unconsciousness',
                            'sleep', 'asleep', 'sleeping', 'slept', 'dozed', 'doze', 'dozing'])
    raw_df['unconscious_flag'] = np.where(raw_df['Circumstances'].str.lower().str.contains(watch_words, na=False), 1, 0)
    raw_df['y_cmb'] = np.where(raw_df['unconscious_flag'] == 1, 1, raw_df['y_cmb'])
    
    raw_df['y_cmb_label'] = np.where(raw_df['y_cmb'] == 0, 'Low', 'High')
    raw_df['y_rank_prob'] = np.sqrt(raw_df['fa_pred'].fillna(100) * raw_df['y_prob2'].fillna(100)).round(3)
    raw_df['model_version'] = MODEL_VERSION
    
    return raw_df

# Apply business rules
raw_df = apply_business_rules(raw_df)

print(f"High risk claims: {raw_df['y_cmb'].sum()} ({raw_df['y_cmb'].sum()/len(raw_df)*100:.2f}%)")
print(f"Flagged by model: {raw_df['flagged_by_model'].sum()}")
print(f"Late night no commuting override: {raw_df['late_night_no_commuting'].sum()}")
print(f"Unconscious flag override: {raw_df['unconscious_flag'].sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Results to Table

# COMMAND ----------

# Convert back to Spark DataFrame
raw_df_spark = spark.createDataFrame(raw_df)

# Add risk reason column
risk_cols = ['flagged_by_model', 'unconscious_flag', 'late_night_no_commuting']
raw_df_spark = raw_df_spark.withColumn(
    "risk_reason",
    array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in risk_cols])
)
raw_df_spark = raw_df_spark.withColumn(
    "risk_reason",
    expr("filter(risk_reason, x -> x is not null)")
).withColumn("vehicle_use_quote", col("vehicle_use_quote").cast("double"))

# Write to table
raw_df_spark.write \
    .format("delta").option("mergeSchema", "true")\
    .mode("append") \
    .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions")

print(f"Successfully saved {raw_df_spark.count()} predictions to daily_svi_predictions table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Summary Statistics

# COMMAND ----------

# Summary statistics
print("="*60)
print("SCORING SUMMARY")
print("="*60)
print(f"Date: {this_day}")
print(f"Total claims scored: {len(raw_df)}")
print(f"High risk claims: {raw_df['y_cmb'].sum()} ({raw_df['y_cmb'].sum()/len(raw_df)*100:.2f}%)")
print(f"\nRisk breakdown:")
print(f"  - Flagged by model: {raw_df['flagged_by_model'].sum()}")
print(f"  - Late night no commuting: {raw_df['late_night_no_commuting'].sum()}")
print(f"  - Unconscious flag: {raw_df['unconscious_flag'].sum()}")
print(f"\nAverage scores:")
print(f"  - FA model: {raw_df['fa_pred'].mean():.4f}")
print(f"  - Interview model: {raw_df['y_prob2'].mean():.4f}")
print(f"  - Combined rank: {raw_df['y_rank_prob'].mean():.4f}")
print("="*60)