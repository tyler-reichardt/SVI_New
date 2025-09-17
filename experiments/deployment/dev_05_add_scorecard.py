# Databricks notebook source
# MAGIC %md
# MAGIC # Model scoring

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook template is used for scoring new claims
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Import libraries
# MAGIC

# COMMAND ----------


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql.functions import *

from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.inspection import partial_dependence
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

# COMMAND ----------

# MAGIC %md
# MAGIC #Get predictions

# COMMAND ----------

import mlflow
import pandas as pd
import datetime

#this_day = '2025-03-01'
#this_day = dbutils.widgets.get("date_range")

numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 'incidentDayOfWeekC', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal']

categorical_features = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 'reported_day_of_week', 'checks_max']

check_cols = [
    'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
    'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
    'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
    'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
]


# columns to fill using mean
mean_fills = [ "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", "incidentDayOfWeekC", "num_failed_checks"]

#boolean or damage columns with neg fills
damage_cols = ["damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal",]
bool_cols = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance","is_police_attendance","veh_age_more_than_10","police_considering_actions","is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved",  "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low","licence_low_1", "total_loss_flag"]
#neg_fills = bool_cols + damage_cols

# fills with ones (rules variables, to trigger manual check)
one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords",]

#fill with word 'missing' (categoricals) 
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
]) - set( ['car_group', 'employment_type_abi_code_5', 'employment_type_abi_code_4', 'employment_type_abi_code_3', 'employment_type_abi_code_2', 'postcode', 'employment_type_abi_code_1']))


# COMMAND ----------


def set_types(raw_df):
    # Recast data types & check schema
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

def do_fills_pd(raw_df):
    
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

#input features for interview model
num_interview = ['voluntary_amount', 'policyholder_ncd_years',
       'max_years_resident_in_uk', 'annual_mileage', 'min_claim_driver_age',
       'incidentHourC', 'areasDamagedHeavy', 'impact_speed',
       'years_resident_in_uk_1', 'vehicle_value', 'areasDamagedTotal',
       'manufacture_yr_claim', 'claim_to_policy_end', 'veh_age',
       'licence_length_years_1', 'num_failed_checks', 'areasDamagedMedium',
       'min_years_resident_in_uk', 'incidentDayOfWeekC',
       'age_at_policy_start_date_1', 'max_age_at_policy_start_date'
]

cat_interview = ['assessment_category', 'left_severity', 'C9_policy_within_30_days',
       'incident_sub_cause', 'rear_right_severity', 'front_left_severity',
       'rear_window_damage_severity', 'incident_cause', 'checks_max',
       'total_loss_flag']


# COMMAND ----------

# Load model
fa_model_run_id = "56ced60a6d464b07835f5a237425b33b"
interview_run_id = 'b4b7c18076b84880aeea3ff5389c3999'
fa_pipeline = mlflow.sklearn.load_model(f'runs:/{fa_model_run_id}/model')
interview_pipeline = mlflow.sklearn.load_model(f'runs:/{interview_run_id}/model')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Data read and Scoring

# COMMAND ----------


#load claims data
table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_claims_svi"

# Read in dataset & display

check_cols = [
    'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
    'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
    'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
    'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
]

other_cols = ['claim_number', 'reported_date', 'start_date',  'num_failed_checks', 'total_loss_flag', 'checks_list', 
              'delay_in_reporting'] \
                + [x for x in check_cols if x not in categorical_features] 
raw_df = spark.table(table_path).withColumn('checks_max', greatest(*[col(c) for c in check_cols]).cast('string'))\
                .withColumn("underbody_damage_severity", lit(None))#.filter("dataset='test'")
print(raw_df.count())
#fix issue with type of some boolean columns
raw_df = raw_df.withColumn('police_considering_actions', col('police_considering_actions').cast('boolean'))
raw_df = raw_df.withColumn('is_crime_reference_provided', col('is_crime_reference_provided').cast('boolean'))
raw_df = raw_df.withColumn('multiple_parties_involved', col('multiple_parties_involved').cast('boolean'))
raw_df = raw_df.withColumn('total_loss_flag', col('total_loss_flag').cast('boolean'))

checks_columns = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords",]

raw_df = raw_df.withColumn(
    "checks_list",
    array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in checks_columns])
)

raw_df = raw_df.withColumn(
    "checks_list",
    expr("filter(checks_list, x -> x is not null)")
).withColumn("num_failed_checks", size(col("checks_list")))
#\.filter(f"DATE(reported_date)='{this_day}'")

from pyspark.sql.functions import col, when
raw_df = raw_df.select(numeric_features+categorical_features+other_cols).toPandas()
raw_df['reported_month'] = pd.to_datetime(raw_df['reported_date']).dt.month
raw_df['reported_year'] = pd.to_datetime(raw_df['reported_date']).dt.year

raw_df['num_failed_checks'] = raw_df['num_failed_checks'].astype('float64')

#TODO: use constants as mean_dict
#mean_dict = raw_df[mean_fills].astype('float').mean().round(4).to_dict()
mean_dict = {'policyholder_ncd_years': 6.7899, 'inception_to_claim': 141.2893, 'min_claim_driver_age': 37.5581, 'veh_age': 11.3038, 'business_mileage': 306.2093, 'annual_mileage': 7372.2649, 'incidentHourC': 12.8702, 'additional_vehicles_owned_1': 0.0022, 'age_at_policy_start_date_1': 39.4507, 'cars_in_household_1': 1.8289, 'licence_length_years_1': 15.3764, 'years_resident_in_uk_1': 34.6192, 'max_additional_vehicles_owned': 0.003, 'min_additional_vehicles_owned': 0.0013, 'max_age_at_policy_start_date': 43.1786, 'min_age_at_policy_start_date': 35.4692, 'max_cars_in_household': 1.8861, 'min_cars_in_household': 1.7626, 'max_licence_length_years': 18.3106, 'min_licence_length_years': 12.2208, 'max_years_resident_in_uk': 38.5058, 'min_years_resident_in_uk': 30.5888, 'impact_speed': 27.1128, 'voluntary_amount': 241.3595, 'vehicle_value': 7861.6867, 'manufacture_yr_claim': 2011.9375, 'outstanding_finance_amount': 0.0, 'claim_to_policy_end': 83.4337, 'incidentDayOfWeekC': 4.0115}

raw_df =  do_fills_pd(raw_df)

raw_df['fa_pred'] = fa_pipeline.predict_proba(raw_df[numeric_features + categorical_features])[:,1].round(4)

#raw_df['y_prob2'] = interview_pipeline.predict_proba(raw_df[num_interview + cat_interview + ['fa_pred']] )[:,1].round(4)
raw_df['y_prob2'] = interview_pipeline.predict_proba(raw_df[num_interview + cat_interview ] )[:,1].round(4)

def evaluate_thresholds(fa_threshold, interview_threshold):
    raw_df['y_pred'] =  ((raw_df['fa_pred'] >= fa_threshold)).astype(int)
    raw_df['y_pred2'] =  ((raw_df['y_prob2'] >= interview_threshold)).astype(int)

    #raw_df['y_2step'] = ((raw_df['y_pred'] == 1) & (raw_df['y_pred2'] == 1)).astype(int) 
    raw_df['y_cmb'] = ((raw_df['y_pred'] == 1) &  (raw_df['y_pred2'] == 1) &  (raw_df['num_failed_checks']>=1)).astype(int)

    cm = confusion_matrix(raw_df['svi_risk'], raw_df['y_cmb'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=['Low Risk', 'High Risk'])
    disp.plot()

    print(classification_report(raw_df['svi_risk'].replace({-1: 0}), raw_df['y_cmb']))

    raw_df['referred_to_tbg'] = raw_df['tbg_risk'].apply(lambda x: 1 if x in [0, 1] else 0)
    raw_df['svi_referred'] = raw_df.apply(lambda row: 1 if row['referred_to_tbg'] == 1 and row['svi_risk'] == 1 else 0, axis=1)
    y_cmb_per_month = raw_df[raw_df['y_cmb'] == 1].copy().groupby(['reported_year', 'reported_month']).agg(
        DS_referred=('y_cmb', 'sum'),
        captured_high_risk=('svi_referred', 'sum'),
    ).reset_index()

    svi_risk_all = raw_df.groupby(['reported_year', 'reported_month']).agg(
        all_high_risk=('svi_referred', 'sum'),
        actual_referral=('referred_to_tbg', 'sum'),
    ).reset_index()

    combined_df = pd.merge(y_cmb_per_month, svi_risk_all, on=['reported_year', 'reported_month'])
    combined_df = combined_df[combined_df['reported_year'] == 2024]
    display_order =  ['reported_year', 'reported_month', 'actual_referral', 'DS_referred',
                    'all_high_risk', 'captured_high_risk',]
    combined_df = combined_df[display_order][(combined_df['reported_year'] == 2024)]

    mean_row = (3*combined_df.mean(numeric_only=True).to_frame().T).round()
    mean_row.iloc[0, 0] = fa_threshold
    mean_row.iloc[0, 1] = interview_threshold
    mean_row.columns = ['fa_threshold', 'interview_threshold'] + mean_row.columns[2:].tolist()
    return mean_row



# COMMAND ----------

fa_threshold = 0.5
interview_threshold = 0.5

raw_df['y_pred'] =  ((raw_df['fa_pred'] >= fa_threshold)).astype(int)
raw_df['y_pred2'] =  ((raw_df['y_prob2'] >= interview_threshold)).astype(int)

#raw_df['y_2step'] = ((raw_df['y_pred'] == 1) & (raw_df['y_pred2'] == 1)).astype(int)

raw_df['y_cmb'] = ((raw_df['y_pred'] == 1) &  (raw_df['y_pred2'] == 1) &  (raw_df['num_failed_checks']>=1)).astype(int)
raw_df['y_cmb'] = np.where(raw_df['policy_cover_type'] == 'Comprehensive', raw_df['y_cmb'], 1)
raw_df['y_cmb_label'] = np.where(raw_df['y_cmb']==0, 'Low', 'High')

raw_df['y_rank_prob'] = np.sqrt(raw_df['fa_pred'].fillna(100) * raw_df['y_prob2'].fillna(100)).round(3)


display(raw_df)


# COMMAND ----------



# COMMAND ----------

#percentage 
percentage = score_df[(score_df['y_cmb'] == 1) & (score_df['score_card'] == 1)].shape[0] / score_df[score_df['y_cmb'] == 1].shape[0] * 100
print(percentage)

percent_card = score_df[(score_df['score_card'] == 1) & (score_df['y_cmb'] == 1)].shape[0] / score_df[score_df['score_card'] == 1].shape[0] * 100
print(percent_card)

# COMMAND ----------

1/0

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Write table

# COMMAND ----------

#uncomment to write to ADP aux catalog
'''
raw_df_spark = spark.createDataFrame(raw_df)
raw_df_spark.write \
    .mode("append") \
    .format("delta").option("mergeSchema", "true")\
    .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions")
'''

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import col, hour, dayofweek, month
from pyspark.sql.functions import *

spark.sql('USE CATALOG prod_adp_certified')

#policy_svi = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_policy_svi")\
#                    .filter(col("reported_date") >= '2025-03-01')

latest_claim_version = spark.table("prod_adp_certified.claim.claim_version")\
    .groupBy("claim_id")\
    .agg(max("claim_version_id").alias("claim_version_id"))

claim_version = spark.table("claim.claim_version")
claim_version_item = spark.table("claim.claim_version_item")
claim = spark.table("claim.claim")
incident = spark.table("claim.incident")
claimant = spark.table("claim.claimant")

check_df = latest_claim_version.alias("lcv")\
    .join(claim_version.alias("cv"), (col("lcv.claim_id") == col("cv.claim_id")), "inner")\
    .join(claim_version_item.alias("cvi"), 
          (col("cv.claim_version_id") == col("cvi.claim_version_id")) & 
          (col("cv.claim_id") == col("cvi.claim_id")), "inner")\
    .join(claim.alias("c"), (col("c.claim_id") == col("cv.claim_id")), "inner")\
    .join(incident.alias("i"), (col("cv.event_identity") == col("i.event_identity")), "left")\
    .join(claimant.alias("cl"), 
          (col("cl.claim_version_id") == col("cvi.claim_version_id")) & 
          (col("cl.claim_version_item_index") == col("cvi.claim_version_item_index")) & 
          (col("cl.event_identity") == col("cvi.event_identity")), "left")\
    .filter((col("cv.claim_number").isNotNull()) & (col("cl.is_first_party") == True) & (col("cvi.claim_version_item_index") == 0))\
    .select(
            col("cv.claim_number"),
            col("cv.policy_number"),
            col("cv.claim_version_id"),
            col("cv.policy_cover_type"),
            col("i.start_date"),
            col("i.reported_date"),
            col("i.multiple_parties_involved"),
            col("i.notification_method"),
            hour(col("i.start_date")).alias("incidentHourC"),
            dayofweek(col("i.start_date")).alias("incidentDayOfWeekC"),
            month(col("i.start_date")).alias("incidentMonthC")
        )\
    .distinct()


# COMMAND ----------

#count_ncd_years = result_df.filter(col("policyholder_ncd_years") == 6.7899).count()
#count_ncd_years

# COMMAND ----------

check_df = check_df.withColumn("claim_number", lower(col("claim_number")))\
            .withColumn("delay_in_reporting", datediff(col("reported_date"), col("start_date")))

score_df = score_df.withColumn("policyholder_ncd_years", when(col("policyholder_ncd_years") == 6.7899, -1).otherwise(col("policyholder_ncd_years")))

result_df = score_df.withColumn("claim_number", lower(col("claim_number"))).alias("sdf")\
    .join(check_df.alias("cdf"), col("sdf.claim_number") == col("cdf.claim_number"), "left")\
    .select("sdf.*", "cdf.*").distinct()

display(result_df)


# COMMAND ----------

#pivot_df = pivot_df.withColumn("high_risk_percentage", (col("1") / (col("0") + col("1"))) * 100)
#pivot_df = pivot_df.withColumn("score_card_percentage", (col("1") / (col("1") + col("0"))) * 100)
#display(pivot_df)

# COMMAND ----------

result_df = result_df.withColumn("score_card", ((col("delay_in_reporting") > 3) & (col("policyholder_ncd_years") < 2)).cast("int"))
#what percentage of high risk claims are flagged by the score card
high_risk_percentage = (result_df.filter((col('score_card') == 1) & (col('y_cmb') == 1)).count() / result_df.filter(col('y_cmb') == 1).count()) * 100
print('high_risk_percentage: ', high_risk_percentage)

#what percentage of score card claims are flagged by y_cmb
score_card_percentage = (result_df.filter((col('score_card') == 1) & (col('y_cmb') == 1)).count() / result_df.filter(col('score_card') == 1).count()) * 100
print('score_card_percentage: ', score_card_percentage)
1/0

# COMMAND ----------

print(score_df.filter(col('score_card') == 1).count())

# COMMAND ----------


score_df = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions")
score_df = score_df.withColumn("policyholder_ncd_years", when(col("policyholder_ncd_years") == 6.7899, -1).otherwise(col("policyholder_ncd_years")))


# ncd<2years
#notify > 3 days

max_reported_date = score_df.agg({"reported_date": "max"}).collect()[0][0]
min_reported_date = score_df.agg({"reported_date": "min"}).collect()[0][0]
display(max_reported_date)
display(min_reported_date)

#2025, 5, 7 to 2025, 3, 1,

# COMMAND ----------

print(sorted(score_df.columns))

# COMMAND ----------



# COMMAND ----------

df2 = spark.read.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions")
earliest_reported_date = df2.agg({"reported_date": "min"}).collect()[0][0]
latest_reported_date = df2.agg({"reported_date": "max"}).collect()[0][0]
display(earliest_reported_date)
display(latest_reported_date)

# COMMAND ----------

dbutils.notebook.exit("Notebook stopped"); 1/0
#spark.sql("TRUNCATE TABLE prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions")
display(spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions")
)
#spark.sql("TRUNCATE TABLE prod_dsexp_auxiliarydata.single_vehicle_incident_checks.#daily_claims_svi")
