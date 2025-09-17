# Databricks notebook source
# MAGIC %md
# MAGIC # PBC - Performance Metrics Template

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook template is used for evaluating model performance. 
# MAGIC * Performance metrics
# MAGIC * Confusion matrix 
# MAGIC * Precision / Recall Curve 
# MAGIC * Lift charts / double lift charts 
# MAGIC * AvsE charts 
# MAGIC *Histogram of errors 
# MAGIC * Hosmer plots
# MAGIC * Feature importance / SHAP beeswarm plots
# MAGIC * PDPs
# MAGIC * Feature interaction strength 
# MAGIC * Translate model performance into expected benefit to business
# MAGIC * Playback to stakeholders
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

numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 'incidentDayOfWeekC', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal', 'areasDamagedRelative']

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
damage_cols = ["damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal", "areasDamagedRelative"]
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
       'incidentHourC', 'areasDamagedHeavy', 'impact_speed', 'vehicle_value',
       'areasDamagedTotal', 'manufacture_yr_claim', 'claim_to_policy_end',
       'veh_age', 'licence_length_years_1', 'num_failed_checks',
       'areasDamagedMedium', 'incidentDayOfWeekC',
       'age_at_policy_start_date_1']

cat_interview = ['assessment_category', 'left_severity', 'C9_policy_within_30_days',
       'incident_sub_cause', 'rear_right_severity', 'front_left_severity',
       'rear_window_damage_severity', 'incident_cause', 'checks_max',
       'total_loss_flag']


# COMMAND ----------

# Load model
fa_model_run_id = "56ced60a6d464b07835f5a237425b33b"
#interview_run_id = '9ce19e7acc2d44acbd31b56d790f6913'
interview_run_id = 'b4b7c18076b84880aeea3ff5389c3999'

fa_pipeline = mlflow.sklearn.load_model(f'runs:/{fa_model_run_id}/model')
interview_pipeline = mlflow.sklearn.load_model(f'runs:/{interview_run_id}/model')
print(interview_pipeline.named_steps['classifier'].get_params())


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Data read and Scoring

# COMMAND ----------


#load claims data
table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"

# Read in dataset & display

check_cols = [
    'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
    'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
    'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
    'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
]

other_cols = ['svi_risk', 'fa_risk', 'claim_number', 'reported_date', 'start_date', 'tbg_risk', 'num_failed_checks', 'total_loss_flag'] + [x for x in check_cols if x not in categorical_features] 
raw_df = spark.table(table_path).withColumn('checks_max', greatest(*[col(c) for c in check_cols]).cast('string'))\
                .withColumn("underbody_damage_severity", lit(None)).filter("dataset='test'")

############## fix bug in data build 
##C1: was the incident on a Friday/Saturday *NIGHT*?
raw_df = raw_df.withColumn("areasDamagedRelative", col("areasDamagedMinimal") + 2*col("areasDamagedMedium") + 3*col("areasDamagedSevere") + 4*col("areasDamagedHeavy"))\
    .withColumn("incident_day_of_week", date_format(col("reported_date"), "E"))

fri_sat_night = ((col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4))))
                                                                                                                    
raw_df = raw_df.withColumn(
    "C1_fri_sat_night",
    when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0))
#####################

#TODO: rename to tbg_risk
raw_df = raw_df.fillna({'fa_risk':0})

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

############## fix bug in data build 
##C1: was the incident on a Friday/Saturday *NIGHT*?
raw_df = raw_df.withColumn("areasDamagedRelative", col("areasDamagedMinimal") + 2*col("areasDamagedMedium") + 3*col("areasDamagedSevere") + 4*col("areasDamagedHeavy"))\
    .withColumn("incident_day_of_week", date_format(col("reported_date"), "E"))

fri_sat_night = ((col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4))))
                                                                                                                    
raw_df = raw_df.withColumn(
    "C1_fri_sat_night",
    when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0))
#####################

#add max outsourcing fees
svi_perf = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_performance")\
                .select("Outsourcing Fee", "Claim Number").coalesce(1).orderBy("Claim Number", "Outsourcing Fee", ascending=False)\
                .drop_duplicates()

# clip values
from pyspark.sql.functions import col, when
'''
raw_df = raw_df.withColumn('damageScore', when(col('damageScore') > 100000.0, 100000.0).otherwise(col('annual_mileage')))
raw_df = raw_df.withColumn('annual_mileage', when(col('annual_mileage') > 50000.0, 50000.0).otherwise(col('annual_mileage')))
raw_df = raw_df.withColumn('vehicle_value', when(col('vehicle_value') > 50000.0, 50000.0).otherwise(col('vehicle_value')))
raw_df = raw_df.withColumn('business_mileage', when(col('business_mileage') > 40000.0, 40000.0).otherwise(col('business_mileage')))
'''

raw_df = raw_df.fillna({'fa_risk': 0})

raw_df = raw_df.select(numeric_features+categorical_features+other_cols).toPandas()
raw_df = raw_df[(raw_df.reported_date<='2024-07-31')]
raw_df['reported_month'] = pd.to_datetime(raw_df['reported_date']).dt.month
raw_df['reported_year'] = pd.to_datetime(raw_df['reported_date']).dt.year

raw_df['num_failed_checks'] = raw_df['num_failed_checks'].astype('float64')

raw_df['svi_risk'] = raw_df['svi_risk'].replace({-1: 0})

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
    combined_df = combined_df[(combined_df['reported_year'] == 2024)]
    display_order =  ['reported_year', 'reported_month', 'actual_referral', 'DS_referred',
                    'all_high_risk', 'captured_high_risk',]
    combined_df = combined_df[display_order][(combined_df['reported_year'] == 2024)]

    mean_row = (3*combined_df.mean(numeric_only=True).to_frame().T).round()
    mean_row.iloc[0, 0] = fa_threshold
    mean_row.iloc[0, 1] = interview_threshold
    mean_row.columns = ['fa_threshold', 'interview_threshold'] + mean_row.columns[2:].tolist()
    return mean_row


# COMMAND ----------

# MAGIC %md
# MAGIC ## Recall/Precision at Various Thresholds

# COMMAND ----------

from sklearn.metrics import precision_score, recall_score
import pandas as pd
import numpy as np

thresholds = np.arange(0.0, 1.05, 0.05)
results = []

for fa_pred_threshold in thresholds:
    for y_prob2_threshold in thresholds:
        y_pred_fa = (raw_df['fa_pred'] >= fa_pred_threshold).astype(int)
        y_pred_y_prob2 = (raw_df['y_prob2'] >= y_prob2_threshold).astype(int)
        y_combined = ((y_pred_fa == 1) & (y_pred_y_prob2 == 1)).astype(int)
        
        precision = precision_score(raw_df['svi_risk'].replace({-1: 0}), y_combined)
        recall = recall_score(raw_df['svi_risk'].replace({-1: 0}), y_combined)
        f1 = f1_score(raw_df['svi_risk'].replace({-1: 0}), y_combined)
           
        results.append({
            'fa_pred_threshold': fa_pred_threshold,
            'y_prob2_threshold': y_prob2_threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

results_df = pd.DataFrame(results)
display(results_df.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration 1

# COMMAND ----------

monthly_cases = pd.DataFrame()

fa_threshold = 0.5
interview_threshold = 0.5
#run to get binary predictions
mean_row = evaluate_thresholds(fa_threshold, interview_threshold)
monthly_cases = pd.concat([monthly_cases, mean_row], ignore_index=True)
display(monthly_cases)
#from figure below, we'd capture 55% of current savings


# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration 2

# COMMAND ----------

fa_threshold = 0.25
interview_threshold = 0.25
#run to get binary predictions
mean_row = evaluate_thresholds(fa_threshold, interview_threshold)
monthly_cases = pd.concat([monthly_cases, mean_row], ignore_index=True)
display(monthly_cases)
#from figure below, we'd capture 86% of current savings

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration 3

# COMMAND ----------

fa_threshold = 0.45
interview_threshold = 0.3
#run to get binary predictions
mean_row = evaluate_thresholds(fa_threshold, interview_threshold)
monthly_cases = pd.concat([monthly_cases, mean_row], ignore_index=True)
display(monthly_cases)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select final thresholds

# COMMAND ----------

#reset dataframe
fa_threshold = 0.35
interview_threshold = 0.55
mean_row = evaluate_thresholds(fa_threshold, interview_threshold)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Write table

# COMMAND ----------

#uncomment to write to ADP aux catalog
'''
raw_df_spark.write \
    .mode("overwrite") \
    .format("delta")#.option("mergeSchema", "true")\
    .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_predictions")
'''

# COMMAND ----------



# COMMAND ----------

display(raw_df[num_interview + cat_interview ].dtypes)


# COMMAND ----------

#get numbers of cases referred per month
print(classification_report(raw_df['svi_risk'].replace({-1: 0}), raw_df['y_cmb']))
raw_df['reported_month'] = pd.to_datetime(raw_df['reported_date']).dt.month
raw_df['reported_year'] = pd.to_datetime(raw_df['reported_date']).dt.year
raw_df['referred_to_tbg'] = raw_df['tbg_risk'].apply(lambda x: 1 if x in [0, 1] else 0)
raw_df['svi_referred'] = raw_df.apply(lambda row: 1 if row['referred_to_tbg'] == 1 and row['svi_risk'] == 1 else 0, axis=1)
y_cmb_per_month = raw_df[raw_df['y_cmb'] == 1].copy().groupby(['reported_year', 'reported_month']).agg(
    predicted_high_risk=('y_cmb', 'sum'),
    actual_high_risk=('svi_referred', 'sum'),
).reset_index()

svi_risk_all = raw_df.groupby(['reported_year', 'reported_month']).agg(
    predicted_referral=('svi_referred', 'sum'),
    actual_referral=('referred_to_tbg', 'sum'),
).reset_index()

combined_df = pd.merge(y_cmb_per_month, svi_risk_all, on=['reported_year', 'reported_month'])
display(combined_df[combined_df['reported_year'] == 2024])
#display(combined_df[['reported_year', 'reported_month', 'predicted_referral', 'referred_to_tbg', 'actual_high_risk', 'all_current_high_risk']][(combined_df['reported_year'] == 2024)])

#ds referral, current referrals, no of repud_DS, no of repud_current
print(raw_df.columns)

# COMMAND ----------

raw_df['incident_month'] = pd.to_datetime(raw_df['reported_date']).dt.month
raw_df['incident_year'] = pd.to_datetime(raw_df['reported_date']).dt.year

y_cmb_per_month = raw_df.groupby(['incident_year', 'incident_month']).agg(
    count=('y_cmb', lambda x: (raw_df.loc[x.index, 'y_cmb'] == 1).sum()),
    pred_sum=('y_cmb', lambda x: (raw_df.loc[x.index, 'y_cmb'] == 1).sum()),
    svi_risk_count=('svi_risk', 'sum')
).reset_index()
display(y_cmb_per_month)

# COMMAND ----------

print(raw_df['svi_risk'].value_counts())

# COMMAND ----------



# COMMAND ----------


percentage = raw_df[(raw_df['y_pred'] == 1) & (raw_df['svi_risk'] == 1)].shape[0] / raw_df[raw_df['y_pred'] == 1].shape[0] * 100
fa_percentage = raw_df[(raw_df['fa_risk'] == 1) & (raw_df['svi_risk'] == 1)].shape[0] / raw_df[raw_df['fa_risk'] == 1].shape[0] * 100
step2_percentage = raw_df[(raw_df['y_pred2'] == 1) & (raw_df['svi_risk'] == 1)].shape[0] / raw_df[raw_df['y_pred2'] == 1].shape[0] * 100
print("Percentage eventually high risk: ")
print('2_step model: ', step2_percentage)
print('FA_Model: ', percentage)
print('Desk check: ', fa_percentage)




# COMMAND ----------



# COMMAND ----------

'''
import matplotlib.pyplot as plt
import pandas as pd

def count_plot(raw_df, predicted, actual, title, ax): 
    # Count the occurrences of 0's and 1's for y_pred and svi_risk
    y_pred_counts = raw_df[predicted].value_counts()
    svi_risk_counts = raw_df[actual].value_counts()
    print(y_pred_counts.values)
    print(svi_risk_counts.values)

    # Create a DataFrame for plotting
    counts_df = pd.DataFrame({
        'predicted': y_pred_counts.values,
        'actuals': svi_risk_counts.values
    }).fillna(0)

    # Plotting
    counts_df.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticklabels(counts_df.index, rotation=0)
    ax.legend(title='Legend')

plt.subplot()
count_plot(raw_df, 'y_pred2' , 'svi_risk', f'interview model', ax=plt.gca())

plt.subplot()
count_plot(raw_df, 'y_pred' , 'fa_risk', f'deskcheck(FA) model', ax=plt.gca())
'''

# COMMAND ----------

'''
# Set up the figure for subplots
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
axes = axes.flatten()

# Loop through each column in check_cols and create a count plot
for i, col in enumerate(check_cols):
    count_plot(raw_df, col, 'svi_risk', f'{col} model', ax=axes[i])

# Adjust layout
plt.tight_layout()
plt.show()
'''

# COMMAND ----------

'''
import pandas as pd

result_df = pd.DataFrame()

for i in range(len(check_cols)):
    col_name = check_cols[i] 

    value_col = 'svi_risk'
    pivot_df = raw_df[[col_name, value_col]].astype(int).pivot_table(
        values=value_col,
        index=col_name,
        aggfunc={value_col: ['sum']}
    ).reset_index()

    # Rename columns for clarity
    pivot_df.columns = [col_name, f'{value_col}_(%)']

    # Divide the count column by the total number of rows in raw_df
    pivot_df[f'{value_col}_(%)'] = (100 * pivot_df[f'{value_col}_(%)'] / raw_df.shape[0]).round(2)

    # Concatenate the results into a single DataFrame
    result_df = pd.concat([result_df, pivot_df.T.reset_index().T.astype(str)], axis=0)

display(result_df)
'''

# COMMAND ----------

'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))
axes = axes.flatten()

for i, col in enumerate(check_cols):
    avg_svi_risk = raw_df.groupby(col)['svi_risk'].mean().reset_index()
    avg_svi_risk.columns = [col, 'avg_svi_risk']
    
    counts = raw_df[col].value_counts().sort_index()
    
    axes[i].bar(avg_svi_risk[col], avg_svi_risk['avg_svi_risk'], color=['blue', 'orange'])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Average svi_risk')
    axes[i].set_title(f'Average svi_risk for {col}')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['0', '1'])
    
    for j, count in enumerate(counts):
        axes[i].text(j, avg_svi_risk['avg_svi_risk'][j], f'n={count}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
'''

# COMMAND ----------

# Group by 'y_pred' and 'svi_risk' and find the average for each group
print((raw_df[['fa_risk']].value_counts()/raw_df.shape[0]).round(2))
print((raw_df[['y_pred']].value_counts()/raw_df.shape[0]).round(2))

print((raw_df[['svi_risk']].value_counts()/raw_df.shape[0]).round(2))
print((raw_df[['y_cmb']].value_counts()/raw_df.shape[0]).round(2))
raw_df['svi_risk2'] = raw_df[['svi_risk']]
print(raw_df.groupby('y_pred').mean().unstack()['svi_risk2'])

# COMMAND ----------



# COMMAND ----------

'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axes = axes.flatten()

interview_cols = ['y_pred2']

for i, col in enumerate(interview_cols):
    avg_svi_risk = raw_df.groupby(col)['svi_risk'].mean().reset_index()
    avg_svi_risk.columns = [col, 'avg_svi_risk']
    
    counts = raw_df[col].value_counts().sort_index()
    
    axes[i].bar(avg_svi_risk[col], avg_svi_risk['avg_svi_risk'], color=['blue', 'orange'])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Average svi_risk')
    axes[i].set_title(f'Average svi_risk for low/high risk predicted by Interview model')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['0', '1'])
    
    for j, count in enumerate(counts):
        axes[i].text(j, avg_svi_risk['avg_svi_risk'][j], f'n={count}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
'''

# COMMAND ----------

'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
axes = axes.flatten()

interview_cols = ['y_pred']

for i, col in enumerate(interview_cols):
    avg_svi_risk = raw_df.groupby(col)['svi_risk'].mean().reset_index()
    avg_svi_risk.columns = [col, 'avg_svi_risk']
    
    counts = raw_df[col].value_counts().sort_index()
    
    axes[i].bar(avg_svi_risk[col], avg_svi_risk['avg_svi_risk'], color=['blue', 'orange'])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Average svi_risk')
    axes[i].set_title(f'Average svi_risk for low/high risk predicted by FA model')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['0', '1'])
    
    for j, count in enumerate(counts):
        axes[i].text(j, avg_svi_risk['avg_svi_risk'][j], f'n={count}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
'''

# COMMAND ----------

raw_df['y_prob2'].hist()
#overlap with threshold


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #Classification Report

# COMMAND ----------

print(classification_report(raw_df['svi_risk'], raw_df['y_cmb']))


# COMMAND ----------

# MAGIC %md
# MAGIC # Gini Score

# COMMAND ----------

def gini(y_true, y_pred):
  # Sort actual and predicted values by predicted values
  assert len(y_true) == len(y_pred)
  
  # Sort by predicted values, shows the ranking order your model has given
  all_data = np.array(list(zip(y_pred, y_true)))
  all_data = all_data[all_data[:, 0].argsort()]
  
  # Get the sorted predicted and actual values
  sorted_predicted = all_data[:, 0]
  sorted_actual = all_data[:, 1]
  
  # Cumulative sum of actual values
  cumulative_actual = np.cumsum(sorted_actual)
  cumulative_actual_normalized = cumulative_actual / cumulative_actual[-1]
  # this array shows the proportional representation of where the actual volumes sit on your predicted specturm
  
  # Gini formula
  gini_score = 1 - 2 * np.sum(cumulative_actual_normalized) / len(y_true) + 1 / len(y_true)
    
  return gini_score
  
def gini_normalized(y_true, y_pred):
  # we normalize the gini score to account for the fact that the actual distribution may not be uniform
  # so by dividing through by that, we see the proportional impact that our predictions have on the order
  return gini(y_true, y_pred) / gini(y_true, y_true)

# COMMAND ----------

print("Gini Score (Desk-check Model): " + str(gini_normalized(raw_df['fa_risk'], raw_df['fa_pred'])))

print("Gini Score (Interview Model): " + str(gini_normalized(raw_df['svi_risk'], raw_df['y_prob2'])))

# COMMAND ----------

# MAGIC %md
# MAGIC #Precision / Recall Curve

# COMMAND ----------

# desk check model
PrecisionRecallDisplay.from_predictions(raw_df['svi_risk'], raw_df['y_prob2'], plot_chance_level=True)

# COMMAND ----------

# interview model
PrecisionRecallDisplay.from_predictions(raw_df['fa_risk'], raw_df['fa_pred'], plot_chance_level=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #Lift charts / double lift charts

# COMMAND ----------

#add this script to the workspace and import the relevant function
#https://dev.azure.com/firstcentralsg/Data%20Science/_git/CodingToolkit?path=/model_evaluate/lift_chart_plots.py. Sample function call is below

#plot_lift_chart_regression(y_pred, y_test, n_bins=10, weights=None)

# COMMAND ----------

# MAGIC %md
# MAGIC #Histogram of errors

# COMMAND ----------

#combined model (deskcheck, interview and num_failed_checks)
y_errors_cmb = raw_df['svi_risk'] - raw_df['y_cmb']
plt.hist(y_errors_cmb)

#deskcheck model
plt.figure()
y_errors_desk = raw_df['fa_risk'] - raw_df['y_pred']
plt.hist(y_errors_desk)

#interview model
plt.figure()
y_errors_interview = raw_df['svi_risk'] - raw_df['y_pred2']
plt.hist(y_errors_interview)

# COMMAND ----------

# MAGIC %md
# MAGIC #In-built Feature importance

# COMMAND ----------

# function that calculates features importance for all features in a dataset
def get_feature_importance_table(model, df):
  """
  Parameters:
  ----------
  model : trained model
      Function will accept any model that works with the partial dependence function from sklearn.inspection
      This includes most tree based models

  df : pandas dataframe
      X_test holdout dataset

  Returns:
  -------
  pandas dataframe

  Dataframe will have a row per feature, with columns to represent the total importance, as well as the porportional importance

  """
  # create dataframe with row per feature
  df_imp = pd.DataFrame(df.columns, columns=['feature'])
  # add in the feature importances
  # if statements to support varying model formats
  if hasattr(model, 'feature_importances_'):
    df_imp['importance'] = model.feature_importances_
  elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_importance'):
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type=importance_type)
    df_imp['importance'] = [importance_dict.get(f, 0) for f in df.columns]
  elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_importance'):
    df_imp['importance'] = model.booster_.feature_importance(importance_type='gain')
  else:
    raise ValueError("Unsupported model type. Must be scikit-learn, XGBoost, or LightGBM model.")

  # calculate relative proportion of gain explained by each feature
  df_imp['importance_pc'] = 100*df_imp['importance']/df_imp['importance'].sum()
  return df_imp.head(20)

def plot_importance(df_imp, num_features=0):
  """
  Parameters:
  ----------
  df_imp : pandas dataframe
      Designed to take output of get_feature_importance_table function
      Require columns for feature and importance_pc

  num_features : int
      Positive non-zero integer to state the number of most important features to plot in chart

  Returns:
  -------
  null (Will plot feature importance chart)

  """
  # sort table by importance, ascending fixes order of chart
  df = df_imp.sort_values('importance_pc',ascending=True)
  # apply cap to number of features plotted
  if num_features > 0:  
    # use tail as values sorted in ascending order
    df.tail(num_features).plot(kind='barh', x='feature', y='importance_pc', color='skyblue', legend=False)
  else:
    df.plot(kind='barh', x='feature', y='importance_pc', color='skyblue', legend=False)

  plt.xlabel('Feature Importance % (Gain)')
  plt.ylabel('Feature')
  plt.title('Feature Importance (% Gain explained)')

  # Show the plot
  plt.show()

# COMMAND ----------

print("desk-check model")
preprocessor = fa_pipeline.named_steps['preprocessor']
fa_model = fa_pipeline.named_steps['classifier']
display((preprocessor))
df_prep = pd.DataFrame(preprocessor.transform(raw_df).toarray(), columns=preprocessor.get_feature_names_out())
plot_importance(get_feature_importance_table(fa_model, df_prep))

# COMMAND ----------

print("interview model")
preprocessor = interview_pipeline.named_steps['preprocessor']
interview_model = interview_pipeline.named_steps['classifier']
df_prep_interview = pd.DataFrame(preprocessor.transform(raw_df), columns=preprocessor.get_feature_names_out())
plot_importance(get_feature_importance_table(interview_model, df_prep_interview))

# COMMAND ----------

import seaborn as sns
print("FA model")
fa_preprocessor = fa_pipeline.named_steps['preprocessor']
fa_model = fa_pipeline.named_steps['classifier']
#plot_importance(get_feature_importance_table(fa_model, df_prep_fa))
fa_imp = pd.DataFrame({'features': fa_preprocessor.get_feature_names_out(), 'importance': fa_model.feature_importances_})\
    .sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(12,8))
sns.set_style('white')
sns.barplot(x='importance', y='features', data=fa_imp.head(20), ax=ax, palette='pastel')

display(fa_imp)


# COMMAND ----------

print("Interview model")
interview_preprocessor = interview_pipeline.named_steps['preprocessor']
interview_model = interview_pipeline.named_steps['classifier']
interview_imp = pd.DataFrame({'features': interview_preprocessor.get_feature_names_out(),
                              'importance': interview_model.feature_importances_})\
                        .sort_values('importance', ascending=False)

nz_interview_imp = interview_imp[interview_imp['importance'] != 0]

fig, ax = plt.subplots(figsize=(12,8))
sns.set_style('white')
sns.barplot(x='importance', y='features', data=interview_imp.head(20), ax=ax, palette='pastel')

display(interview_imp)


# COMMAND ----------

print(interview_imp.features.to_list())

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #SHAP beeswarm plots

# COMMAND ----------

import shap

print ("desk check model")
# Initialize the SHAP TreeExplainer with the model
explainer = shap.TreeExplainer(fa_model)

# Calculate SHAP values for the test dataset
shap_values = explainer.shap_values(df_prep)

# Generate a summary plot of the SHAP values
shap.summary_plot(shap_values, df_prep)

display(len(df_prep.columns))

# COMMAND ----------

print ("interview model")
# Initialize the SHAP TreeExplainer with the model
explainer = shap.TreeExplainer(interview_model)

# Calculate SHAP values for the test dataset
shap_values = explainer.shap_values(df_prep_interview)

# Generate a summary plot of the SHAP values
shap.summary_plot(shap_values, df_prep_interview)



# COMMAND ----------

# MAGIC %md
# MAGIC #PDPs

# COMMAND ----------

# Define function to plot partial dependence for a list of features, given a model and dataset

def plot_pdps(model, df, feature_list=[]):
  """
  Plots PDPs with volume distributions, for a given set of features
  for a given model and dataset

  Parameters:
  ----------
  model : trained model
      Function will accept any model that works with the partial dependence function from sklearn.inspection
      This includes most tree based models

  df : pandas dataframe
      Dataset which you wish to calculate PDPs for

  feature_list (optional) : list of strings
      denoting names of features from the dataset for which you wish to plot partial dependence

  Returns:
  -------
  null (Will plot to cell output the required charts)

  """
  # by default, use all features in dataset
  if feature_list == []:
    feature_list = df.columns
  else:
    # check if passed features are in dataset, throw error if not
    for feature in feature_list:
      if feature not in df.columns:
        raise ValueError("Feature '" + str(feature) + "' not in dataset")

  # iterate through features, plot chart for each
  for feature in feature_list:
    # calculate partial dependence
    pdp_result = partial_dependence(model, df, [feature], kind="average")

    # We want to plot on same graph the distribution of the feature, so next section of code get the counts for 
    # each value evauluated as part of the partial dependence calculations
    
    # Get list of analysed grid values of feature
    grid_values = pdp_result["values"][0]
    #grid_values = pdp_result["grid_values"][0]
    
    # Get list of raw values of feature
    feature_values = np.array(df[feature])
    # Count how many cases fall into each grid value
    counts = Counter(np.digitize(feature_values, grid_values))
    # Look up by index to get ordered volume for each grid value
    freq = [counts[i+1] for i in range(len(grid_values))]

    # Plot the PDPs and Feature distributions on the same graph
    fig, ax1 = plt.subplots(figsize=(12,8))

    # Plot the PDP on the primary y-axis
    ax1.plot(grid_values, pdp_result['average'][0], label='Partial Dependence', marker='o', color='blue')
    ax1.set_xlabel('Value of ' + str(feature))
    ax1.set_ylabel('Partial dependence')
    ax1.tick_params(axis="y")

    grid_range = max(grid_values) - min(grid_values)

    # Create a second y-axis for the volume distribution
    ax2 = ax1.twinx() 
    ax2.bar(grid_values, freq, color="grey", alpha=0.4, width = grid_range/len(grid_values)*0.9, label="Volume")
    ax2.set_ylabel("Volume")
    ax2.tick_params(axis="y")

    fig.legend()

    plt.title('PDP for ' + str(feature))
    plt.show()
    
    # End of for loop for each feature

  # End of plot_pdps function

# COMMAND ----------

from sklearn.inspection import PartialDependenceDisplay
#for desk check model
features_list =[ 'num__policyholder_ncd_years', 'cat__C2_reporting_delay_1', 'num__incidentHourC', 'cat__incident_cause_Lost Control - No Third Party Involved', 'cat__assessment_category_UnroadworthyTotalLoss', 'cat__C5_is_night_incident_1', 'num__manufacture_yr_claim', 'cat__impact_speed_range_Unknown', 'cat__is_police_attendance_1', 'cat__incident_cause_Debris/Object', 'num__min_age_at_policy_start_date', 'num__max_years_resident_in_uk', 'cat__incident_sub_cause_Avoiding known TP vehicle', 'cat__assessment_category_None', 'cat__assessment_category_DriveableRepair', 'cat__is_crime_reference_provided_-1', 'cat__ncd_protected_flag_-1', 'cat__incidentMonthC_5', 'cat__incident_cause_Attempted To Avoid Collision', 'cat__incidentMonthC_1', 'cat__ncd_protected_flag_0', 'cat__incidentMonthC_4', 'cat__left_front_wheel_severity_missing', 'cat__vehicle_overnight_location_id_3.0', 'cat__first_party_confirmed_tp_notified_claim_1', 'num__areasDamagedSevere', 'cat__incident_sub_cause_Street Furniture & Highways', 'cat__right_front_wheel_severity_missing', 'num__inception_to_claim', 'cat__front_severity_missing', 'cat__incident_sub_cause_Avoiding Unknown TP vehicle', 'num__impact_speed', 'cat__incidentMonthC_2', 'num__vehicle_value' ]

num_features = [x for x in features_list if 'num' in str.lower(x)]
cat_features = [x for x in features_list if x not in num_features]

def plot_pdps(model, df, feature_list=[], per_row=2):
    num_plots = len(feature_list)
    num_rows = (num_plots + per_row - 1) // per_row

    fig, axes = plt.subplots(num_rows, per_row, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(feature_list):
        ax = axes[i]
        #ax.set_title(f"PDP for {feature}")
        pfig = PartialDependenceDisplay.from_estimator(
            model, df, [feature], ax=ax, n_jobs=-1, grid_resolution=10,
        )
        pfig.axes_[0,0].set_xlabel(feature, fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_pdps(fa_model, df_prep, num_features)


# COMMAND ----------

def categorical_pdps(model, df, feature_list=[], per_row=2):
    num_plots = len(feature_list)
    num_rows = (num_plots + per_row - 1) // per_row

    fig, axes = plt.subplots(num_rows, per_row, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(feature_list):
        ax = axes[i]
        #ax.set_title(f"PDP for {feature}")
        pfig = PartialDependenceDisplay.from_estimator(
            model, df, [feature], ax=ax, n_jobs=-1, grid_resolution=5,
            categorical_features=[feature]
        )
        #ax.set_xlabel(feature, fontsize=35)
        pfig.axes_[0,0].set_xlabel(feature, fontsize=15)


    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

categorical_pdps(fa_model, df_prep, cat_features[0:7])

# COMMAND ----------



# COMMAND ----------

preprocessor = interview_pipeline.named_steps['preprocessor']
interview_model = interview_pipeline.named_steps['classifier']
df_prep_interview = pd.DataFrame(preprocessor.transform(raw_df), columns=preprocessor.get_feature_names_out())
interview_features = nz_interview_imp.features
num_features = [x for x in nz_interview_imp.features if 'num' in str.lower(x)]
cat_features = [x for x in nz_interview_imp.features if x not in num_features]

plot_pdps(interview_model, df_prep_interview, num_features, per_row=4)

# COMMAND ----------

categorical_pdps(interview_model, df_prep_interview, cat_features, per_row=2)


# COMMAND ----------

#re-evaluate on chosen threshold
fa_threshold = 0.5
interview_threshold = 0.5
mean_row = evaluate_thresholds(fa_threshold, interview_threshold)

# COMMAND ----------

ref_rate = (raw_df.groupby(['y_cmb']).agg(fraud_rate=('svi_risk', 'mean'),
                               fraud_count=('svi_risk', 'sum'),
                               cases=('y_cmb', 'count')).round(2)).reset_index()
display(ref_rate)
ref_rate.fraud_rate.plot(kind='bar', color='lightgreen', xlabel='model prediction', ylabel='actual fraud rate')
