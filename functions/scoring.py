"""
Model Scoring Functions for SVI Fraud Detection
Standalone functions extracted from notebooks for model scoring and daily pipelines
"""

import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql import DataFrame
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_claims_data(spark, date_str):
    """Load claims data for scoring"""
    table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_claims_svi"
    
    # Define check columns
    check_cols = [
        'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
        'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
        'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
        'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
    ]
    
    # Read in dataset
    other_cols = ['claim_number', 'reported_date', 'start_date', 'num_failed_checks',
                  'total_loss_flag', 'checks_list', 'delay_in_reporting', 'claim_id', 
                  'position_status', 'vehicle_use_quote', 'Circumstances'] + \
                  [x for x in check_cols if x not in categorical_features]
    
    raw_df = spark.table(table_path).withColumn('checks_max', greatest(*[col(c) for c in check_cols]).cast('string'))\
                    .withColumn("underbody_damage_severity", lit(None))\
                    .filter(f"DATE(reported_date)='{date_str}'")
    
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
    
    # Define features for select
    numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 
                       'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 
                       'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 
                       'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 
                       'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 
                       'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 
                       'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 
                       'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 
                       'incidentDayOfWeekC', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 
                       'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal']

    categorical_features = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 
                           'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 
                           'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 
                           'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 
                           'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 
                           'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 
                           'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 
                           'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
                           'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 
                           'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 
                           'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 
                           'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 
                           'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 
                           'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 
                           'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 
                           'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 
                           'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
                           'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 
                           'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
                           'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 
                           'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 
                           'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 
                           'reported_day_of_week', 'checks_max', 'total_loss_flag']
    
    # Convert to pandas
    raw_df = raw_df.select(numeric_features + categorical_features + other_cols).toPandas()
    
    raw_df['reported_month'] = pd.to_datetime(raw_df['reported_date']).dt.month
    raw_df['reported_year'] = pd.to_datetime(raw_df['reported_date']).dt.year
    
    raw_df['num_failed_checks'] = raw_df['num_failed_checks'].astype('float64')
    raw_df['score_card'] = ((raw_df['delay_in_reporting'] > 3) | (raw_df['policyholder_ncd_years'] < 2)).astype(int)
    
    return raw_df


def set_types(raw_df):
    """Recast data types & check schema"""
    numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 
                       'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 
                       'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 
                       'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 
                       'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 
                       'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 
                       'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 
                       'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 
                       'incidentDayOfWeekC', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 
                       'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal', 'num_failed_checks']

    categorical_features = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 
                           'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 
                           'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 
                           'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 
                           'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 
                           'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 
                           'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 
                           'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
                           'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 
                           'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 
                           'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 
                           'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 
                           'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 
                           'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 
                           'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 
                           'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 
                           'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
                           'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 
                           'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
                           'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 
                           'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 
                           'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 
                           'reported_day_of_week', 'checks_max', 'total_loss_flag']
    
    cols_groups = {
        "float": numeric_features,
        "string": categorical_features 
    }
    
    for dtype, column_list in cols_groups.items():
        existing_cols = [col for col in column_list if col in raw_df.columns]
        if dtype == "float":
            raw_df[existing_cols] = raw_df[existing_cols].astype(float)
        elif dtype == "integer":
            raw_df[existing_cols] = raw_df[existing_cols].astype(int)
        elif dtype == "string":
            raw_df[existing_cols] = raw_df[existing_cols].astype('str')    
        elif dtype == "bool":
            raw_df[existing_cols] = raw_df[existing_cols].astype(int).astype('str')

    return raw_df


def do_fills_pd(raw_df, mean_dict):
    """Apply missing value imputation"""
    # Columns to fill using mean
    mean_fills = ["policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", 
                  "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", 
                  "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", 
                  "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", 
                  "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", 
                  "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", 
                  "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", 
                  "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", 
                  "incidentDayOfWeekC", "num_failed_checks"]

    # Boolean or damage columns with neg fills
    damage_cols = ["damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy",
                   "areasDamagedSevere","areasDamagedTotal"]
    bool_cols = ["vehicle_unattended","excesses_applied","is_first_party",
                 "first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance",
                 "is_ambulance_attendance","is_fire_service_attendance","is_police_attendance",
                 "veh_age_more_than_10","police_considering_actions","is_crime_reference_provided",
                 "ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved", 
                 "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low",
                 "licence_low_1", "total_loss_flag"]

    # Fills with ones (rules variables, to trigger manual check)
    one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday",
                 "C5_is_night_incident","C6_no_commuting_but_rush_hour",
                 "C7_police_attended_or_crime_reference","C9_policy_within_30_days", 
                 "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
                 "C12_expensive_for_driver_age", "C14_contains_watchwords"]

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
    ]) - set(['car_group', 'employment_type_abi_code_5', 'employment_type_abi_code_4', 
               'employment_type_abi_code_3', 'employment_type_abi_code_2', 'postcode', 'employment_type_abi_code_1']))
    
    # Recast data types & check schema
    cols_groups = {
        "float": mean_fills + damage_cols,
        "string": string_cols,
        "bool": bool_cols + one_fills 
    }
    
    # Fillna columns
    neg_fills_dict = {x: -1 for x in bool_cols + damage_cols if x in raw_df.columns}
    one_fills_dict = {x: 1 for x in one_fills if x in raw_df.columns}
    string_fills_dict = {x: 'missing' for x in string_cols if x in raw_df.columns}
    combined_fills = {**one_fills_dict, **neg_fills_dict, **string_fills_dict, **mean_dict}
    raw_df = raw_df.fillna(combined_fills)

    for dtype, column_list in cols_groups.items():
        existing_cols = [col for col in column_list if col in raw_df.columns]
        if dtype == "float":
            raw_df[existing_cols] = raw_df[existing_cols].astype(float)
        elif dtype == "integer":
            raw_df[existing_cols] = raw_df[existing_cols].astype(int)
        elif dtype == "string":
            raw_df[existing_cols] = raw_df[existing_cols].astype('str')        
        elif dtype == "bool":
            raw_df[existing_cols] = raw_df[existing_cols].astype(int).astype('str')

    return raw_df


# ============================================================================
# Business Rules Functions
# ============================================================================

def apply_business_rules(raw_df, fa_threshold=0.5, interview_threshold=0.5):
    """Apply business rules and overrides to model predictions"""
    
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
    
    return raw_df


def score_with_models(raw_df, fa_pipeline, interview_pipeline):
    """Score claims with both FA and Interview models"""
    
    # Define features for each model
    numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 
                       'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 
                       'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 
                       'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 
                       'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 
                       'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 
                       'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 
                       'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 
                       'incidentDayOfWeekC', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 
                       'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal']

    categorical_features = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 
                           'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 
                           'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 
                           'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 
                           'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 
                           'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 
                           'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 
                           'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
                           'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 
                           'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 
                           'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 
                           'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 
                           'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 
                           'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 
                           'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 
                           'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 
                           'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
                           'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 
                           'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
                           'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 
                           'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 
                           'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 
                           'reported_day_of_week', 'checks_max', 'total_loss_flag']
    
    # Score with FA (Desk Check) model
    existing_features = [f for f in numeric_features + categorical_features if f in raw_df.columns]
    raw_df['fa_pred'] = fa_pipeline.predict_proba(raw_df[existing_features])[:,1].round(4)
    
    # Interview model features
    num_interview = ['voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 
                     'annual_mileage', 'min_claim_driver_age', 'incidentHourC', 
                     'areasDamagedHeavy', 'impact_speed', 'years_resident_in_uk_1', 
                     'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 
                     'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 'num_failed_checks',
                     'areasDamagedMedium', 'min_years_resident_in_uk', 'incidentDayOfWeekC',
                     'age_at_policy_start_date_1', 'max_age_at_policy_start_date']
    
    cat_interview = ['assessment_category', 'left_severity', 'C9_policy_within_30_days',
                     'incident_sub_cause', 'rear_right_severity', 'front_left_severity',
                     'rear_window_damage_severity', 'incident_cause', 'checks_max',
                     'total_loss_flag']
    
    # Filter to existing columns
    interview_features = [f for f in num_interview + cat_interview if f in raw_df.columns]
    
    # Score with Interview model
    raw_df['y_prob2'] = interview_pipeline.predict_proba(raw_df[interview_features])[:,1].round(4)
    
    return raw_df


def save_predictions_to_table(spark, raw_df_pd, model_version=0.1):
    """Save prediction results to Spark table"""
    
    # Add model version
    raw_df_pd['model_version'] = model_version
    
    # Convert back to Spark DataFrame
    raw_df_spark = spark.createDataFrame(raw_df_pd)
    
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
    
    return raw_df_spark.count()


def generate_scoring_summary(raw_df, date_str, model_version=0.1):
    """Generate summary statistics for scoring results"""
    
    summary = {
        'date': date_str,
        'model_version': model_version,
        'total_claims_scored': len(raw_df),
        'high_risk_claims': raw_df['y_cmb'].sum(),
        'high_risk_percentage': raw_df['y_cmb'].sum()/len(raw_df)*100,
        'flagged_by_model': raw_df['flagged_by_model'].sum(),
        'late_night_no_commuting': raw_df['late_night_no_commuting'].sum(),
        'unconscious_flag': raw_df['unconscious_flag'].sum(),
        'avg_fa_score': raw_df['fa_pred'].mean(),
        'avg_interview_score': raw_df['y_prob2'].mean(),
        'avg_combined_rank': raw_df['y_rank_prob'].mean()
    }
    
    # Print summary
    print("="*60)
    print("SCORING SUMMARY")
    print("="*60)
    print(f"Date: {summary['date']}")
    print(f"Total claims scored: {summary['total_claims_scored']}")
    print(f"High risk claims: {summary['high_risk_claims']} ({summary['high_risk_percentage']:.2f}%)")
    print(f"\nRisk breakdown:")
    print(f"  - Flagged by model: {summary['flagged_by_model']}")
    print(f"  - Late night no commuting: {summary['late_night_no_commuting']}")
    print(f"  - Unconscious flag: {summary['unconscious_flag']}")
    print(f"\nAverage scores:")
    print(f"  - FA model: {summary['avg_fa_score']:.4f}")
    print(f"  - Interview model: {summary['avg_interview_score']:.4f}")
    print(f"  - Combined rank: {summary['avg_combined_rank']:.4f}")
    print("="*60)
    
    return summary


def load_models_from_mlflow(fa_run_id, interview_run_id):
    """Load trained models from MLflow"""
    
    # Load models
    fa_pipeline = mlflow.sklearn.load_model(f'runs:/{fa_run_id}/model')
    interview_pipeline = mlflow.sklearn.load_model(f'runs:/{interview_run_id}/model')
    
    print(f"Loaded FA model from run: {fa_run_id}")
    print(f"Loaded Interview model from run: {interview_run_id}")
    
    return fa_pipeline, interview_pipeline


# ============================================================================
# Default Mean Dictionary for Imputation
# ============================================================================

def get_default_mean_dict():
    """Return default mean dictionary for imputation"""
    return {
        'policyholder_ncd_years': 6.7899, 'inception_to_claim': 141.2893, 'min_claim_driver_age': 37.5581, 
        'veh_age': 11.3038, 'business_mileage': 306.2093, 'annual_mileage': 7372.2649, 'incidentHourC': 12.8702, 
        'additional_vehicles_owned_1': 0.0022, 'age_at_policy_start_date_1': 39.4507, 'cars_in_household_1': 1.8289, 
        'licence_length_years_1': 15.3764, 'years_resident_in_uk_1': 34.6192, 'max_additional_vehicles_owned': 0.003, 
        'min_additional_vehicles_owned': 0.0013, 'max_age_at_policy_start_date': 43.1786, 
        'min_age_at_policy_start_date': 35.4692, 'max_cars_in_household': 1.8861, 'min_cars_in_household': 1.7626, 
        'max_licence_length_years': 18.3106, 'min_licence_length_years': 12.2208, 'max_years_resident_in_uk': 38.5058, 
        'min_years_resident_in_uk': 30.5888, 'impact_speed': 27.1128, 'voluntary_amount': 241.3595, 
        'vehicle_value': 7861.6867, 'manufacture_yr_claim': 2011.9375, 'outstanding_finance_amount': 0.0, 
        'claim_to_policy_end': 83.4337, 'incidentDayOfWeekC': 4.0115, 'num_failed_checks': 0
    }
</content># Standalone scoring functions extracted from notebooks

import pandas as pd
import numpy as np
from pyspark.sql.functions import *
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, List, Optional


def load_claims_data(spark, table_path, this_day):
    """Load claims data for scoring"""
    
    check_cols = [
        'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
        'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
        'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
        'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
    ]
    
    # Read in dataset
    other_cols = ['claim_number', 'reported_date', 'start_date', 'num_failed_checks',
                  'total_loss_flag', 'checks_list', 'delay_in_reporting', 'claim_id', 
                  'position_status', 'vehicle_use_quote', 'Circumstances']
    
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
    
    return raw_df


def do_fills_pd_scoring(raw_df, mean_dict):
    """Apply missing value imputation for scoring"""
    
    # Define fill columns
    mean_fills = ["policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", 
                  "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", 
                  "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", 
                  "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", 
                  "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", 
                  "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", 
                  "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", 
                  "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", 
                  "incidentDayOfWeekC", "num_failed_checks"]

    damage_cols = ["damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy",
                   "areasDamagedSevere","areasDamagedTotal"]
    
    bool_cols = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim",
                 "is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance",
                 "is_police_attendance","veh_age_more_than_10","police_considering_actions",
                 "is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open",
                 "multiple_parties_involved", "is_incident_weekend","is_reported_monday","driver_age_low_1",
                 "claim_driver_age_low","licence_low_1", "total_loss_flag"]

    one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday",
                 "C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference",
                 "C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
                 "C12_expensive_for_driver_age", "C14_contains_watchwords"]

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
    ]) - set(['car_group', 'employment_type_abi_code_5', 'employment_type_abi_code_4', 
              'employment_type_abi_code_3', 'employment_type_abi_code_2', 'postcode', 'employment_type_abi_code_1']))
    
    # Fillna columns
    neg_fills_dict = {x: -1 for x in bool_cols + damage_cols}
    one_fills_dict = {x: 1 for x in one_fills}
    string_fills_dict = {x: 'missing' for x in string_cols}
    combined_fills = {**one_fills_dict, **neg_fills_dict, **string_fills_dict, **mean_dict}
    raw_df = raw_df.fillna(combined_fills)
    
    # Recast data types
    cols_groups = {
        "float": mean_fills + damage_cols,
        "string": string_cols,
        "bool": bool_cols + one_fills 
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


def apply_business_rules(raw_df, fa_threshold=0.5, interview_threshold=0.5, model_version=0.1):
    """Apply business rules and overrides to model predictions"""
    
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
    raw_df['model_version'] = model_version
    
    return raw_df


def score_claims_batch(spark, claims_df, fa_pipeline, interview_pipeline, 
                       numeric_features, categorical_features, 
                       num_interview, cat_interview, mean_dict):
    """Score a batch of claims using trained models"""
    
    # Convert to pandas if needed
    if hasattr(claims_df, 'toPandas'):
        raw_df = claims_df.toPandas()
    else:
        raw_df = claims_df
    
    # Apply fills
    raw_df = do_fills_pd_scoring(raw_df, mean_dict)
    
    # Score with FA (Desk Check) model
    raw_df['fa_pred'] = fa_pipeline.predict_proba(raw_df[numeric_features + categorical_features])[:,1].round(4)
    
    # Score with Interview model
    raw_df['y_prob2'] = interview_pipeline.predict_proba(raw_df[num_interview + cat_interview])[:,1].round(4)
    
    # Apply business rules
    raw_df = apply_business_rules(raw_df)
    
    return raw_df


def save_predictions(spark, predictions_df, table_path):
    """Save prediction results to table"""
    
    # Convert back to Spark DataFrame
    raw_df_spark = spark.createDataFrame(predictions_df)
    
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
        .saveAsTable(table_path)
    
    return raw_df_spark.count()


def load_models_from_mlflow(fa_model_run_id, interview_run_id):
    """Load models from MLflow"""
    
    fa_pipeline = mlflow.sklearn.load_model(f'runs:/{fa_model_run_id}/model')
    interview_pipeline = mlflow.sklearn.load_model(f'runs:/{interview_run_id}/model')
    
    print(f"Loaded FA model from run: {fa_model_run_id}")
    print(f"Loaded Interview model from run: {interview_run_id}")
    
    return fa_pipeline, interview_pipeline


def get_feature_lists():
    """Get feature lists for scoring"""
    
    numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 
                       'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 
                       'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 
                       'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 
                       'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 
                       'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 
                       'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 
                       'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 
                       'incidentDayOfWeekC', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 
                       'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal']

    categorical_features = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 
                           'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 
                           'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 
                           'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 
                           'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 
                           'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 
                           'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 
                           'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
                           'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 
                           'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 
                           'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 
                           'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 
                           'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 
                           'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 
                           'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 
                           'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 
                           'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
                           'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 
                           'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
                           'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 
                           'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 
                           'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 
                           'reported_day_of_week', 'checks_max']

    num_interview = ['voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 'annual_mileage', 
                     'min_claim_driver_age', 'incidentHourC', 'areasDamagedHeavy', 'impact_speed', 
                     'years_resident_in_uk_1', 'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 
                     'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 'num_failed_checks', 
                     'areasDamagedMedium', 'min_years_resident_in_uk', 'incidentDayOfWeekC', 
                     'age_at_policy_start_date_1', 'max_age_at_policy_start_date']

    cat_interview = ['assessment_category', 'left_severity', 'C9_policy_within_30_days', 'incident_sub_cause', 
                     'rear_right_severity', 'front_left_severity', 'rear_window_damage_severity', 'incident_cause', 
                     'checks_max', 'total_loss_flag']
    
    return numeric_features, categorical_features, num_interview, cat_interview


def get_mean_dict():
    """Get pre-calculated mean dictionary for imputation"""
    
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
    
    return mean_dict


def print_scoring_summary(predictions_df, this_day):
    """Print summary statistics for scoring"""
    
    print("="*60)
    print("SCORING SUMMARY")
    print("="*60)
    print(f"Date: {this_day}")
    print(f"Total claims scored: {len(predictions_df)}")
    print(f"High risk claims: {predictions_df['y_cmb'].sum()} ({predictions_df['y_cmb'].sum()/len(predictions_df)*100:.2f}%)")
    print(f"\nRisk breakdown:")
    print(f"  - Flagged by model: {predictions_df['flagged_by_model'].sum()}")
    print(f"  - Late night no commuting: {predictions_df['late_night_no_commuting'].sum()}")
    print(f"  - Unconscious flag: {predictions_df['unconscious_flag'].sum()}")
    print(f"\nAverage scores:")
    print(f"  - FA model: {predictions_df['fa_pred'].mean():.4f}")
    print(f"  - Interview model: {predictions_df['y_prob2'].mean():.4f}")
    print(f"  - Combined rank: {predictions_df['y_rank_prob'].mean():.4f}")
    print("="*60)