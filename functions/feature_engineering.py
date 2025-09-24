"""
Feature Engineering Functions for SVI Fraud Detection
Standalone functions extracted from notebooks for feature engineering
"""

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, FloatType
from pyspark.sql import Window
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from functools import reduce


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
    
    # Calculate checks_max and checks_list
    if check_cols:
        df = df.withColumn('checks_max', greatest(*[col(c) for c in check_cols]))
        
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
    driver_cols = [
        'additional_vehicles_owned', 'age_at_policy_start_date', 
        'cars_in_household', 'licence_length_years', 'years_resident_in_uk'
    ]
    
    for col_name in driver_cols:
        # Check if columns exist
        existing_cols = [f"{col_name}_{i}" for i in range(1, 6) if f"{col_name}_{i}" in df.columns]
        
        if existing_cols:
            df = df.withColumn(
                f"max_{col_name}", 
                greatest(*[col(c) for c in existing_cols])
            )
            df = df.withColumn(
                f"min_{col_name}", 
                least(*[col(c) for c in existing_cols])
            )
    
    return df


def fill_missing_values_spark(df):
    """Fill missing values with appropriate strategies for PySpark DataFrame"""
    # Mean fill columns
    mean_fills = [
        "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", 
        "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", 
        "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", 
        "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", 
        "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", 
        "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", 
        "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", 
        "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end"
    ]
    
    # Boolean or damage columns with neg fills
    neg_fills = [
        "vehicle_unattended", "excesses_applied", "is_first_party", 
        "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
        "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance", 
        "veh_age_more_than_10", "damageScore", "areasDamagedMinimal", "areasDamagedMedium", 
        "areasDamagedHeavy", "areasDamagedSevere", "areasDamagedTotal", 
        "police_considering_actions", "is_crime_reference_provided", "ncd_protected_flag", 
        "boot_opens", "doors_open", "multiple_parties_involved", "is_incident_weekend", 
        "is_reported_monday", "driver_age_low_1", "claim_driver_age_low", "licence_low_1", 
        "total_loss_flag"
    ]
    
    # Fills with ones (rules variables, to trigger manual check)
    one_fills = [
        "C1_fri_sat_night", "C2_reporting_delay", "C3_weekend_incident_reported_monday", 
        "C5_is_night_incident", "C6_no_commuting_but_rush_hour", 
        "C7_police_attended_or_crime_reference", "C9_policy_within_30_days", 
        "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
        "C12_expensive_for_driver_age", "C14_contains_watchwords"
    ]
    
    # Fill with word 'missing' (categoricals)
    string_cols = [
        'vehicle_overnight_location_id', 'incidentDayOfWeekC', 'incidentMonthC', 
        'policy_type', 'assessment_category', 'engine_damage', 
        'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 
        'policy_cover_type', 'notification_method', 'impact_speed_unit', 
        'impact_speed_range', 'incident_type', 'incident_cause', 
        'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 
        'front_left_severity', 'front_right_severity', 'left_severity', 
        'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 
        'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 
        'rear_left_severity', 'rear_right_severity', 'rear_window_damage_severity', 
        'right_severity', 'right_back_seat_severity', 'right_front_wheel_severity', 
        'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 
        'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 
        'windscreen_damage_severity', 'incident_day_of_week', 'reported_day_of_week'
    ]
    
    # Apply fills
    fill_dict = {}
    
    for col_name in mean_fills:
        if col_name in df.columns:
            fill_dict[col_name] = 0
    
    for col_name in neg_fills:
        if col_name in df.columns:
            fill_dict[col_name] = -1
    
    for col_name in one_fills:
        if col_name in df.columns:
            fill_dict[col_name] = 1
    
    for col_name in string_cols:
        if col_name in df.columns:
            fill_dict[col_name] = 'missing'
    
    df = df.fillna(fill_dict)
    
    return df


def create_additional_features(df):
    """Create additional engineered features"""
    
    # Vehicle age calculation
    if "manufacture_yr_claim" in df.columns and "start_date" in df.columns:
        df = df.withColumn("veh_age", 
                          round(datediff(col("start_date"), 
                                       to_date(concat(col("manufacture_yr_claim"), 
                                                     lit('-01-01')))) / 365.25, 0))
        df = df.withColumn("veh_age_more_than_10", (col("veh_age") > 10).cast("int"))
    
    # Total damage areas
    if all(col_name in df.columns for col_name in ["areasDamagedMinimal", "areasDamagedMedium", 
                                                    "areasDamagedSevere", "areasDamagedHeavy"]):
        df = df.withColumn("areasDamagedTotal", 
                          col("areasDamagedMinimal") + col("areasDamagedMedium") + 
                          col("areasDamagedSevere") + col("areasDamagedHeavy"))
    
    # High risk indicators
    if "min_claim_driver_age" in df.columns:
        df = df.withColumn("claim_driver_age_low", 
                          when(col("min_claim_driver_age") < 25, 1)
                          .when(col("min_claim_driver_age").isNull(), 1)
                          .otherwise(0))
    
    return df


def select_modeling_features(df):
    """Select features for modeling"""
    # Define feature groups
    numeric_features = [
        'policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 
        'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 
        'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 
        'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 
        'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 
        'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 
        'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 
        'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim', 
        'outstanding_finance_amount', 'claim_to_policy_end', 'incidentDayOfWeekC', 
        'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy', 
        'areasDamagedSevere', 'areasDamagedTotal', 'areasDamagedRelative', 
        'num_failed_checks', 'delay_in_reporting'
    ]
    
    categorical_features = [
        'vehicle_unattended', 'excesses_applied', 'is_first_party', 
        'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 
        'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 
        'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 
        'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 
        'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 
        'claim_driver_age_low', 'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 
        'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
        'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 
        'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 
        'C12_expensive_for_driver_age', 'C14_contains_watchwords', 
        'vehicle_overnight_location_id', 'incidentMonthC', 'policy_type', 
        'assessment_category', 'engine_damage', 'sales_channel', 
        'overnight_location_abi_code', 'vehicle_overnight_location_name', 
        'policy_cover_type', 'notification_method', 'impact_speed_unit', 
        'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 
        'front_severity', 'front_bonnet_severity', 'front_left_severity', 
        'front_right_severity', 'left_severity', 'left_back_seat_severity', 
        'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
        'left_underside_severity', 'rear_severity', 'rear_left_severity', 
        'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 
        'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 
        'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 
        'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 
        'incident_day_of_week', 'reported_day_of_week', 'checks_max', 'total_loss_flag'
    ]
    
    id_features = [
        'claim_number', 'claim_id', 'policy_number', 'dataset', 'start_date', 
        'reported_date', 'checks_list'
    ]
    
    target_features = ['svi_risk', 'tbg_risk', 'fa_risk']
    
    # Get all available features
    all_features = numeric_features + categorical_features + id_features + target_features
    
    # Filter to only existing columns
    existing_features = [col for col in all_features if col in df.columns]
    
    return df.select(existing_features)
</content># Standalone feature engineering functions extracted from notebooks

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from functools import reduce
from operator import add

def apply_damage_score_calculation(df):
    """Apply damage score calculations to the dataframe"""
    
    # Calculate relative damage score
    df = df.withColumn("areasDamagedRelative", 
                      col("areasDamagedMinimal") + 
                      2*col("areasDamagedMedium") + 
                      3*col("areasDamagedSevere") + 
                      4*col("areasDamagedHeavy"))
    
    # Calculate maximum damage severity
    damage_cols = [
        'boot_opens', 'doors_open', 'front_severity', 'front_bonnet_severity',
        'front_left_severity', 'front_right_severity', 'left_severity',
        'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity',
        'left_rear_wheel_severity', 'left_underside_severity',
        'rear_severity', 'rear_left_severity', 'rear_right_severity',
        'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity',
        'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity',
        'right_roof_severity', 'right_underside_severity', 'roof_damage_severity',
        'underbody_damage_severity', 'windscreen_damage_severity'
    ]
    
    # Create mapping for severity to numeric value
    severity_map = when(col("severity") == "Minimal", 1)\
        .when(col("severity") == "Medium", 2)\
        .when(col("severity") == "Heavy", 3)\
        .when(col("severity") == "Severe", 4)\
        .otherwise(0)
    
    # Calculate damage_sev_max
    max_severity_exprs = []
    for damage_col in damage_cols:
        if damage_col in df.columns:
            max_severity_exprs.append(
                when(col(damage_col) == "Severe", 4)
                .when(col(damage_col) == "Heavy", 3)
                .when(col(damage_col) == "Medium", 2)
                .when(col(damage_col) == "Minimal", 1)
                .otherwise(0)
            )
    
    if max_severity_exprs:
        df = df.withColumn("damage_sev_max", greatest(*max_severity_exprs))
    
    return df


def create_time_based_features(df):
    """Create time-based features from dates"""
    
    # Day of week features
    df = df.withColumn("incident_day_of_week", dayofweek(col("start_date")))
    df = df.withColumn("reported_day_of_week", dayofweek(col("reported_date")))
    
    # Weekend flags
    df = df.withColumn(
        "incident_weekend",
        when(col("incident_day_of_week").isin(1, 7), 1).otherwise(0)
    )
    df = df.withColumn(
        "is_incident_weekend",
        col("incident_weekend")
    )
    
    # Monday reporting flag
    df = df.withColumn(
        "reported_monday",
        when(col("reported_day_of_week") == 2, 1).otherwise(0)
    )
    df = df.withColumn(
        "is_reported_monday",
        col("reported_monday")
    )
    
    # Hour and month features
    df = df.withColumn("incidentHourC", hour(col("start_date")))
    df = df.withColumn("incidentDayOfWeekC", col("incident_day_of_week"))
    df = df.withColumn("incidentMonthC", month(col("start_date")))
    
    # Night incident flag
    df = df.withColumn(
        "night_incident",
        when((hour(col("start_date")) >= 23) | (hour(col("start_date")) <= 5), 1).otherwise(0)
    )
    
    # High risk combo: weekend + night
    df = df.withColumn(
        "high_risk_combo",
        when((col("incident_weekend") == 1) & (col("night_incident") == 1), 1).otherwise(0)
    )
    
    return df


def calculate_delays(df):
    """Calculate various delay features"""
    
    # Inception to claim delay
    df = df.withColumn(
        "inception_to_claim",
        datediff(col("start_date"), col("policy_start_date"))
    )
    df = df.withColumn(
        "inception_to_claim_days",
        col("inception_to_claim")
    )
    
    # Claim to policy end delay
    df = df.withColumn(
        "claim_to_policy_end",
        datediff(col("policy_renewal_date"), col("start_date"))
    )
    
    # Delay in reporting
    df = df.withColumn(
        "delay_in_reporting",
        datediff(col("reported_date"), col("start_date"))
    )
    
    # Vehicle age at claim
    df = df.withColumn(
        "veh_age",
        year(col("start_date")) - col("vehicle_year")
    )
    
    # Manufacture year at claim
    df = df.withColumn(
        "manufacture_yr_claim",
        col("vehicle_year")
    )
    
    # Vehicle age flag
    df = df.withColumn(
        "veh_age_more_than_10",
        when(col("veh_age") > 10, 1).otherwise(0)
    )
    
    return df


def create_driver_features(df):
    """Create driver-related features"""
    
    # Driver age low flags
    df = df.withColumn(
        "driver_age_low_1",
        when(col("age_at_policy_start_date_1") < 25, 1).otherwise(0)
    )
    
    df = df.withColumn(
        "claim_driver_age_low",
        when(col("min_claim_driver_age") < 25, 1).otherwise(0)
    )
    
    # Licence low flag
    df = df.withColumn(
        "licence_low_1",
        when(col("licence_length_years_1") <= 3, 1).otherwise(0)
    )
    
    return df


def create_policy_features(df):
    """Create policy-related features"""
    
    # Vehicle use quote (1 = not commuting)
    df = df.withColumn(
        "vehicle_use_quote",
        when(col("vehicle_use") != "Commuting", 1).otherwise(0)
    )
    
    # Total loss flag
    df = df.withColumn(
        "total_loss_flag",
        when(col("assessment_category").isin("DriveableTotalLoss", "UnroadworthyTotalLoss"), True).otherwise(False)
    )
    
    # Total loss new
    df = df.withColumn(
        "total_loss_new",
        when(col("total_loss_flag") == True, 1).otherwise(0)
    )
    
    return df


def handle_missing_values_fe(df):
    """Handle missing values in feature engineering"""
    
    # Numeric columns - fill with 0
    numeric_fill_zero = [
        'damage_score', 'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium',
        'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal',
        'areasDamagedRelative', 'damage_sev_max'
    ]
    
    for col_name in numeric_fill_zero:
        if col_name in df.columns:
            df = df.fillna({col_name: 0})
    
    # Boolean columns - fill with 0
    boolean_cols = [
        'incident_weekend', 'is_incident_weekend', 'reported_monday', 'is_reported_monday',
        'night_incident', 'high_risk_combo', 'veh_age_more_than_10',
        'driver_age_low_1', 'claim_driver_age_low', 'licence_low_1'
    ]
    
    for col_name in boolean_cols:
        if col_name in df.columns:
            df = df.fillna({col_name: 0})
    
    return df


def create_aggregated_features(df):
    """Create aggregated features across drivers and policies"""
    
    # Check if aggregated columns already exist from preprocessing
    driver_agg_cols = [
        'max_additional_vehicles_owned', 'min_additional_vehicles_owned',
        'max_age_at_policy_start_date', 'min_age_at_policy_start_date',
        'max_cars_in_household', 'min_cars_in_household',
        'max_licence_length_years', 'min_licence_length_years',
        'max_years_resident_in_uk', 'min_years_resident_in_uk'
    ]
    
    # If they don't exist, create them from base columns
    for col_base in ['additional_vehicles_owned', 'age_at_policy_start_date',
                     'cars_in_household', 'licence_length_years', 'years_resident_in_uk']:
        
        # Check for columns with _1, _2, etc. suffixes
        matching_cols = [c for c in df.columns if c.startswith(col_base + "_")]
        
        if matching_cols and f"max_{col_base}" not in df.columns:
            # Calculate max across all matching columns
            df = df.withColumn(f"max_{col_base}", greatest(*[col(c) for c in matching_cols]))
            df = df.withColumn(f"min_{col_base}", least(*[col(c) for c in matching_cols]))
    
    return df


def select_modeling_features(df):
    """Select final features for modeling"""
    
    # Define feature groups
    target_cols = ['svi_risk', 'tbg_risk', 'fa_risk', 'referred_to_tbg']
    
    id_cols = ['claim_number', 'dataset', 'start_date', 'reported_date', 
               'checks_list', 'Outcome_of_Investigation', 'position_status']
    
    numeric_features = [
        'damage_score', 'damageScore', 'damage_sev_max',
        'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy',
        'areasDamagedSevere', 'areasDamagedTotal', 'areasDamagedRelative',
        'min_claim_driver_age', 'vehicle_value',
        'inception_to_claim', 'inception_to_claim_days', 'claim_to_policy_end',
        'voluntary_amount', 'policyholder_ncd_years', 'annual_mileage',
        'veh_age', 'business_mileage', 'impact_speed', 'manufacture_yr_claim',
        'outstanding_finance_amount', 'excesses_applied', 
        'incidentDayOfWeekC', 'incidentHourC', 'incidentMonthC',
        'reported_day_of_week', 'delay_in_reporting'
    ]
    
    check_features = [
        'C1_fri_sat_night', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday',
        'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
        'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days',
        'C10_claim_to_policy_end', 'C11_young_or_inexperienced',
        'C12_expensive_for_driver_age', 'C14_contains_watchwords'
    ]
    
    categorical_features = [
        'is_driveable', 'policy_type', 'sales_channel',
        'incident_day_of_week', 'vehicle_overnight_location_id',
        'assessment_category', 'engine_damage',
        'overnight_location_abi_code', 'vehicle_overnight_location_name',
        'policy_cover_type', 'notification_method', 'impact_speed_unit',
        'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause'
    ]
    
    driver_features = [
        'max_age_at_policy_start_date', 'min_age_at_policy_start_date',
        'max_licence_length_years', 'min_licence_length_years',
        'max_additional_vehicles_owned', 'min_additional_vehicles_owned',
        'max_cars_in_household', 'min_cars_in_household',
        'max_years_resident_in_uk', 'min_years_resident_in_uk',
        'additional_vehicles_owned_1', 'age_at_policy_start_date_1',
        'cars_in_household_1', 'licence_length_years_1', 'years_resident_in_uk_1'
    ]
    
    flag_features = [
        'num_failed_checks', 'night_incident', 'high_risk_combo',
        'ncd_protected_flag', 'vehicle_unattended', 'is_first_party',
        'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance',
        'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance',
        'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided',
        'multiple_parties_involved', 'is_incident_weekend', 'is_reported_monday',
        'driver_age_low_1', 'claim_driver_age_low', 'licence_low_1', 
        'checks_max', 'total_loss_flag', 'total_loss_new'
    ]
    
    damage_cols = [
        'boot_opens', 'doors_open', 'front_severity', 'front_bonnet_severity',
        'front_left_severity', 'front_right_severity', 'left_severity',
        'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity',
        'left_rear_wheel_severity', 'left_underside_severity',
        'rear_severity', 'rear_left_severity', 'rear_right_severity',
        'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity',
        'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity',
        'right_roof_severity', 'right_underside_severity', 'roof_damage_severity',
        'underbody_damage_severity', 'windscreen_damage_severity'
    ]
    
    other_cols = ['vehicle_use_quote', 'circumstances', 'Circumstances']
    
    # Combine all features
    all_features = (id_cols + target_cols + numeric_features + 
                   check_features + categorical_features +
                   driver_features + flag_features + damage_cols + other_cols)
    
    # Select only columns that exist
    existing_features = [col for col in all_features if col in df.columns]
    
    return df.select(existing_features)