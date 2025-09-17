import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
import re
from typing import List
from pyspark.sql.types import IntegerType, DoubleType, DateType, StringType
from pyspark.sql.window import Window
from functools import reduce
from operator import add

def read_and_process_delta(spark, file_path, useful_columns=None, drop_columns=None, skiprows=0):
    """Read a Delta table, select useful columns or drop unnecessary columns, and return the DataFrame."""
    print(file_path)
    df = spark.read.format("delta").table(file_path)
    
    # Select useful columns or drop unnecessary columns
    if useful_columns:
        df = df.select(*useful_columns)
    if drop_columns:
        df = df.drop(*drop_columns)
    
    # Skip rows if needed (in Spark this will be equivalent to filtering rows)
    if skiprows > 0:
        df = df.withColumn("row_index", row_number().over(Window.orderBy(monotonically_increasing_id())))
        df = df.filter(df.row_index > skiprows).drop("row_index")
    
    return df

def recast_dtype(raw_df: DataFrame, column_list: List[str], dtype: str) -> DataFrame:
    """
    Recast the data type of specified columns in a DataFrame.

    Parameters:
    raw_df (DataFrame): The current Spark DataFrame.
    column_list (List[str]): List of columns to cast to the given data type.
    dtype (str): The target data type to cast the columns to.

    Returns:
    DataFrame: A DataFrame with the specified columns cast to the given data type.
    """
    for column_name in column_list:
        raw_df = raw_df.withColumn(column_name, col(column_name).cast(dtype))
    
    return raw_df

class FeatureEngineering:
    """
    Feature engineering pipeline for SVI fraud detection.
    Creates derived features from preprocessed claim and policy data.
    """
    
    def __init__(self, spark, env_config):
        self.spark = spark
        self.env_config = env_config
        self.damage_score_udf = self.register_udf()

    def get_table_path(self, schema, table, catalog_type='auxiliary'):
        """
        Constructs a fully qualified table path based on the current environment.
        
        Parameters:
        schema: The schema name (e.g., 'single_vehicle_incident_checks')
        table: The table name (e.g., 'claim_referral_log')
        catalog_type: Either 'auxiliary' (for reading), 'mlstore' (for writing), or 'adp' (for ADP certified data)
        
        Returns:
        Fully qualified table path (e.g., 'prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claim_referral_log')
        """
        
        if catalog_type == 'auxiliary':
            catalog = self.env_config['auxiliary_catalog']
        elif catalog_type == 'mlstore':
            catalog = self.env_config['mlstore_catalog']
        elif catalog_type == 'adp':
            catalog = self.env_config['adp_catalog']
        else:
            raise ValueError(f"Invalid catalog_type: {catalog_type}. Must be 'auxiliary', 'mlstore', or 'adp'")
        
        return f"{catalog}.{schema}.{table}"

    def register_udf(self):
        """
        Register the UDF for calculating damage scores.
        """
        def calculate_damage_score(*args):
            """
            UDF to calculate damage score based on severity levels.
            
            Scoring:
            - 2x for Minimal
            - 3x for Medium
            - 4x for Heavy
            - 5x for Severe
            """
            severity_groups = ['Minimal', 'Medium', 'Heavy', 'Severe']
            group_weights = [2, 3, 4, 5]
            weight_dict = dict(zip(severity_groups, group_weights))
            
            total_score = 0
            for i in args:
                if i is not None and i in severity_groups:
                    total_score += weight_dict[i]
        
            return total_score

        # Register UDF
        return udf(calculate_damage_score, IntegerType())

    def apply_damage_score_calculation(self, df):
        """
        Apply damage score calculation to all damage severity columns.
        """
        # List of damage severity columns
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
        
        # Apply damage score UDF
        df = df.withColumn("damage_score", self.damage_score_udf(*damage_cols))
        
        # Also create damageScore alias for compatibility
        df = df.withColumn("damageScore", col("damage_score"))
        
        # Calculate areas damaged by severity (from experimental notebooks)
        df = df.withColumn(
            "areasDamagedMinimal",
            reduce(lambda a, b: a + b, [when(col(c) == "Minimal", 1).otherwise(0) for c in damage_cols])
        )
        df = df.withColumn(
            "areasDamagedMedium",
            reduce(lambda a, b: a + b, [when(col(c) == "Medium", 1).otherwise(0) for c in damage_cols])
        )
        df = df.withColumn(
            "areasDamagedHeavy",
            reduce(lambda a, b: a + b, [when(col(c) == "Heavy", 1).otherwise(0) for c in damage_cols])
        )
        df = df.withColumn(
            "areasDamagedSevere",
            reduce(lambda a, b: a + b, [when(col(c) == "Severe", 1).otherwise(0) for c in damage_cols])
        )
        df = df.withColumn(
            "areasDamagedTotal",
            col("areasDamagedMinimal") + col("areasDamagedMedium") + 
            col("areasDamagedHeavy") + col("areasDamagedSevere")
        )
        
        print("\nDamage score calculations completed")
        return df
    
    def create_time_based_features(self, df):
        """
        Create time-based features from dates.
        """
        # Day of week features
        df = df.withColumn("incident_day_of_week", dayofweek(col("latest_event_time")))
        df = df.withColumn("reported_day_of_week", dayofweek(col("latest_event_time")))
        
        # Weekend flags
        df = df.withColumn(
            "incident_weekend",
            when(col("incident_day_of_week").isin(1, 7), 1).otherwise(0)
        )
        
        # Monday reporting flag
        df = df.withColumn(
            "reported_monday",
            when(col("reported_day_of_week") == 2, 1).otherwise(0)
        )
        
        print("\nTime-based features created")
        return df
    
    def calculate_delays(self, df):
        """
        Calculate various delay features.
        """
        
        # Inception to claim delay
        df = df.withColumn(
            "inception_to_claim_days",
            datediff(col("start_date"), col("policy_start_date"))
        )
        
        # Claim to policy end delay
        df = df.withColumn(
            "claim_to_policy_end",
            datediff(col("policy_renewal_date"), col("start_date"))
        )
        
        print("\nDelay features created")
        return df

    def create_check_variables(self, df):
        """
        Create check variables (C1-C14) for fraud risk assessment.
        """
        # C1: Friday/Saturday night incidents (logic from experimental)
        # Friday: dayofweek = 6, Saturday: dayofweek = 7, Sunday: dayofweek = 1
        df = df.withColumn("incident_day_of_week_name", date_format(col("start_date"), "E"))
        
        fri_sat_night = (
            (col("incident_day_of_week_name").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | 
            (col("incident_day_of_week_name").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4)))
        )

        df = df.withColumn(
            "total_loss_new",
            when(col("assessment_category").isin("DriveableTotalLoss", "UnroadworthyTotalLoss"), 1).otherwise(0)
        )
        
        df = df.withColumn(
            "C1_fri_sat_night",
            when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0)
        )
        
        # C2: Reporting delays (3+ days)
        df = df.withColumn(
            "C2_reporting_delay",
            when(col("delay_in_reporting") >= 3, 1).otherwise(0)
        )
        
        # C3: Weekend incident reported on Monday
        df = df.withColumn(
            "C3_weekend_incident_reported_monday",
            when(
                (col("incident_weekend") == 1) & 
                (col("reported_monday") == 1),
                1
            ).otherwise(0)
        )
        
        # C4: Total loss flag
        df = df.withColumn(
            "C4_total_loss",
            when(col("total_loss_flag") == True, 1).otherwise(0)
        )
        
        # C5: Night incident (11pm - 5am)
        df = df.withColumn(
            "C5_is_night_incident",
           when((hour(col("start_date")) >= 23) | (hour(col("start_date")) <= 5) | (hour(col("start_date"))).isNull(), 1).otherwise(0)
        )
        
        # C6: No commuting policy but rush hour incident
        not_commuting_rush = (
            (lower(col("vehicle_use_quote")) == "1") &  # vehicle_use_quote == 1 means NOT commuting
            (hour(col("start_date")).between(6, 10)) | (hour(col("start_date")).between(15, 18))
        )
        df = df.withColumn(
            "C6_no_commuting_but_rush_hour",
            when(not_commuting_rush, 1).when(not_commuting_rush.isNull(), 1).otherwise(0)
        )
        
        # C7: Police attendance or crime reference
        df = df.withColumn(
            "C7_police_attended_or_crime_reference",
            when(
                (col("is_police_attendance") == True) | 
                (col("is_crime_reference_provided") == True),
                1
            ).otherwise(0)
        )
        
        # C8: Vehicle unattended
        df = df.withColumn(
            "C8_vehicle_unattended",
            when(col("vehicle_unattended") == 1, 1).otherwise(0)
        )
        
        # C9: Policy inception within 30 days
        df = df.withColumn(
            "C9_policy_within_30_days",
            when(col("inception_to_claim_days").between(0, 30), 1).when(col("inception_to_claim_days").isNull(), 1).otherwise(0)
        )
        
        # C10: Claim within 60 days of policy end
        df = df.withColumn(
            "C10_claim_to_policy_end",
            when(col("claim_to_policy_end") < 60, 1).when(col("claim_to_policy_end").isNull(), 1).otherwise(0)
        )
        
        # C11: Young or inexperienced driver  
        # Note: experimental notebooks use driver_age_low_1 and licence_low_1 which we need to calculate first
        df = df.withColumn(
            "driver_age_low_1", 
            when(col("age_at_policy_start_date_1") < 25, 1).when(col("age_at_policy_start_date_1").isNull(), 1).otherwise(0)
        )
        df = df.withColumn(
            "licence_low_1", 
            when(col("licence_length_years_1") <= 3, 1).otherwise(0)
        )
        
        condition_inexperienced = (col("driver_age_low_1") == 1) | (col("licence_low_1") == 1)
        df = df.withColumn(
            "C11_young_or_inexperienced",
            when(condition_inexperienced, 1).when(condition_inexperienced.isNull(), 1).otherwise(0)
        )
        
        # C12: Expensive vehicle for driver age
        condition_expensive_car = (
            ((col("age_at_policy_start_date_1") < 25) & (col("vehicle_value") >= 20000)) | 
            ((col("age_at_policy_start_date_1") >= 25) & (col("vehicle_value") >= 30000))
        )
        df = df.withColumn(
            "C12_expensive_for_driver_age",
            when(condition_expensive_car, 1).when(condition_expensive_car.isNull(), 1).otherwise(0)
        )
        
        # C14: Watchwords in circumstances (from experimental)
        watch_words = "|".join([
            "commut", "deliver", "parcel", "drink", "police", "custody", "arrest", 
            "alcohol", "drug", "station", "custody"
        ])
        
        df = df.withColumn(
            "C14_contains_watchwords",
            when(lower(col("circumstances")).rlike(watch_words), 1)
            .when(col("circumstances").isNull(), 1)
            .otherwise(0)
        )
        
        # Get list of all check columns
        check_cols = [
            'C1_fri_sat_night', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday',
            'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
            'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days',
            'C10_claim_to_policy_end', 'C11_young_or_inexperienced',
            'C12_expensive_for_driver_age', 'C14_contains_watchwords'
        ]
        
        # Check if any check is true (checks_max)
        df = df.withColumn('checks_max', greatest(*[col(c) for c in check_cols]))

        # Check list 
        df = df.withColumn(
                "checks_list",
                array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in check_cols])
            )

        df = df.withColumn(
                        "checks_list",
                        expr("filter(checks_list, x -> x is not null)")
                    ).withColumn("num_failed_checks", size(col("checks_list")))
        
        print("\nCheck variables (C1-C14) created")
        return df
    
    def aggregate_driver_features(self, df):
        """
        Aggregate driver features across multiple drivers.
        Note: This is already done in preprocessing, but we ensure completeness here.
        """
        # These aggregations should already exist from preprocessing
        driver_agg_cols = [
            'max_additional_vehicles_owned', 'min_additional_vehicles_owned',
            'max_age_at_policy_start_date', 'min_age_at_policy_start_date',
            'max_cars_in_household', 'min_cars_in_household',
            'max_licence_length_years', 'min_licence_length_years',
            'max_years_resident_in_uk', 'min_years_resident_in_uk'
        ]
        
        # Verify columns exist
        missing_cols = [col for col in driver_agg_cols if col not in df.columns]
        if missing_cols:
            print(f"\nMissing driver aggregation columns: {missing_cols}")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values with appropriate imputation strategies.
        """
        # Numeric columns - fill with 0 or median
        numeric_fill_zero = [
            'damage_score', 'areas_damaged',
            'inception_to_claim_days'
        ]
        
        for col_name in numeric_fill_zero:
            if col_name in df.columns:
                df = df.fillna({col_name: 0})
        
        # Boolean/flag columns - fill with 0
        boolean_cols = [c for c in df.columns if c.startswith('C') and '_' in c]
        for col_name in boolean_cols:
            df = df.fillna({col_name: 0})
        
        print("\nMissing values handled")
        return df
    
    def create_target_variable(self, df):
        """
        Create target variable for modeling.
        """
        # Target variable
        df = df.withColumn(
            "referred_to_tbg",
            when(col("tbg_risk").isin([0, 1]), 1).otherwise(0)
        )
        print("\nTarget variable created")
        return df
    
    def select_final_features(self, df):
        """
        Select final set of features for modeling.
        """
        # Define feature groups
        target_cols = ['svi_risk', 'tbg_risk', 'fa_risk', 'referred_to_tbg']
        
        id_cols = ['claim_number', 'dataset', 'start_date', 'checks_list', 'Outcome_of_Investigation', 'position_status']
        
        numeric_features = [
            'areas_damaged', 'damage_sev_max',
            'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy',
            'areasDamagedSevere', 'areasDamagedTotal', 'areasDamagedRelative',
            'min_claim_driver_age', 'vehicle_value',
            'inception_to_claim', 'claim_to_policy_end',
            'voluntary_amount', 'policyholder_ncd_years', 'annual_mileage',
            'veh_age', 'business_mileage', 'impact_speed', 'manufacture_yr_claim',
            'outstanding_finance_amount', 'excesses_applied', 'incidentDayOfWeekC', 'incidentHourC', 'reported_day_of_week',
            'reported_date'
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
            'incidentMonthC', 'assessment_category', 'engine_damage',
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
            'driver_age_low_1', 'claim_driver_age_low', 'licence_low_1', 'checks_max', 'total_loss_flag', 'total_loss_new'
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

        other_cols = ['vehicle_use_quote', 'circumstances']
        
        # Combine all features
        all_features = (id_cols + target_cols + numeric_features + 
                       check_features + categorical_features +
                       driver_features + flag_features + damage_cols + other_cols)
        
        # Select only columns that exist
        existing_features = [col for col in all_features if col in df.columns]
        
        print(f"\nSelected {len(existing_features)} features for modeling")
        return df.select(existing_features)
    
    def save_feature_data(self, df):
        """
        Save feature-engineered data to MLStore catalog.
        """

        # Get dynamic table path
        table_path = self.get_table_path("single_vehicle_incident_checks", "svi_features", "mlstore")
        
        print(f"\nSaving feature data to: {table_path}")

        # Check if file path exists, if not create it
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.env_config['mlstore_catalog']}.single_vehicle_incident_checks")
        
        # Write with partitioning
        df.write.mode("overwrite").partitionBy("dataset").option("overwriteSchema", "true").saveAsTable(table_path)
        
        print("\nFeature data saved successfully")
    
    def run_feature_engineering_pipeline(self, input_df, save_data=True):
        """
        Run the complete feature engineering pipeline.
        
        Args:
            input_df: Preprocessed data from DataPreprocessing module
            save_data (bool): Whether to save the feature data to MLStore
            
        Returns:
            Spark DataFrame with engineered features
        """
        print("Starting feature engineering pipeline")
        
        # Apply damage score calculations
        df = self.apply_damage_score_calculation(input_df)
        
        # Create time-based features
        df = self.create_time_based_features(df)
        
        # Calculate delays
        df = self.calculate_delays(df)
        
        # Create check variables
        df = self.create_check_variables(df)
        
        # Aggregate driver features (verify they exist)
        df = self.aggregate_driver_features(df)

        # Remove nan in column causing model training issue
        df = df.dropna(subset=['vehicle_overnight_location_id'])
        
        # Handle missing values
        df = self.handle_missing_values(df)

        # Create target variable
        df = self.create_target_variable(df)
        
        # Select final features
        df = self.select_final_features(df)
        
        # Save if requested
        if save_data:
            self.save_feature_data(df)
        
        print("\nFeature engineering pipeline completed successfully")
        return df