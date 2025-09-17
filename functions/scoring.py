from pyspark.sql.functions import *
from pyspark.sql import Window, DataFrame
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.sql.types import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from xgboost.spark import SparkXGBRegressor
import xgboost as xgb

from mlflow.models.signature import ModelSignature, Schema, infer_signature
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec

import pandas as pd
import numpy as np
import re, sys, os, yaml, time, mlflow

from typing import Tuple, List, Dict, Any

class SVIModelScoring:
    """
    Two-stage model training pipeline for SVI (Single Vehicle Incident) fraud detection.
    Trains Desk Check and Interview models with multi-environment support.
    """
    
    def __init__(self, spark, env_config):
        self.spark = spark
        self.env_config = env_config
        
        # Initialize MLflowClient here
        self.client = mlflow.tracking.MlflowClient()
        
        # Model names based on environment
        self.desk_check_model_name = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_desk_check_lgbm"
        self.interview_model_name = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_interview_lgbm"

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
        
    def load_training_data(self):
        """
        Load feature-engineered data for model training.
        """
        table_path = self.get_table_path("single_vehicle_incident_checks", "daily_svi_features", "mlstore")
        print(f"Loading training data from: {table_path}")
        
        df = self.spark.table(table_path).toPandas()

        print(f"Dataframe set size: {df.count()}")
        
        return df
    
    def get_all_feature_names_defined_in_prepare_methods(self):
        """Helper to get all feature names used across both prepare methods."""
        
        # Desk check features
        mean_fills_dc = [
            "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", 
            "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", 
            "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", 
            "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", 
            "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", 
            "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", 
            "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", 
            "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", "incident_day_of_week"
        ]
        damage_cols_dc = ["areasDamagedMinimal", "areasDamagedMedium", "areasDamagedHeavy", "areasDamagedSevere", "areasDamagedTotal"]
        bool_cols_dc = [
            "vehicle_unattended", "excesses_applied", "is_first_party", "first_party_confirmed_tp_notified_claim", 
            "is_air_ambulance_attendance", "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance", 
            "veh_age_more_than_10", "police_considering_actions", "is_crime_reference_provided", "ncd_protected_flag", 
            "boot_opens", "doors_open", "multiple_parties_involved", "is_incident_weekend", "is_reported_monday", 
            "driver_age_low_1", "claim_driver_age_low", "licence_low_1"
        ]
        one_fills_dc = [
            "C1_fri_sat_night", "C2_reporting_delay", "C3_weekend_incident_reported_monday", "C5_is_night_incident", 
            "C6_no_commuting_but_rush_hour", "C7_police_attended_or_crime_reference", "C9_policy_within_30_days", 
            "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", 
            "C14_contains_watchwords", "checks_max"
        ]
        string_cols_dc = [
            'car_group', 'vehicle_overnight_location_id', 'incidentMonthC', 'employment_type_abi_code_5', 
            'employment_type_abi_code_4', 'employment_type_abi_code_3', 'employment_type_abi_code_2', 'policy_type', 
            'postcode', 'assessment_category', 'engine_damage', 'sales_channel', 'overnight_location_abi_code', 
            'vehicle_overnight_location_name', 'policy_cover_type', 'notification_method', 'impact_speed_unit', 
            'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 'front_severity', 
            'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 'left_severity', 
            'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
            'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 
            'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 'right_front_wheel_severity', 
            'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 
            'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 
            'employment_type_abi_code_1', 'incident_day_of_week', 'reported_day_of_week'
        ]
        other_cols_dc = ['claim_number','svi_risk', 'dataset', 'fa_risk', 'tbg_risk', 'referred_to_tbg', 'reported_date']
        high_cardinality_dc = ['car_group', 'employment_type_abi_code_1', 'employment_type_abi_code_2', 
                               'employment_type_abi_code_3', 'employment_type_abi_code_4', 'employment_type_abi_code_5', 
                               'postcode', 'damageScore']
                               
        string_cols_dc_filtered = list(set(string_cols_dc) - set(high_cardinality_dc))
        all_features_dc = mean_fills_dc + damage_cols_dc + bool_cols_dc + one_fills_dc + string_cols_dc_filtered + other_cols_dc

        # Interview features
        final_features_int = [
            'voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 'annual_mileage', 
            'min_claim_driver_age', 'incidentHourC', 'assessment_category', 'left_severity', 
            'C9_policy_within_30_days', 'areasDamagedHeavy', 'impact_speed', 'incident_sub_cause', 
            'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 'rear_right_severity', 
            'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 'num_failed_checks', 
            'front_left_severity', 'areasDamagedMedium', 'rear_window_damage_severity', 
            'incident_cause', 'incident_day_of_week', 'age_at_policy_start_date_1', 'checks_max', 
            'total_loss_flag'
        ]
        # Mapping needs to be applied
        column_mappings_int = {
            'incidentHourC': 'incident_hour', 'incident_day_of_week': 'incident_day_of_week',
            'veh_age': 'vehicle_age', 'claim_to_policy_end': 'claim_to_policy_end_days'
        }
        # Apply mappings within this method scope for consistency with the actual prepare methods
        mapped_features_int = [column_mappings_int.get(f, f) for f in final_features_int]
        all_features_int = mapped_features_int + ['desk_check_pred']

        # Combine all unique feature names from both sets
        all_unique_features = set(all_features_dc).union(set(all_features_int))
        return list(all_unique_features)

    def prepare_desk_check_features(self, df):
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
        
        damage_cols = ["areasDamagedMinimal", "areasDamagedMedium", "areasDamagedHeavy", 
                       "areasDamagedSevere", "areasDamagedTotal"]
        bool_cols = [
            "vehicle_unattended", "excesses_applied", "is_first_party", 
            "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
            "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance", 
            "veh_age_more_than_10", "police_considering_actions", "is_crime_reference_provided", 
            "ncd_protected_flag", "boot_opens", "doors_open", "multiple_parties_involved", 
            "is_incident_weekend", "is_reported_monday", "driver_age_low_1", "claim_driver_age_low", 
            "licence_low_1"
        ]
        
        one_fills = [
            "C1_fri_sat_night", "C2_reporting_delay", "C3_weekend_incident_reported_monday", 
            "C5_is_night_incident", "C6_no_commuting_but_rush_hour", 
            "C7_police_attended_or_crime_reference", "C9_policy_within_30_days", 
            "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
            "C12_expensive_for_driver_age", "C14_contains_watchwords", "checks_max"
        ]
        
        string_cols = [
            'car_group', 'vehicle_overnight_location_id', 'incidentMonthC', 
            'employment_type_abi_code_5', 'employment_type_abi_code_4', 'employment_type_abi_code_3', 
            'employment_type_abi_code_2', 'policy_type', 'postcode', 'assessment_category', 
            'engine_damage', 'sales_channel', 'overnight_location_abi_code', 
            'vehicle_overnight_location_name', 'policy_cover_type', 'notification_method', 
            'impact_speed_unit', 'impact_speed_range', 'incident_type', 'incident_cause', 
            'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 'front_left_severity', 
            'front_right_severity', 'left_severity', 'left_back_seat_severity', 
            'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
            'left_underside_severity', 'rear_severity', 'rear_left_severity', 
            'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 
            'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 
            'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 
            'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 
            'employment_type_abi_code_1', 'incident_day_of_week', 'reported_day_of_week'
        ]
        
        high_cardinality = ['car_group', 'employment_type_abi_code_1', 
                            'employment_type_abi_code_2', 'employment_type_abi_code_3', 
                            'employment_type_abi_code_4', 'employment_type_abi_code_5', 
                            'postcode', 'damageScore']
        
        string_cols = list(set(string_cols) - set(high_cardinality))
        
        all_features = mean_fills + damage_cols + bool_cols + one_fills + string_cols
        
        features = [f for f in all_features if f in df.columns]
        
        numeric_features = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 'incident_day_of_week', 'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal']

        categorical_features = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 'reported_day_of_week', 'checks_max']
        
        boolean_features = [f for f in features if f in bool_cols + one_fills]
        
        return features
    
    def prepare_interview_features(self, df):
        """
        Prepare features for interview model (from experimental notebooks).
        """
        final_features = [
            'voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 
            'annual_mileage', 'min_claim_driver_age', 'incidentHourC', 'assessment_category', 
            'left_severity', 'C9_policy_within_30_days', 'areasDamagedHeavy', 'impact_speed', 
            'incident_sub_cause', 'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 
            'rear_right_severity', 'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 
            'num_failed_checks', 'front_left_severity', 'areasDamagedMedium', 
            'rear_window_damage_severity', 'incident_cause', 'incident_day_of_week', 
            'age_at_policy_start_date_1', 'checks_max', 'total_loss_flag', 'years_resident_in_uk_1'
        ]
          
        final_features = final_features + ['desk_check_pred']

        num_interview = ['voluntary_amount', 'policyholder_ncd_years',
            'max_years_resident_in_uk', 'annual_mileage', 'min_claim_driver_age',
            'incidentHourC', 'areasDamagedHeavy', 'impact_speed',
            'years_resident_in_uk_1', 'vehicle_value', 'areasDamagedTotal',
            'manufacture_yr_claim', 'claim_to_policy_end', 'veh_age',
            'licence_length_years_1', 'num_failed_checks', 'areasDamagedMedium',
            'min_years_resident_in_uk', 'incident_day_of_week',
            'age_at_policy_start_date_1', 'max_age_at_policy_start_date'
        ]

        cat_interview = ['assessment_category', 'left_severity', 'C9_policy_within_30_days',
            'incident_sub_cause', 'rear_right_severity', 'front_left_severity',
            'rear_window_damage_severity', 'incident_cause', 'checks_max',
            'total_loss_flag']
        
        boolean_features = ['C9_policy_within_30_days', 'checks_max', 'total_loss_flag']
        
        numeric_features = [f for f in num_interview if f in final_features]
        categorical_features = [f for f in cat_interview if f in final_features]
        boolean_features = [f for f in boolean_features if f in final_features]
        
        return final_features
    

    def desk_check_scoring(self, df, features):
        model_name = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_desk_check_lgbm"
        client = mlflow.tracking.MlflowClient()

        # Find champion version
        champion_details = client.get_model_version_by_alias(name=model_name, alias="champion")
        champion_version = champion_details.version

        # Load champion model using the alias
        model_uri = f"models:/{model_name}@champion"
        champion_model = mlflow.sklearn.load_model(model_uri)

        # Predict and assign
        y_pred = champion_model.predict_proba(df[features])[:, 1].round(4)
        df['fa_pred'] = y_pred
        df['desk_check_pred'] = (df['fa_pred'] >= 0.5).astype(int)
        df['desk_check_model_version'] = champion_version

        return df

    def interview_scoring(self, df, features):
        model_name = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_interview_lgbm"
        client = mlflow.tracking.MlflowClient()

        # Find champion version
        champion_details = client.get_model_version_by_alias(name=model_name, alias="champion")
        champion_version = champion_details.version

        # Load champion model using the alias
        model_uri = f"models:/{model_name}@champion"
        champion_model = mlflow.sklearn.load_model(model_uri)

        # Predict and assign
        y_prob2 = champion_model.predict_proba(df[features])[:, 1].round(4)
        df['y_prob2'] = y_prob2
        df['y_pred2'] = (df['y_prob2'] >= 0.5).astype(int)

        # Combined logic
        df['y_cmb'] = (
            (df['desk_check_pred'] == 1) &
            (df['y_pred2'] == 1) &
            (df['num_failed_checks'] >= 1)
        ).astype(int)
        df['flagged_by_model'] = np.where(df['y_cmb'] == 1, 1, 0)

        # Late night no commuting
        late_night_no_commuting = (
            (df['vehicle_use_quote'].fillna(0).astype(int) == 1) &
            (df['incidentHourC'].between(1, 4))
        )
        df['late_night_no_commuting'] = np.where(late_night_no_commuting, 1, 0)
        df['y_cmb'] = np.where(
            df['late_night_no_commuting'] == 1, 1, df['y_cmb']
        )

        # Unconscious flag
        watch_words = "|".join([
            "pass out", "passed out", "passing out", "blackout", "black out",
            "blacked out", "blacking out", "unconscious", "unconsciousness",
            "sleep", "asleep", "sleeping", "slept", "dozed", "doze", "dozing"
        ])
        df['unconscious_flag'] = np.where(
            df['circumstances'].str.lower().str.contains(watch_words, na=False), 1, 0
        )
        df['y_cmb'] = np.where(
            df['unconscious_flag'] == 1, 1, df['y_cmb']
        )

        df['y_cmb_label'] = np.where(
            df['y_cmb'] == 0, 'Low', 'High'
        )
        df['y_rank_prob'] = np.sqrt(
            df['fa_pred'].fillna(100) * df['y_prob2'].fillna(100)
        ).round(3)
        df['interview_model_version'] = champion_version

        return df


    def final_table_write(self, df):
        # Create risk reason
        risk_cols = ['flagged_by_model', 'unconscious_flag', 'late_night_no_commuting']

        df_spark = self.spark.createDataFrame(df)

        df_spark = df_spark.withColumn(
            "risk_reason",
            array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in risk_cols])
        )
        df_spark = df_spark.withColumn(
            "risk_reason",
            expr("filter(risk_reason, x -> x is not null)")
        )

        # Get table name
        self.get_table_path("single_vehicle_incident_checks", "daily_svi_predictions", "mlstore")

        # Write table to delta
        df_spark.write \
            .mode("overwrite") \
            .format("delta") \
            .option("overwriteSchema", "true") \
            .saveAsTable(f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_svi_predictions")

        print(f"Table saved to {self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.daily_svi_predictions")
            
    def run_scoring_pipeline(self):
        print("Starting SVI model scoring pipeline")
        
        df_scoring = self.load_training_data()
        
        df_scoring['referred_to_tbg'] = df_scoring['referred_to_tbg'].fillna(0).astype(int)
        
        print("\n" + "="*60)
        print("STAGE 1: DESK CHECK MODEL SCORING")
        print("="*60)
        
        desk_check_features = self.prepare_desk_check_features(df_scoring)
        df_scoring = self.desk_check_scoring(
            df_scoring,
            desk_check_features
        )
        
        print("\n" + "="*60)
        print("STAGE 2: INTERVIEW MODEL SCORING")
        print("="*60)
        
        interview_features = self.prepare_interview_features(df_scoring)
        df_scoring = self.interview_scoring(
            df_scoring,
            interview_features
        )

        self.final_table_write(df_scoring)
        
        return df_scoring
