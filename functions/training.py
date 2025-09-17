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
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from xgboost.spark import SparkXGBRegressor
import xgboost as xgb
from lightgbm import LGBMClassifier

from mlflow.models.signature import ModelSignature, Schema, infer_signature
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec

import pandas as pd
import numpy as np
import re, sys, os, yaml, time, mlflow

from typing import Tuple, List, Dict, Any

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve, auc
)

import seaborn as sns
from datetime import datetime

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
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from xgboost.spark import SparkXGBRegressor
import xgboost as xgb
from lightgbm import LGBMClassifier

from mlflow.models.signature import ModelSignature, Schema, infer_signature
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec

import pandas as pd
import numpy as np
import re, sys, os, yaml, time, mlflow

from typing import Tuple, List, Dict, Any

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve, auc
)

import seaborn as sns
from datetime import datetime


class SVIModelTraining:
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
        table_path = self.get_table_path("single_vehicle_incident_checks", "svi_features", "mlstore")
        print(f"Loading training data from: {table_path}")
        
        df = self.spark.table(table_path)
        train_df = df.filter(col('dataset') == 'train').toPandas()
        test_df = df.filter(col('dataset') == 'test').toPandas()

        print(f"Train set size: {train_df.count()}, Test set size: {test_df.count()}")
        
        return train_df, test_df
    
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
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', 'passthrough')  # No scaling as per experimental
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'  # Pass through boolean features
        )
        
        return features, preprocessor
    
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
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', 'passthrough')
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return final_features, preprocessor
    
    def train_desk_check_model(self, X_train, y_train, X_test, y_test, features, preprocessor):
        with mlflow.start_run(run_name=f"desk_check_model"):
            print("Training desk check model...")
            model = LGBMClassifier(verbose=-1, scale_pos_weight=12)
            pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
            
            # Using fixed param_grid to avoid GridSearchCV's long run during testing
            param_grid = {
                'classifier__n_estimators': [30], 
                'classifier__max_depth': [4], 
                'classifier__learning_rate': [0.1], 
                'classifier__num_leaves': [5], 
                'classifier__min_child_weight': [0.1]
            }

            #param_grid = {
            #    'classifier__n_estimators': [30, 100, 200],
            #    'classifier__max_depth': [3, 5, 7, 10],
            #    'classifier__learning_rate': [0.01, 0.1, 0.2],
            #    'classifier__num_leaves': [20, 31, 40, 50],
            #    'classifier__min_child_weight': [0.001, 0.1, 1],
            #}

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='recall', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train[features], y_train)
            y_pred = grid_search.predict(X_test[features])
            y_proba = grid_search.predict_proba(X_test[features])[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            self.plot_confusion_matrix(y_test, y_pred, "Desk Check Model")
            self.log_feature_importance(grid_search.best_estimator_, features, "desk_check")
            self.simple_classification_report(y_test, y_pred)
            signature = infer_signature(X_test[features], y_proba)
            mlflow.sklearn.log_model(
                grid_search.best_estimator_, "desk_check_model",
                registered_model_name=self.desk_check_model_name, signature=signature
            )
            version = self.get_latest_model_version(self.desk_check_model_name)
            if self.client:
                self.client.set_registered_model_alias(self.desk_check_model_name, 'champion', version)
            print(f"Desk check model training completed. F1 Score: {f1:.3f}")
            return grid_search.best_estimator_, y_pred, y_test
    
    def train_interview_model(self, X_train, y_train, X_test, y_test, 
                                desk_check_pred_train, desk_check_pred_test,
                                features, preprocessor):
        with mlflow.start_run(run_name=f"interview_model_"):
            print("Training interview model...")
            X_train_with_pred = X_train.copy()
            X_train_with_pred['desk_check_pred'] = desk_check_pred_train
            X_test_with_pred = X_test.copy()
            X_test_with_pred['desk_check_pred'] = desk_check_pred_test
            model = LGBMClassifier(verbose=0)
            undersampler = RandomUnderSampler(random_state=12)
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor), ('undersampler', undersampler), ('classifier', model)
            ])
            
            # Using fixed param_grid to avoid GridSearchCV's long run during testing
            param_grid = {
                'classifier__n_estimators': [20], 
                'classifier__max_depth': [3], 
                'classifier__learning_rate': [0.1], 
                'classifier__num_leaves': [10], 
                'classifier__min_child_weight': [0.1]
            }

            #param_grid = {
            #    'classifier__n_estimators': [20, 100, 200],
            #    'classifier__max_depth': [3, 5, 7, 10],
            #    'classifier__learning_rate': [0.01, 0.1, 0.2],
            #    'classifier__num_leaves': [20, 31, 40, 50],
            #    'classifier__min_child_weight': [0.001, 0.1, 1],
            #}

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_with_pred[features], y_train)
            y_pred = grid_search.predict(X_test_with_pred[features])
            y_proba = grid_search.predict_proba(X_test_with_pred[features])[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            self.plot_confusion_matrix(y_test, y_pred, "Interview Model")
            self.log_feature_importance(grid_search.best_estimator_, features, "interview")
            self.simple_classification_report(y_test, y_pred)
            signature = infer_signature(X_test_with_pred[features], y_proba)
            mlflow.sklearn.log_model(
                grid_search.best_estimator_, "interview_model",
                registered_model_name=self.interview_model_name, signature=signature
            )
            version = self.get_latest_model_version(self.interview_model_name)
            if self.client:
                self.client.set_registered_model_alias(self.interview_model_name, 'champion', version)
            print(f"Interview model training completed. F1 Score: {f1:.3f}")
            return grid_search.best_estimator_, y_pred, y_test

    def simple_classification_report(self, y_prob, y_true, threshold=0.5):
        y_pred = (y_prob > threshold).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        report = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': pr_auc
        }
        return report

    def plot_confusion_matrix(self, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        # Ensure 'Artefacts' directory exists if running locally
        import os
        os.makedirs('Artefacts', exist_ok=True)
        plt.savefig(f'Artefacts/confusion_matrix_{title.lower().replace(" ", "_")}.png')
        mlflow.log_artifact(f'Artefacts/confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def log_feature_importance(self, pipeline, features, model_type):
        """
        Log feature importance plot.
        """
        classifier = pipeline.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            preprocessor_fitted = pipeline.named_steps['preprocessor']
            
            try:
                # Pass the original input feature names to ColumnTransformer's method
                all_feature_names = list(preprocessor_fitted.get_feature_names_out(input_features=features))
            except Exception as e:
                print(f"Warning: ColumnTransformer.get_feature_names_out() failed ({e}). Falling back to manual extraction (less robust).")
                
                all_feature_names = []
                # Iterate through the *fitted* transformers to get their output names
                for name, fitted_transformer, original_cols_input_to_this_transformer in preprocessor_fitted.transformers_:
                    if name == 'num':
                        # Numeric features retain their names.
                        all_feature_names.extend(original_cols_input_to_this_transformer)
                    elif name == 'cat':
                        # Categorical features are one-hot encoded.
                        if hasattr(fitted_transformer, 'named_steps') and 'onehot' in fitted_transformer.named_steps and \
                           hasattr(fitted_transformer.named_steps['onehot'], 'get_feature_names_out'):
                            # The `get_feature_names_out` of OneHotEncoder takes `input_features`
                            # as the original names it was fitted on.
                            cat_transformed_names = fitted_transformer.named_steps['onehot'].get_feature_names_out(original_cols_input_to_this_transformer)
                            all_feature_names.extend(cat_transformed_names)
                
                all_transformed_from_num_cat = [
                    col_name for name, _, cols in preprocessor_fitted.transformers_ if name in ['num', 'cat'] for col_name in cols
                ]
                passthrough_features = [f for f in features if f not in all_transformed_from_num_cat]
                all_feature_names.extend(passthrough_features)


            # Ensure the length of feature names matches importance array
            if len(all_feature_names) != len(classifier.feature_importances_):
                print(f"Warning: Mismatch between number of feature names ({len(all_feature_names)}) "
                      f"and feature importances ({len(classifier.feature_importances_)}). "
                      "Adjusting feature names list length.")
                all_feature_names = all_feature_names[:len(classifier.feature_importances_)]
            
            importance_df = pd.DataFrame({
                'feature': all_feature_names,
                'importance': classifier.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
            plt.title(f'Top 20 Feature Importances - {model_type}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Ensure 'Artefacts' directory exists if running locally
            import os
            os.makedirs('Artefacts', exist_ok=True)
            plt.savefig(f'Artefacts/feature_importance_{model_type}.png')
            mlflow.log_artifact(f'Artefacts/feature_importance_{model_type}.png')
            plt.close()
        else:
            print(f"Classifier of type {type(classifier).__name__} does not have 'feature_importances_'. Skipping feature importance logging for {model_type}.")

    def get_latest_model_version(self, model_name):
        try:
            latest_version = 1
            if self.client:
                for mv in self.client.search_model_versions(f"name='{model_name}'"):
                    version_int = int(mv.version)
                    if version_int > latest_version:
                        latest_version = version_int
            return latest_version
        except Exception as e:
            print(f"Error getting latest model version for {model_name}: {e}.")
            return 1
            
    def run_training_pipeline(self):
        print("Starting SVI model training pipeline")
        
        train_df, test_df = self.load_training_data()
        
        y_train_desk_check = train_df['referred_to_tbg'].fillna(0).astype(int)
        y_test_desk_check = test_df['referred_to_tbg'].fillna(0).astype(int)
        
        y_train_interview = train_df['svi_risk'].fillna(0).astype(int)
        y_test_interview = test_df['svi_risk'].fillna(0).astype(int)
        
        print("\n" + "="*60)
        print("STAGE 1: DESK CHECK MODEL TRAINING")
        print("="*60)
        
        desk_check_features, desk_check_preprocessor = self.prepare_desk_check_features(train_df)
        desk_check_model, desk_check_pred_test, desk_check_actual_test = self.train_desk_check_model(
            train_df, y_train_desk_check,
            test_df, y_test_desk_check,
            desk_check_features, desk_check_preprocessor
        )
        
        desk_check_pred_train = desk_check_model.predict_proba(train_df[desk_check_features])[:, 1]
        
        print("\n" + "="*60)
        print("STAGE 2: INTERVIEW MODEL TRAINING")
        print("="*60)
        
        interview_features, interview_preprocessor = self.prepare_interview_features(train_df)
        interview_model, interview_model_pred_test, interview_model_actual_test = self.train_interview_model(
            train_df, y_train_interview,
            test_df, y_test_interview,
            desk_check_pred_train, desk_check_pred_test,
            interview_features, interview_preprocessor
        )
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED")
        print("="*60)
        
        print("\n" + "="*60)
        print("MODEL METRICS")
        print("="*60)
        print(f"Desk Check Model Accuracy: {self.simple_classification_report(desk_check_pred_test, desk_check_actual_test)}")
        print(f"Interview Model Accuracy: {self.simple_classification_report(interview_model_pred_test, interview_model_actual_test)}")

        print(f"Desk Check Model: {self.desk_check_model_name}")
        print(f"Interview Model: {self.interview_model_name}")
        
        return desk_check_model, interview_model
