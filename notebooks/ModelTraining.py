# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training Module
# MAGIC
# MAGIC Production-ready model training pipeline for Single Vehicle Incident (SVI) fraud detection.
# MAGIC This module handles:
# MAGIC - Two-stage model training (Desk Check + Interview models)
# MAGIC - Multi-environment support (dsexp, modelbuild, modelpromotion, modeldeployment)
# MAGIC - Hyperparameter tuning using LightGBM
# MAGIC - Model registration to MLflow with environment-aware naming
# MAGIC - Comprehensive model evaluation and metrics tracking

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports and Configuration Setup

# COMMAND ----------

# Import required libraries
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from pyspark.sql.functions import col, when, greatest, lit, mean
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, make_scorer,
    average_precision_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, List, Any
import warnings
warnings.filterwarnings('ignore')

# MLflow setup
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Load configuration
extract_column_transformation_lists("/config_files/training.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# Get current environment
current_env = get_current_environment()
env_config = get_environment_config()

print(f"Running model training in environment: {current_env}")

# Define model names based on environment
desk_check_model_name = f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_desk_check_lgbm"
interview_model_name = f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_interview_lgbm"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Helper Functions

# COMMAND ----------

# Define fill columns
mean_fills = [ "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", "incidentDayOfWeekC"]

damage_cols = ["areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal"]
bool_cols = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance","is_police_attendance","veh_age_more_than_10","police_considering_actions","is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved",  "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low","licence_low_1"]

one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords"]

string_cols = [
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
]

def do_fills_pd(raw_df, mean_dict):
    """Apply missing value imputation to pandas DataFrame"""
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

def simple_classification_report(y_prob, y_true, threshold=0.5):
    """Generate classification metrics report"""
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

def generate_classification_report(y_prob, y_true, threshold=0.5):   
    """Generate detailed classification report"""
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

def pr_auc(y_true, y_scores):
    """Calculate PR AUC score"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Prepare Data

# COMMAND ----------

def load_training_data(spark, env_config):
    """Load training data from the feature table"""
    table_path = f"{env_config['mlstore_catalog']}.single_vehicle_incident_checks.features_engineered"
    
    # Try auxiliary catalog if mlstore fails
    try:
        raw_df = spark.table(table_path)
    except:
        table_path = f"{env_config['auxiliary_catalog']}.single_vehicle_incident_checks.claims_pol_svi"
        raw_df = spark.table(table_path)
    
    # Add necessary columns if missing
    raw_df = raw_df.withColumn('referred_to_tbg', when(col('tbg_risk').isin([0, 1]), 1).otherwise(0))
    raw_df = raw_df.withColumn('underbody_damage_severity', lit(None))
    
    # Aggregate check columns
    check_cols = [
        'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
        'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
        'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
        'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
    ]
    
    raw_df = raw_df.withColumn('checks_max', greatest(*[col(c) for c in check_cols]))
    
    # Fix boolean column types
    raw_df = raw_df.withColumn('police_considering_actions', col('police_considering_actions').cast('boolean'))
    raw_df = raw_df.withColumn('is_crime_reference_provided', col('is_crime_reference_provided').cast('boolean'))
    raw_df = raw_df.withColumn('multiple_parties_involved', col('multiple_parties_involved').cast('boolean'))
    
    # Convert to pandas
    high_cardinality = ['car_group', 'employment_type_abi_code_1', 'employment_type_abi_code_2', 
                        'employment_type_abi_code_3', 'employment_type_abi_code_4', 
                        'employment_type_abi_code_5', 'postcode', 'damageScore']
    
    raw_df_pd = raw_df.drop(*high_cardinality).toPandas()
    raw_df_pd['svi_risk'] = raw_df_pd['svi_risk'].replace(-1, 0)
    
    # Split train/test
    train_df = raw_df_pd[raw_df_pd.dataset == 'train'].drop('dataset', axis=1)
    test_df = raw_df_pd[raw_df_pd.dataset == 'test'].drop('dataset', axis=1)
    
    return train_df, test_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Model Training

# COMMAND ----------

print("Starting SVI model training pipeline")
print(f"Environment: {current_env}")
print(f"Catalog: {env_config['mlstore_catalog']}")

# Load training data
train_df, test_df = load_training_data(spark, env_config)

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Prepare target variables
train_df['tbg_risk'] = train_df['tbg_risk'].fillna(0).astype(int)
test_df['tbg_risk'] = test_df['tbg_risk'].fillna(0).astype(int)
train_df['svi_risk'] = train_df['svi_risk'].fillna(0).astype(int)
test_df['svi_risk'] = test_df['svi_risk'].fillna(0).astype(int)

y_train_desk_check = train_df['referred_to_tbg'].fillna(0).astype(int)
y_test_desk_check = test_df['referred_to_tbg'].fillna(0).astype(int)

y_train_interview = train_df['svi_risk'].fillna(0).astype(int)
y_test_interview = test_df['svi_risk'].fillna(0).astype(int)

print(f"\nTarget distribution:")
print(f"Desk Check - Train: {y_train_desk_check.value_counts().to_dict()}")
print(f"Desk Check - Test: {y_test_desk_check.value_counts().to_dict()}")
print(f"Interview - Train: {y_train_interview.value_counts().to_dict()}")
print(f"Interview - Test: {y_test_interview.value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Desk Check Model Training

# COMMAND ----------

print("="*60)
print("STAGE 1: DESK CHECK (FA) MODEL TRAINING")
print("="*60)

# Prepare features for desk check model
X_train = train_df.drop(['referred_to_tbg','claim_number', 'fa_risk', 'tbg_risk', 'reported_date', 'svi_risk'], axis=1, errors='ignore')
X_test = test_df.drop(['referred_to_tbg','claim_number', 'fa_risk', 'tbg_risk', 'reported_date', 'svi_risk'], axis=1, errors='ignore')

# Calculate mean dictionary for imputation
REFRESH_MEANS = True
if REFRESH_MEANS:
    mean_dict = X_train[mean_fills].astype('float').mean().round(4).to_dict()
    print("Calculated mean values for imputation")
else: 
    mean_dict = {'policyholder_ncd_years': 6.5934, 'inception_to_claim': 141.7436, 'min_claim_driver_age': 37.0723, 'veh_age': 11.4678, 'business_mileage': 276.1819, 'annual_mileage': 7355.2092, 'incidentHourC': 12.741, 'additional_vehicles_owned_1': 0.003, 'age_at_policy_start_date_1': 39.1134, 'cars_in_household_1': 1.8386, 'licence_length_years_1': 15.0401, 'years_resident_in_uk_1': 34.0761, 'max_additional_vehicles_owned': 0.0036, 'min_additional_vehicles_owned': 0.001, 'max_age_at_policy_start_date': 42.8814, 'min_age_at_policy_start_date': 34.9985, 'max_cars_in_household': 1.8968, 'min_cars_in_household': 1.7635, 'max_licence_length_years': 17.9961, 'min_licence_length_years': 11.8057, 'max_years_resident_in_uk': 37.9595, 'min_years_resident_in_uk': 29.8881, 'impact_speed': 28.0782, 'voluntary_amount': 236.8366, 'vehicle_value': 7616.7295, 'manufacture_yr_claim': 2011.7687, 'outstanding_finance_amount': 0.0, 'claim_to_policy_end': 83.6548, 'incidentDayOfWeekC': 4.0115}

# Apply fills
X_train = do_fills_pd(X_train, mean_dict)
X_test = do_fills_pd(X_test, mean_dict)

print(f"X_train shape after fills: {X_train.shape}")
print(f"X_test shape after fills: {X_test.shape}")

# COMMAND ----------

mlflow.sklearn.autolog(max_tuning_runs=None)

with mlflow.start_run(run_name="desk_check_model"):
    mlflow.set_tag("model_name", "svi_desk_check_model")
    mlflow.set_tag("stage", "desk_check")
    mlflow.set_tag("environment", current_env)
    
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['number']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Preprocessing for numeric data (no scaling as per old model)
    numeric_transformer = Pipeline(steps=[('scaler', 'passthrough')])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Define the model
    lgbm_model = LGBMClassifier(verbose=-1, scale_pos_weight=12, random_state=42)
    
    # Create pipeline
    desk_check_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgbm_model)
    ])
    
    # Define parameter grid for GridSearch
    param_grid = {
        'classifier__n_estimators': [10, 20, 30, 50],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.1],
        'classifier__num_leaves': [5, 10, 15, 31],
        'classifier__min_child_weight': [0.1, 0.5]
    }
    
    # Set up GridSearchCV
    pr_auc_scorer = make_scorer(pr_auc, needs_proba=True)
    grid_search = GridSearchCV(
        estimator=desk_check_pipeline, 
        param_grid=param_grid, 
        cv=3, 
        scoring='recall', 
        verbose=0,
        n_jobs=-1
    )
    
    # Fit model
    grid_search.fit(X_train, y_train_desk_check)
    
    # Log the best model
    desk_check_model = grid_search.best_estimator_
    
    signature = infer_signature(X_train, desk_check_model.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=desk_check_model, 
        artifact_path="model", 
        signature=signature,
        pyfunc_predict_fn="predict_proba",
        registered_model_name=desk_check_model_name
    )
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Predictions
    desk_check_pred_train = desk_check_model.predict_proba(X_train)[:, 1]
    desk_check_pred_test = desk_check_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_metrics = simple_classification_report(desk_check_pred_train, y_train_desk_check)
    test_metrics = simple_classification_report(desk_check_pred_test, y_test_desk_check)
    
    # Log metrics
    for metric_name, metric_value in train_metrics.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)
    
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    # Display metrics
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=['train', 'test']).round(3)
    print("\nDesk Check Model Metrics:")
    print(metrics_df)
    
    desk_check_run_id = mlflow.active_run().info.run_id
    print(f"\nDesk Check Model run ID: {desk_check_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Interview Model Training

# COMMAND ----------

print("="*60)
print("STAGE 2: INTERVIEW MODEL TRAINING")
print("="*60)

# Define interview model features
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

# Add desk check predictions as features
train_df['desk_check_pred'] = desk_check_pred_train
test_df['desk_check_pred'] = desk_check_pred_test

# Add num_failed_checks if not present
if 'num_failed_checks' not in train_df.columns:
    check_cols = [c for c in train_df.columns if c.startswith('C') and '_' in c]
    train_df['num_failed_checks'] = train_df[check_cols].sum(axis=1)
    test_df['num_failed_checks'] = test_df[check_cols].sum(axis=1)

# Add total_loss_flag if not present
if 'total_loss_flag' not in train_df.columns:
    train_df['total_loss_flag'] = 0
    test_df['total_loss_flag'] = 0

# Prepare interview features
interview_features = num_interview + cat_interview + ['desk_check_pred']

# Filter to available features
available_features = [f for f in interview_features if f in train_df.columns]
print(f"Using {len(available_features)} features for interview model")

X_train_interview = train_df[available_features]
X_test_interview = test_df[available_features]

# COMMAND ----------

with mlflow.start_run(run_name="interview_model"):
    mlflow.set_tag("model_name", "svi_interview_model")
    mlflow.set_tag("stage", "interview")
    mlflow.set_tag("environment", current_env)
    
    # Identify numeric and categorical columns for interview model
    numeric_features_int = X_train_interview.select_dtypes(include=['number']).columns
    categorical_features_int = X_train_interview.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numeric features: {len(numeric_features_int)}")
    print(f"Categorical features: {len(categorical_features_int)}")
    
    # Preprocessing
    numeric_transformer_int = Pipeline(steps=[('scaler', 'passthrough')])
    categorical_transformer_int = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    
    preprocessor_int = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_int, numeric_features_int),
            ('cat', categorical_transformer_int, categorical_features_int)
        ],
        remainder='passthrough'
    )
    
    # Define the model
    lgbm_model_int = LGBMClassifier(
        verbose=-1, 
        scale_pos_weight=15,  # Adjusted for interview model
        random_state=42
    )
    
    # Create pipeline
    interview_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_int),
        ('classifier', lgbm_model_int)
    ])
    
    # Parameter grid for interview model
    param_grid_int = {
        'classifier__n_estimators': [20, 30, 50, 100],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [10, 15, 31],
        'classifier__min_child_weight': [0.1, 0.5, 1.0]
    }
    
    # GridSearchCV
    grid_search_int = GridSearchCV(
        estimator=interview_pipeline,
        param_grid=param_grid_int,
        cv=3,
        scoring='recall',
        verbose=0,
        n_jobs=-1
    )
    
    # Fit model
    grid_search_int.fit(X_train_interview, y_train_interview)
    
    # Log the best model
    interview_model = grid_search_int.best_estimator_
    
    signature_int = infer_signature(X_train_interview, interview_model.predict(X_train_interview))
    model_info_int = mlflow.sklearn.log_model(
        sk_model=interview_model,
        artifact_path="model",
        signature=signature_int,
        pyfunc_predict_fn="predict_proba",
        registered_model_name=interview_model_name
    )
    
    mlflow.log_params(grid_search_int.best_params_)
    mlflow.log_metric("best_cv_score", grid_search_int.best_score_)
    
    print(f"Best parameters: {grid_search_int.best_params_}")
    print(f"Best CV score: {grid_search_int.best_score_:.4f}")
    
    # Predictions
    interview_pred_train = interview_model.predict_proba(X_train_interview)[:, 1]
    interview_pred_test = interview_model.predict_proba(X_test_interview)[:, 1]
    
    # Calculate metrics
    train_metrics_int = simple_classification_report(interview_pred_train, y_train_interview)
    test_metrics_int = simple_classification_report(interview_pred_test, y_test_interview)
    
    # Log metrics
    for metric_name, metric_value in train_metrics_int.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)
    
    for metric_name, metric_value in test_metrics_int.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    # Display metrics
    metrics_df_int = pd.DataFrame([train_metrics_int, test_metrics_int], index=['train', 'test']).round(3)
    print("\nInterview Model Metrics:")
    print(metrics_df_int)
    
    interview_run_id = mlflow.active_run().info.run_id
    print(f"\nInterview Model run ID: {interview_run_id}")

# COMMAND ----------

print("\n" + "="*60)
print("TRAINING PIPELINE COMPLETED")
print("="*60)
print(f"\nDesk Check Model: {desk_check_model_name}")
print(f"Run ID: {desk_check_run_id}")
print(f"\nInterview Model: {interview_model_name}")
print(f"Run ID: {interview_run_id}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## The End
