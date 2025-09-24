"""
Model Training Functions for SVI Fraud Detection
Standalone functions extracted from notebooks for model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, make_scorer,
    average_precision_score, precision_score, recall_score, confusion_matrix,
    precision_recall_curve, auc, classification_report
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


# ============================================================================
# Data Preparation Functions
# ============================================================================

def do_fills_pd(raw_df, mean_dict):
    """Apply missing value imputation to pandas DataFrame"""
    
    # Define column groups
    mean_fills = [
        "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", 
        "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", 
        "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", 
        "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", 
        "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", 
        "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", 
        "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", 
        "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", 
        "claim_to_policy_end", "incidentDayOfWeekC"
    ]

    damage_cols = [
        "damageScore", "areasDamagedMinimal", "areasDamagedMedium", 
        "areasDamagedHeavy", "areasDamagedSevere", "areasDamagedTotal"
    ]
    
    bool_cols = [
        "vehicle_unattended", "excesses_applied", "is_first_party", 
        "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
        "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance", 
        "veh_age_more_than_10", "police_considering_actions", "is_crime_reference_provided", 
        "ncd_protected_flag", "boot_opens", "doors_open", "multiple_parties_involved",  
        "is_incident_weekend", "is_reported_monday", "driver_age_low_1", 
        "claim_driver_age_low", "licence_low_1", "total_loss_flag"
    ]

    one_fills = [
        "C1_fri_sat_night", "C2_reporting_delay", "C3_weekend_incident_reported_monday", 
        "C5_is_night_incident", "C6_no_commuting_but_rush_hour", 
        "C7_police_attended_or_crime_reference", "C9_policy_within_30_days", 
        "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
        "C12_expensive_for_driver_age", "C14_contains_watchwords"
    ]

    string_cols = [
        'vehicle_overnight_location_id', 'incidentMonthC', 
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
    
    # Build fillna dictionary
    neg_fills_dict = {x: -1 for x in bool_cols + damage_cols if x in raw_df.columns}
    one_fills_dict = {x: 1 for x in one_fills if x in raw_df.columns}
    string_fills_dict = {x: 'missing' for x in string_cols if x in raw_df.columns}
    
    # Combine all fills
    combined_fills = {**one_fills_dict, **neg_fills_dict, **string_fills_dict, **mean_dict}
    raw_df = raw_df.fillna(combined_fills)

    # Recast data types
    cols_groups = {
        "float": mean_fills + damage_cols,
        "string": string_cols,
        "bool": bool_cols + one_fills 
    }
    
    for dtype, column_list in cols_groups.items():
        existing_cols = [col for col in column_list if col in raw_df.columns]
        if dtype == "float":
            raw_df[existing_cols] = raw_df[existing_cols].astype(float)
        elif dtype == "string":
            raw_df[existing_cols] = raw_df[existing_cols].astype('str')        
        elif dtype == "bool":
            raw_df[existing_cols] = raw_df[existing_cols].astype(int).astype('str')

    return raw_df


# ============================================================================
# Model Evaluation Functions
# ============================================================================

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
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate text report
    text_report = classification_report(y_true, y_pred)
    
    report = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': text_report
    }
    return report


def pr_auc(y_true, y_scores):
    """Calculate PR AUC score"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_training_data(spark, env_config):
    """Load training data from the feature table"""
    from pyspark.sql.functions import col, when, greatest, lit
    
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
    
    # Check which columns exist
    existing_check_cols = [c for c in check_cols if c in raw_df.columns]
    
    if existing_check_cols:
        raw_df = raw_df.withColumn('checks_max', greatest(*[col(c) for c in existing_check_cols]))
    
    # Fix boolean column types
    boolean_cols = ['police_considering_actions', 'is_crime_reference_provided', 
                   'multiple_parties_involved', 'total_loss_flag']
    for bool_col in boolean_cols:
        if bool_col in raw_df.columns:
            raw_df = raw_df.withColumn(bool_col, col(bool_col).cast('boolean'))
    
    return raw_df


def prepare_training_data(df_pd, target_col='fa_risk'):
    """Prepare data for model training"""
    
    # Define feature columns
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
        'areasDamagedSevere', 'areasDamagedTotal'
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
    
    # Filter to existing columns
    numeric_features = [f for f in numeric_features if f in df_pd.columns]
    categorical_features = [f for f in categorical_features if f in df_pd.columns]
    
    # Prepare features and target
    X = df_pd[numeric_features + categorical_features]
    y = df_pd[target_col] if target_col in df_pd.columns else None
    
    return X, y, numeric_features, categorical_features


# ============================================================================
# Model Training Functions
# ============================================================================

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create preprocessing pipeline for model training"""
    
    # Create preprocessors
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns as is
    )
    
    return preprocessor


def train_desk_check_model(X_train, y_train, X_test, y_test, 
                          numeric_features, categorical_features):
    """Train the Desk Check (FA) model"""
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Define model
    lgbm_model = LGBMClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgbm_model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = generate_classification_report(y_pred_proba, y_test)
    
    return pipeline, metrics


def train_interview_model(X_train, y_train, X_test, y_test, 
                         numeric_features, categorical_features):
    """Train the Interview model"""
    
    # Filter to interview model features
    num_interview = [
        'voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 
        'annual_mileage', 'min_claim_driver_age', 'incidentHourC', 
        'areasDamagedHeavy', 'impact_speed', 'years_resident_in_uk_1', 
        'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 
        'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 
        'areasDamagedMedium', 'min_years_resident_in_uk', 'incidentDayOfWeekC',
        'age_at_policy_start_date_1', 'max_age_at_policy_start_date'
    ]
    
    cat_interview = [
        'assessment_category', 'left_severity', 'C9_policy_within_30_days',
        'incident_sub_cause', 'rear_right_severity', 'front_left_severity',
        'rear_window_damage_severity', 'incident_cause', 'checks_max',
        'total_loss_flag'
    ]
    
    # Filter to existing columns
    num_interview = [f for f in num_interview if f in X_train.columns]
    cat_interview = [f for f in cat_interview if f in X_train.columns]
    
    # Subset data
    X_train_int = X_train[num_interview + cat_interview]
    X_test_int = X_test[num_interview + cat_interview]
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(num_interview, cat_interview)
    
    # Define model with tuned hyperparameters
    lgbm_model = LGBMClassifier(
        random_state=42,
        n_estimators=150,
        max_depth=8,
        learning_rate=0.08,
        num_leaves=40,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        class_weight='balanced'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgbm_model)
    ])
    
    # Train model
    pipeline.fit(X_train_int, y_train)
    
    # Make predictions
    y_pred_proba = pipeline.predict_proba(X_test_int)[:, 1]
    
    # Calculate metrics
    metrics = generate_classification_report(y_pred_proba, y_test)
    
    return pipeline, metrics


def hyperparameter_tuning(X_train, y_train, numeric_features, categorical_features,
                         n_iter=20, cv=5, scoring='roc_auc'):
    """Perform hyperparameter tuning using RandomizedSearchCV"""
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Define parameter grid
    param_dist = {
        'classifier__n_estimators': [50, 100, 150, 200],
        'classifier__max_depth': [5, 7, 10, 15],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15],
        'classifier__num_leaves': [20, 31, 50, 100],
        'classifier__min_child_samples': [10, 20, 30, 40],
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'classifier__reg_alpha': [0, 0.01, 0.05, 0.1],
        'classifier__reg_lambda': [0, 0.01, 0.05, 0.1]
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline, 
        param_dist, 
        n_iter=n_iter, 
        cv=cv, 
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


# ============================================================================
# MLflow Functions
# ============================================================================

def register_model_to_mlflow(model, X_train, model_name, run_name, metrics=None):
    """Register model to MLflow model registry"""
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            params = model.named_steps['classifier'].get_params()
            for param, value in params.items():
                mlflow.log_param(param, value)
        
        # Log metrics
        if metrics:
            for metric, value in metrics.items():
                if not isinstance(value, (list, np.ndarray)):
                    mlflow.log_metric(metric, value)
        
        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            registered_model_name=model_name
        )
        
        return run.info.run_id


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models"""
    
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        clf = model.named_steps['classifier']
        
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            
            # Get feature names after preprocessing
            if 'preprocessor' in model.named_steps:
                feature_names_transformed = model.named_steps['preprocessor'].get_feature_names_out()
            else:
                feature_names_transformed = feature_names
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names_transformed,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()
            
            return importance_df
    
    return None
</content>