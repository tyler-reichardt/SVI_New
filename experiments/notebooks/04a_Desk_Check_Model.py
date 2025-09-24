# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview
# MAGIC
# MAGIC Script to train model for SVI claims
# MAGIC
# MAGIC Model 1: Desk Check (FA) Model
# MAGIC
# MAGIC Models the internal checks current at project inception

# COMMAND ----------

#import packages:
import pandas as pd
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score, average_precision_score, precision_recall_curve

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

#mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Read
# MAGIC Common steps:
# MAGIC - Read in data
# MAGIC - Perform train test split
# MAGIC - Save off train and test sets
# MAGIC - Dataframe size checks

# COMMAND ----------

# columns to fill using mean
mean_fills = [ "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", "incidentDayOfWeekC"]

#boolean or damage columns with neg fills
damage_cols = ["areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal",]
bool_cols = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance","is_police_attendance","veh_age_more_than_10","police_considering_actions","is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved",  "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low","licence_low_1"]
#neg_fills = bool_cols + damage_cols

# fills with ones (rules variables, to trigger manual check)
one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords",]

#fill with word 'missing' (categoricals) 
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

other_cols = ['claim_number','svi_risk', 'dataset', 'fa_risk', 'tbg_risk', 'referred_to_tbg', 'reported_date']

useful_cols = mean_fills + bool_cols + damage_cols + one_fills + string_cols + other_cols

table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"

# Read in dataset & display
raw_df = spark.table(table_path)
raw_df = raw_df.withColumn('referred_to_tbg', when(col('tbg_risk').isin([0, 1]), 1).otherwise(0))
raw_df = raw_df.withColumn('underbody_damage_severity', lit(None)).select(useful_cols)

#fix issue with type of some boolean columns
raw_df = raw_df.withColumn('police_considering_actions', col('police_considering_actions').cast('boolean'))
raw_df = raw_df.withColumn('is_crime_reference_provided', col('is_crime_reference_provided').cast('boolean'))
raw_df = raw_df.withColumn('multiple_parties_involved', col('multiple_parties_involved').cast('boolean'))

#aggregate check columns
check_cols = [
    'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
    'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
    'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
    'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
]

# see if any of the checks are true
raw_df = raw_df.withColumn('checks_max', greatest(*[col(c) for c in check_cols])) 
one_fills.append("checks_max") # add to list of features
display(raw_df.limit(100)) 

final_features = ['C10_claim_to_policy_end', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C9_policy_within_30_days', 'annual_mileage', 'areasDamagedHeavy', 'areasDamagedMedium', 'areasDamagedMinimal', 'areasDamagedSevere', 'areasDamagedTotal', 'assessment_category', 'business_mileage', 'checks_max', 'claim_to_policy_end', 'damageScore', 'doors_open', 'first_party_confirmed_tp_notified_claim', 'front_bonnet_severity', 'front_severity', 'impact_speed', 'impact_speed_range', 'inception_to_claim', 'incidentDayOfWeekC', 'incidentHourC', 'incidentMonthC', 'incident_cause', 'incident_day_of_week', 'incident_sub_cause', 'is_crime_reference_provided', 'is_police_attendance', 'is_reported_monday', 'manufacture_yr_claim', 'max_cars_in_household', 'min_claim_driver_age', 'min_licence_length_years', 'min_years_resident_in_uk', 'ncd_protected_flag', 'notification_method', 'policy_cover_type', 'policy_type', 'policyholder_ncd_years', 'right_rear_wheel_severity', 'veh_age', 'vehicle_overnight_location_id', 'vehicle_value', 'voluntary_amount',
 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 'C14_contains_watchwords',
 'C1_fri_sat_night', 'C6_no_commuting_but_rush_hour',  'C7_police_attended_or_crime_reference']
 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing
# MAGIC Common steps:
# MAGIC - Missing imputation
# MAGIC - Categorical encoding
# MAGIC - Standard scaling 
# MAGIC
# MAGIC Ideally these stages would be assembled as part of a preprocessing and modelling pipeline but sometimes it will be preferable to perform separately

# COMMAND ----------

from pyspark.sql.functions import mean

high_cardinality = ['car_group', 'employment_type_abi_code_1', 'employment_type_abi_code_2', 'employment_type_abi_code_3', 'employment_type_abi_code_4', 'employment_type_abi_code_5', 'postcode', 'damageScore']
string_cols = list(set(string_cols) - set(high_cardinality))

# Convert Spark DataFrame to Pandas DataFrame
raw_df_pd = raw_df.drop(*high_cardinality).toPandas().sort_values(by='claim_number', ascending=False)#.drop(['claim_number', 'fa_risk', 'tbg_risk'], axis=1)



# COMMAND ----------

#volumes by month + mean target 
raw_df_pd['svi_risk'] = raw_df_pd['svi_risk'].replace(-1, 0)
raw_df_pd['reported_date'] = pd.to_datetime(raw_df_pd['reported_date'])
display(raw_df_pd)

# COMMAND ----------

#off back of the above decide to use june 2023 - june 2024
raw_df_pd = raw_df_pd[(raw_df_pd.reported_date<='2024-07-31')]

# COMMAND ----------

raw_df_pd.referred_to_tbg.describe()

# COMMAND ----------

import matplotlib.pyplot as plt
raw_df_pd.referred_to_tbg.hist()

# COMMAND ----------

raw_df_pd.shape

# COMMAND ----------

display(raw_df_pd)

# COMMAND ----------

#create train and test now 
# Split the data into train, test, and validation sets 
#train_df, test_df = train_test_split(raw_df_pd, test_size=0.2, random_state=42, stratify=raw_df_pd.svi_risk)

train_df = raw_df_pd[raw_df_pd.dataset == 'train'].drop('dataset', axis=1)
train_df['svi_risk'] = train_df['svi_risk'].replace(-1, 0)
#train_non_tbg = train_df[train_df['svi_risk']==-1] #these were not specially investigated

test_df = raw_df_pd[raw_df_pd.dataset == 'test'].drop('dataset', axis=1)
test_df['svi_risk'] = test_df['svi_risk'].replace(-1, 0)

# Separate features and target variable
X_train = train_df.drop(['referred_to_tbg','claim_number', 'fa_risk', 'tbg_risk', 'reported_date', 'svi_risk'], axis=1)
y_train = train_df['referred_to_tbg']
X_test = test_df.drop(['referred_to_tbg','claim_number', 'fa_risk', 'tbg_risk', 'reported_date', 'svi_risk'], axis=1)
y_test = test_df['referred_to_tbg']

REFRESH_MEANS = True
if REFRESH_MEANS:
    # Calculate mean for each column
    mean_dict = X_train[mean_fills].astype('float').mean().round(4).to_dict()
    print(mean_dict)
else: 
    #replace if data is uploaded
    mean_dict = {'policyholder_ncd_years': 6.5934, 'inception_to_claim': 141.7436, 'min_claim_driver_age': 37.0723, 'veh_age': 11.4678, 'business_mileage': 276.1819, 'annual_mileage': 7355.2092, 'incidentHourC': 12.741, 'additional_vehicles_owned_1': 0.003, 'age_at_policy_start_date_1': 39.1134, 'cars_in_household_1': 1.8386, 'licence_length_years_1': 15.0401, 'years_resident_in_uk_1': 34.0761, 'max_additional_vehicles_owned': 0.0036, 'min_additional_vehicles_owned': 0.001, 'max_age_at_policy_start_date': 42.8814, 'min_age_at_policy_start_date': 34.9985, 'max_cars_in_household': 1.8968, 'min_cars_in_household': 1.7635, 'max_licence_length_years': 17.9961, 'min_licence_length_years': 11.8057, 'max_years_resident_in_uk': 37.9595, 'min_years_resident_in_uk': 29.8881, 'impact_speed': 28.0782, 'voluntary_amount': 236.8366, 'vehicle_value': 7616.7295, 'manufacture_yr_claim': 2011.7687, 'outstanding_finance_amount': 0.0, 'claim_to_policy_end': 83.6548}    

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

X_train =  do_fills_pd(X_train)
X_test =  do_fills_pd(X_test)

#X_train = X_train[final_features] 
#X_test = X_test[final_features] 

display(X_train.dtypes.reset_index().astype(str))

# COMMAND ----------

print(X_train.shape)
print(X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning
# MAGIC This is where model tuning takes place using cross validation. Depending on the model, use case and dataset size GridSearch, RandomSearch or Hyperopt may be used

# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

def simple_classification_report(model, X_test, y_true, threshold=0.5):
    y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)
    y_prob = model.predict_proba(X_test)[:,1]
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
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)  # AUC requires recall (x-axis) and precision (y-axis)
pr_auc_scorer = make_scorer(pr_auc, needs_proba=True)

# COMMAND ----------

mlflow.sklearn.autolog(max_tuning_runs=None)

with mlflow.start_run():
    # mlflow.sklearn.autolog()
    mlflow.set_tag("model_name", "svi_referred_to_tbg_hb")
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include =['number']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    print("numeric: ", numeric_features)
    print("categorical_features: ", categorical_features)

    # Preprocessing for numeric data
    #numeric_transformer = Pipeline(steps=[ ('scaler', StandardScaler()) ])
    #don't do scaling
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
        ])

    # Define the model
    xgb_model = LGBMClassifier(verbose=-1, scale_pos_weight=12)
    # Create a pipeline that combines preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])
   
    # Define the parameter grid
    param_grid = {
        'classifier__n_estimators': [10, 20, 30, 50],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__learning_rate': [0.1],
        'classifier__num_leaves': [5, 10, 15, 31],
        'classifier__min_child_weight': [0.1, 0.5]
    }
  
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='recall', verbose=0)

    grid_search.fit(X_train, y_train)

    # Log the best model
    best_model = grid_search.best_estimator_    
    
    signature = infer_signature(X_train, best_model.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model, artifact_path="model", signature=signature,
        pyfunc_predict_fn="predict_proba"
    )
    mlflow.log_params(grid_search.best_params_)

    print(model_info)

    # mlflow.log_table(mean_dict)

    # Log the best parameters
    print(grid_search.best_params_)

    # Predict on train set
    train_predictions = grid_search.predict(X_train)

    # Log validation metrics
    train_f1 = f1_score(y_train, train_predictions)
    print("train_f1_score", train_f1)

    # Predict on test set
    test_predictions = grid_search.predict(X_test)

    # Log test metrics
    test_f1 = f1_score(y_test, test_predictions)
    print("test_f1_score", test_f1)

    metrics_train = simple_classification_report(grid_search, X_train, y_train)
    metrics_test = simple_classification_report(grid_search, X_test, y_test)

    metrics_df = pd.DataFrame([metrics_train, metrics_test], index=['train', 'test']).round(3).reset_index()
    display(metrics_df)

# COMMAND ----------

print(y_train.value_counts())
7653/736

# COMMAND ----------

# MAGIC %md
# MAGIC #Evaluation

# COMMAND ----------

test_probas = grid_search.predict_proba(X_test)[:, 1]
test_df['proba_pred'] = test_probas
plt.hist(test_probas)

# COMMAND ----------

importances = best_model.named_steps['classifier'].feature_importances_
feature_names = best_model[:-1].get_feature_names_out()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

importance_df['importance_share'] = importance_df.importance / importance_df.importance.sum()

importance_df.head(20)

# COMMAND ----------

#PR curve
precision, recall, thresholds = precision_recall_curve(y_test, test_probas)
average_precision = average_precision_score(y_test, test_probas)

#plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision:.2f})', color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

#PR curve for overall svi risk
#PR curve
precision, recall, thresholds = precision_recall_curve(test_df.svi_risk, test_probas)
average_precision = average_precision_score(test_df.svi_risk, test_probas)

#plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision:.2f})', color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Referral Volume Analysis

# COMMAND ----------

display(test_df)

# COMMAND ----------

from sklearn.metrics import classification_report
#check referrals per month and number of cases captured 
test_df['reported_year_month'] = test_df.reported_date.dt.to_period('M')

#tune around with thresh here:
thresh = 0.52

test_df['prediction'] = test_df['proba_pred'] > thresh

#avg monthly referrals 
print('Avg Monthly Referrals: ', int(test_df.groupby('reported_year_month')['prediction'].sum().mean() / 0.3), '\n')

# class report for 
print('TBG Referral Classification Report: \n', classification_report(y_test, test_df['prediction']))

#Class report for overall svi risk
print('SVI Risk Classification Report: \n', classification_report(test_df.svi_risk, test_df['prediction']))

#plot monthly referrals
(test_df.groupby('reported_year_month')['prediction'].sum() / 0.3).plot(kind='bar', color='purple', title='Referrals per Month')
plt.show()
