# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview
# MAGIC
# MAGIC Script to train model for SVI claims
# MAGIC
# MAGIC Model 1: Interview Model
# MAGIC
# MAGIC Models the external processes and interviews

# COMMAND ----------

#import packages:
import pandas as pd
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score, average_precision_score

#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
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



# COMMAND ----------

# columns to fill using mean
mean_fills = [ "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end", "incidentDayOfWeekC", "num_failed_checks"]

#boolean or damage columns with neg fills
damage_cols = ["areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal","areasDamagedRelative"]
bool_cols = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance","is_police_attendance","veh_age_more_than_10","police_considering_actions","is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved",  "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low","licence_low_1", "total_loss_flag"]

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

#aggregate check columns
check_cols = [
    'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
    'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
    'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
    'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
]

#other_cols = ['claim_number', 'svi_risk', 'policy_number', 'policy_transaction_id',  'start_date', 'policy_start_date', #'fa_risk', 'fraud_risk', 'tbg_risk']
other_cols = ['claim_number','svi_risk', 'dataset', 'fa_risk', 'tbg_risk', 'reported_date']

useful_cols = mean_fills + bool_cols + damage_cols + one_fills + string_cols + other_cols

table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"

# Read in dataset & display
raw_df = spark.table(table_path).withColumn('underbody_damage_severity', lit(None))\
                    .withColumn('checks_max', greatest(*[col(c) for c in check_cols]))


############## fix bug in data build 
##C1: was the incident on a Friday/Saturday *NIGHT*?

raw_df = raw_df.withColumn("areasDamagedRelative", col("areasDamagedMinimal") + 2*col("areasDamagedMedium") + 3*col("areasDamagedSevere") + 4*col("areasDamagedHeavy"))\
    .withColumn("incident_day_of_week", date_format(col("reported_date"), "E"))

fri_sat_night = ((col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4))))
                                                                                                                    
raw_df = raw_df.withColumn(
    "C1_fri_sat_night",
    when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0))
#####################

raw_df = raw_df.withColumn(
    "checks_list",
    array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in check_cols])
)

raw_df = raw_df.withColumn(
    "checks_list",
    expr("filter(checks_list, x -> x is not null)")
).withColumn("num_failed_checks", size(col("checks_list")))

raw_df = raw_df.select(useful_cols)

#raw_df = raw_df.withColumn('svi_risk', when(col('tbg_risk').isin([0, 1]), 1).otherwise(0))

#fix issue with type of some boolean columns
raw_df = raw_df.withColumn('police_considering_actions', col('police_considering_actions').cast('boolean'))
raw_df = raw_df.withColumn('is_crime_reference_provided', col('is_crime_reference_provided').cast('boolean'))
raw_df = raw_df.withColumn('multiple_parties_involved', col('multiple_parties_involved').cast('boolean'))

# see if any of the checks are true
raw_df = raw_df.withColumn('checks_max', greatest(*[col(c) for c in check_cols])) 
one_fills.append("checks_max") # add to list of features

final_features1c = ['C10_claim_to_policy_end', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C9_policy_within_30_days', 'annual_mileage', 'areasDamagedHeavy', 'areasDamagedMedium', 'areasDamagedMinimal', 'areasDamagedSevere', 'areasDamagedTotal', 'assessment_category', 'business_mileage', 'checks_max', 'claim_to_policy_end',  'doors_open', 'first_party_confirmed_tp_notified_claim', 'front_bonnet_severity', 'front_severity', 'impact_speed', 'impact_speed_range', 'inception_to_claim', 'incidentDayOfWeekC', 'incidentHourC', 'incidentMonthC', 'incident_cause', 'incident_day_of_week', 'incident_sub_cause', 'is_crime_reference_provided', 'is_police_attendance', 'is_reported_monday', 'manufacture_yr_claim', 'max_cars_in_household', 'min_claim_driver_age', 'min_licence_length_years', 'min_years_resident_in_uk', 'ncd_protected_flag', 'notification_method', 'policy_cover_type', 'policy_type', 'policyholder_ncd_years', 'right_rear_wheel_severity', 'veh_age', 'vehicle_overnight_location_id', 'vehicle_value', 'voluntary_amount',
 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 'C14_contains_watchwords',
 'C1_fri_sat_night', 'C6_no_commuting_but_rush_hour',  'C7_police_attended_or_crime_reference', 'areasDamagedRelative'] #'damageScore',

# presented on 24/04/2024
final_features = ['voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 'annual_mileage', 'min_claim_driver_age', 'incidentHourC', 'assessment_category', 'left_severity', 'C9_policy_within_30_days', 'areasDamagedHeavy', 'impact_speed', 'years_resident_in_uk_1', 'incident_sub_cause', 'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 'rear_right_severity', 'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 'num_failed_checks', 'front_left_severity', 'areasDamagedMedium', 'rear_window_damage_severity', 'incident_cause', 'min_years_resident_in_uk', 'incidentDayOfWeekC', 'age_at_policy_start_date_1', 'max_age_at_policy_start_date', 'checks_max', 'total_loss_flag']

final_features = ['voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 'annual_mileage', 'min_claim_driver_age', 'incidentHourC', 'assessment_category', 'left_severity', 'C9_policy_within_30_days', 'areasDamagedHeavy', 'impact_speed', 'incident_sub_cause', 'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 'rear_right_severity', 'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 'num_failed_checks', 'front_left_severity', 'areasDamagedMedium', 'rear_window_damage_severity', 'incident_cause', 'incidentDayOfWeekC', 'age_at_policy_start_date_1', 'checks_max', 'total_loss_flag']


# COMMAND ----------



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



# COMMAND ----------

run_id = "56ced60a6d464b07835f5a237425b33b"
logged_model = f'runs:/{run_id}/model'
fa_model = mlflow.sklearn.load_model(logged_model)
fa_numeric = ['policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 'veh_age', 'business_mileage', 'annual_mileage', 'incidentHourC', 'additional_vehicles_owned_1', 'age_at_policy_start_date_1', 'cars_in_household_1', 'licence_length_years_1', 'years_resident_in_uk_1', 'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'max_cars_in_household', 'min_cars_in_household', 'max_licence_length_years', 'min_licence_length_years', 'max_years_resident_in_uk', 'min_years_resident_in_uk', 'impact_speed', 'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim', 'outstanding_finance_amount', 'claim_to_policy_end', 'incidentDayOfWeekC', 'areasDamagedMinimal', 'areasDamagedMedium', 'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal']

fa_categorical = ['vehicle_unattended', 'excesses_applied', 'is_first_party', 'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 'claim_driver_age_low', 'licence_low_1', 'C1_fri_sat_night', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 'C14_contains_watchwords', 'vehicle_overnight_location_id', 'incidentMonthC', 'policy_type', 'assessment_category', 'engine_damage', 'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 'policy_cover_type', 'notification_method', 'impact_speed_unit', 'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 'incident_day_of_week', 'reported_day_of_week', 'checks_max']

# COMMAND ----------

mlflow.autolog(disable=True) 
from pyspark.sql.functions import mean

high_cardinality = ['car_group', 'employment_type_abi_code_1', 'employment_type_abi_code_2', 'employment_type_abi_code_3', 'employment_type_abi_code_4', 'employment_type_abi_code_5', 'postcode', 'damageScore']
string_cols = list(set(string_cols) - set(high_cardinality))

# Convert Spark DataFrame to Pandas DataFrame
raw_df_pd = raw_df.drop(*high_cardinality).toPandas().sort_values(by='claim_number', ascending=False).drop(['claim_number', 'fa_risk', 'tbg_risk'], axis=1)

#using june 2023 - june 2024 (most complete data)
#raw_df_pd = raw_df_pd[(raw_df_pd.reported_date<'2024-06-30') & (raw_df_pd.reported_date>'2023-06-01')]
raw_df_pd = raw_df_pd[raw_df_pd.reported_date<='2024-07-31']
raw_df_pd = raw_df_pd.drop(['reported_date'], axis=1)

# Split the data into train, test, and validation sets
#train_df, test_df = train_test_split(raw_df_pd, test_size=0.2, random_state=42, stratify=raw_df_pd.svi_risk)
train_df = raw_df_pd[raw_df_pd.dataset == 'train'].drop('dataset', axis=1)
train_df['svi_risk'] = train_df['svi_risk'].replace(-1, 0)
#train_df= train_df[train_df.svi_risk.isin([0,1])]


#train_non_tbg = train_df[train_df['svi_risk']==-1] #these were not specially investigated

test_df = raw_df_pd[raw_df_pd.dataset == 'test'].drop('dataset', axis=1)
test_df['svi_risk'] = test_df['svi_risk'].replace(-1, 0)
#test_df= test_df[test_df.svi_risk.isin([0,1])]

# Separate features and target variable
X_train = train_df.drop('svi_risk', axis=1)
y_train = train_df['svi_risk']
X_test = test_df.drop('svi_risk', axis=1)
y_test = test_df['svi_risk']
#X_val = val_df.drop('svi_risk', axis=1)
#y_val = val_df['svi_risk']

REFRESH_MEANS = False 
if REFRESH_MEANS:
    # Calculate mean for each column
    mean_dict = X_train[mean_fills].astype('float').mean().round(4).to_dict()
    print(mean_dict)
else: 
    #from desk check (FA) SVI model
    mean_dict = {'policyholder_ncd_years': 6.7402, 'inception_to_claim': 144.9076, 'min_claim_driver_age': 37.9725, 'veh_age': 11.1574, 'business_mileage': 301.2223, 'annual_mileage': 7388.2417, 'incidentHourC': 12.7965, 'additional_vehicles_owned_1': 0.0018, 'age_at_policy_start_date_1': 39.9154, 'cars_in_household_1': 1.8266, 'licence_length_years_1': 15.6955, 'years_resident_in_uk_1': 35.0835, 'max_additional_vehicles_owned': 0.0028, 'min_additional_vehicles_owned': 0.0013, 'max_age_at_policy_start_date': 43.485, 'min_age_at_policy_start_date': 35.8294, 'max_cars_in_household': 1.8847, 'min_cars_in_household': 1.7608, 'max_licence_length_years': 18.5382, 'min_licence_length_years': 12.4781, 'max_years_resident_in_uk': 38.8198, 'min_years_resident_in_uk': 30.9442, 'impact_speed': 26.4062, 'voluntary_amount': 240.4995, 'vehicle_value': 7964.3209, 'manufacture_yr_claim': 2012.1099, 'outstanding_finance_amount': 0.0, 'claim_to_policy_end': 77.5255, 'incidentDayOfWeekC': 4.022}

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

X_train['fa_pred'] = fa_model.predict_proba(X_train[fa_numeric + fa_categorical])[:,1]
X_test['fa_pred'] = fa_model.predict_proba(X_test[fa_numeric + fa_categorical])[:,1]

X_train = X_train[final_features] # + ['fa_pred']]
X_test = X_test[final_features] # + ['fa_pred']]

display(X_train.dtypes.reset_index().astype(str))


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

y_train.value_counts()
8188/201
len(y_test)/(len(y_test) + len(y_train))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tune model

# COMMAND ----------


from imblearn.under_sampling import RandomUnderSampler

sampling = RandomUnderSampler(random_state=12)

with mlflow.start_run():
    mlflow.sklearn.autolog()
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include =['number']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    #print("numeric: ", numeric_features)
    #print("categorical_features: ", categorical_features)

    # Preprocessing for numeric data
    #numeric_transformer = Pipeline(steps=[ ('scaler', StandardScaler()) ])
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

    #X_prep_train = preprocessor.fit_transform(X_train) 
    #X_prep_test = preprocessor.transform(X_test)

    #mlflow.sklearn.log_model(preprocessor, "preprocessor")

    xgb_model = LGBMClassifier(verbose=0)
    # Create a pipeline that combines preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampling', sampling),
        ('classifier', xgb_model)
    ])

    # Define the parameter grid
    # Define the parameter grid
    param_grid = {'classifier__n_estimators': [10, 20, 30, 50],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.1],
        'classifier__num_leaves': [5, 10, 15, 31, 61],
        'classifier__min_child_weight': [0.1, 0.5],
        'classifier__scale_pos_weight': [1]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1', verbose=0)
    
    #X_prep_train = preprocessor.fit_transform(X_train)

    all_features = list(X_train.columns)

    # Fit the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)

    signature = infer_signature(X_train, best_model.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model, artifact_path="model", signature=signature,
        pyfunc_predict_fn="predict_proba"
    )
    mlflow.log_params(grid_search.best_params_)

    print(model_info)

    metrics_train = simple_classification_report(grid_search, X_train, y_train)
    metrics_test = simple_classification_report(grid_search, X_test, y_test)

    metrics_df = pd.DataFrame([metrics_train, metrics_test], index=['train', 'test']).round(3).reset_index()
    display(metrics_df)


# COMMAND ----------

y_prob_train = grid_search.predict(X_train)

print(pd.DataFrame(y_prob_train).value_counts())

# COMMAND ----------

preprocessor = grid_search.best_estimator_.named_steps['preprocessor']
X_train_prep = pd.DataFrame(preprocessor.transform(X_train), columns=preprocessor.get_feature_names_out())
'''
X_train_prep['target'] = y_train
correlation_with_target = X_train_prep.corr()['target'].drop('target').sort_values(ascending=False)
display(pd.DataFrame(correlation_with_target).reset_index())
'''
correlation_matrix = X_train_prep.corr().abs()
high_corr_pairs = (
    correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']


high_corr_pairs['Same_Characters'] = high_corr_pairs.apply(
    lambda row: row['Feature_1'].rsplit('_', 1)[0] == row['Feature_2'].rsplit('_', 1)[0], axis=1
)

high_corr_pairs = high_corr_pairs[(high_corr_pairs['Correlation'] > 0.5) & (high_corr_pairs['Correlation'] != 1) & (high_corr_pairs['Same_Characters'] == False)]
display(high_corr_pairs)


# COMMAND ----------

feature_names = grid_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_

mdi_importances = pd.Series(
    importances, index=feature_names
).sort_values(ascending=False)
display(pd.DataFrame(mdi_importances).reset_index())
ax = mdi_importances.head(20).plot.barh()
ax.set_title("Feature Importances")
ax.figure.tight_layout()

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_prob_test = grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_prob_test)
print('Confusion matrix: ', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['0', '1'])
disp.plot()

# COMMAND ----------

print("numeric: ", numeric_features)
print("categorical_features: ", categorical_features)

# COMMAND ----------



# COMMAND ----------

import matplotlib.pyplot as plt

#PR curve
y_prob = grid_search.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
average_precision = average_precision_score(y_test, y_prob)

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


