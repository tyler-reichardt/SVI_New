# Databricks notebook source
# MAGIC %md
# MAGIC ### Overview
# MAGIC
# MAGIC Script to train model for SVI claims

# COMMAND ----------

#import packages:
import pandas as pd
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score, average_precision_score

#from sklearn.pipeline import Pipeline
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
damage_cols = ["damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal",]
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

#other_cols = ['claim_number', 'svi_risk', 'policy_number', 'policy_transaction_id',  'start_date', 'policy_start_date', #'fa_risk', 'fraud_risk', 'tbg_risk']
other_cols = ['claim_number','svi_risk', 'dataset']

useful_cols = mean_fills + bool_cols + damage_cols + one_fills + string_cols + other_cols

table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"
"prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"

# Read in dataset & display
raw_df = spark.table(table_path).withColumn('underbody_damage_severity', lit(None)).select(useful_cols)

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

old_final_features = ['C10_claim_to_policy_end','C2_reporting_delay','C3_weekend_incident_reported_monday','C5_is_night_incident','C9_policy_within_30_days','age_at_policy_start_date_1','annual_mileage','areasDamagedHeavy','areasDamagedMedium','areasDamagedMinimal','areasDamagedSevere','areasDamagedTotal','assessment_category','business_mileage','min_claim_driver_age','claim_driver_age_low','claim_to_policy_end','damageScore','doors_open','first_party_confirmed_tp_notified_claim','front_bonnet_severity','front_severity','impact_speed','impact_speed_range','inception_to_claim','incidentDayOfWeekC','incidentHourC','incidentMonthC','incident_cause','incident_day_of_week','incident_sub_cause','is_crime_reference_provided','is_police_attendance','is_reported_monday','licence_length_years_1','manufacture_yr_claim','max_age_at_policy_start_date','max_cars_in_household','max_licence_length_years','max_years_resident_in_uk','min_age_at_policy_start_date','min_licence_length_years','min_years_resident_in_uk','ncd_protected_flag','notification_method','policy_cover_type','policy_type','policyholder_ncd_years','right_rear_wheel_severity','veh_age','vehicle_overnight_location_id','vehicle_value','voluntary_amount','years_resident_in_uk_1', 'checks_max']

final_features = ['C10_claim_to_policy_end', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C9_policy_within_30_days', 'annual_mileage', 'areasDamagedHeavy', 'areasDamagedMedium', 'areasDamagedMinimal', 'areasDamagedSevere', 'areasDamagedTotal', 'assessment_category', 'business_mileage', 'checks_max', 'claim_to_policy_end', 'damageScore', 'doors_open', 'first_party_confirmed_tp_notified_claim', 'front_bonnet_severity', 'front_severity', 'impact_speed', 'impact_speed_range', 'inception_to_claim', 'incidentDayOfWeekC', 'incidentHourC', 'incidentMonthC', 'incident_cause', 'incident_day_of_week', 'incident_sub_cause', 'is_crime_reference_provided', 'is_police_attendance', 'is_reported_monday', 'manufacture_yr_claim', 'max_cars_in_household', 'min_claim_driver_age', 'min_licence_length_years', 'min_years_resident_in_uk', 'ncd_protected_flag', 'notification_method', 'policy_cover_type', 'policy_type', 'policyholder_ncd_years', 'right_rear_wheel_severity', 'veh_age', 'vehicle_overnight_location_id', 'vehicle_value', 'voluntary_amount',
 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 'C14_contains_watchwords',
 'C1_fri_sat_night', 'C6_no_commuting_but_rush_hour',  'C7_police_attended_or_crime_reference']
 

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

from pyspark.sql.functions import mean
high_cardinality = ['car_group', 'employment_type_abi_code_1', 'employment_type_abi_code_2', 'employment_type_abi_code_3', 'employment_type_abi_code_4', 'employment_type_abi_code_5', 'postcode']
string_cols = list(set(string_cols) - set(high_cardinality))

# Convert Spark DataFrame to Pandas DataFrame
raw_df_pd = raw_df.drop(*high_cardinality).toPandas().sort_values(by='claim_number', ascending=False).drop('claim_number', axis=1)

# Split the data into train, test, and validation sets
#train_df, test_df = train_test_split(raw_df_pd, test_size=0.2, random_state=42, stratify=raw_df_pd.svi_risk)
train_df = raw_df_pd[raw_df_pd.dataset == 'train'].drop('dataset', axis=1)
train_df['svi_risk'] = train_df['svi_risk'].replace(-1, 0)
#train_df = train_df[train_df['svi_risk'].isin([0, 1])]

test_df = raw_df_pd[raw_df_pd.dataset == 'test'].drop('dataset', axis=1)
#test_df = test_df[test_df['svi_risk'].isin([0, 1])]
test_df['svi_risk'] = test_df['svi_risk'].replace(-1, 0)

# Separate features and target variable
X_train = train_df.drop('svi_risk', axis=1)
y_train = train_df['svi_risk']
X_test = test_df.drop('svi_risk', axis=1)
y_test = test_df['svi_risk']
#X_val = val_df.drop('svi_risk', axis=1)
#y_val = val_df['svi_risk']

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_curve, auc, make_scorer


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

def generate_classification_report(model, X_test, y_true, threshold=0.5):   
    y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)
    X_test ['y_pred'] = y_pred
    y_pred =  X_test[["checks_max", "y_pred"]].astype('int').max(axis=1)
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


def pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)  # AUC requires recall (x-axis) and precision (y-axis)
pr_auc_scorer = make_scorer(pr_auc, needs_proba=True)

 


# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperparameter Tuning
# MAGIC This is where model tuning takes place using cross validation. Depending on the model, use case and dataset size GridSearch, RandomSearch or Hyperopt may be used

# COMMAND ----------

#mlflow.autolog()
#mlflow.end_run()
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

model_list = []
model_scores = []
for sampling in [RandomUnderSampler(random_state=12)]: #, RandomUnderSampler(random_state=12), SMOTE(random_state=12), SMOTEENN(random_state=12)]:
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include =['number']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

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
    #xgb_model = xgb.XGBClassifier(objective='binary:logistic')
    xgb_model = LGBMClassifier(verbose=0)
    # Create a pipeline that combines preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampling', sampling),
        ('classifier', xgb_model)
    ])

    # Define the parameter grid
    param_grid = {
        'classifier__n_estimators': [20, 50, 70, 100, 200],
        'classifier__max_depth': [4,6, 8, 16],
        'classifier__learning_rate': [0.01],
        'classifier__num_leaves': [16, 24, 32, 64],
        'classifier__min_child_samples': [8, 16, 32]
    }
    
    # 'min_child_weight': (0, 0.2), 'subsample': (0.8,1)    
    #{'classifier__learning_rate': 0.01, 'classifier__max_depth': 6, 'classifier__n_estimators': 70, 'classifier__num_leaves': 16}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring=pr_auc_scorer,  verbose=0, refit=True)

    # Set up GridSearchCV
    #grid_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, cv=3, #scoring='precision', n_iter=10, verbose=0, refit=True)
    
    # Fit the model
    grid_search.fit(X_train, y_train)

    model_list.append(grid_search)  
    model_scores.append(grid_search.best_score_) 
    print(grid_search)
    print(grid_search.best_score_)

print(model_list)
print(model_scores)

# COMMAND ----------

metrics_list =[]
for this_model in model_list:    
    metrics_list.append( simple_classification_report(this_model, X_train, y_train))
    metrics_list.append( simple_classification_report(this_model, X_test, y_test))

metrics_df = pd.DataFrame(metrics_list, index=['Undersampling', 'Test']).round(3).reset_index()
display(metrics_df)


# COMMAND ----------

print(len(X_train))

# COMMAND ----------



# COMMAND ----------

'''
    'classifier__n_estimators': [50, 70, 100],
    'classifier__max_depth': [3, 6, 16, 32],
    'classifier__learning_rate': [0.01],
    'classifier__num_leaves': [16, 24, 32],
    'classifier__min_child_samples': [32]
'''

# COMMAND ----------



# COMMAND ----------

print("numeric_features: ", numeric_features)
print("categorical_features: ", categorical_features) 

# COMMAND ----------


    # 2^max_depth > num_leaves
    #best: {'classifier__learning_rate': 0.01, 'classifier__max_depth': 16, 'classifier__min_data_in_leaf': 32, 'classifier__n_estimators': 100, 'classifier__num_leaves': 32}'

# COMMAND ----------



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

# if it's TP/comprehensive

# COMMAND ----------

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Calculate permutation importance
result = permutation_importance(
    best_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=20
)

# Get the mean importances and their corresponding feature names
importances_mean = result.importances_mean
features = X_test.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances_mean
}).sort_values(by='Importance', ascending=False).head(20)
display(importance_df)

# Plot the mean permutation importances as a bar chart
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Mean Decrease in Accuracy')
plt.title('Permutation Importances (train set)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

lgb_classifier = best_model.named_steps['classifier']
lg_tree = lgb_classifier.booster_.trees_to_dataframe()

# COMMAND ----------

!pip list

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

preprocessor = grid_search.best_estimator_.named_steps['preprocessor']
# Assuming X_train, X_test, y_train, y_test are already defined
log_reg = LogisticRegression()
log_reg.fit(preprocessor.transform(X_train), y_train)

# Predict using the logistic regression model
y_pred_log_reg = log_reg.predict(preprocessor.transform(X_test))

# Calculate accuracy for logistic regression model
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)

# Assuming best_model is already defined and trained
y_pred_best_model = best_model.predict(X_test)

# Calculate accuracy for the best model
best_model_accuracy = accuracy_score(y_test, y_pred_best_model)

# Display the accuracies
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")
print(f"Best Model Accuracy: {best_model_accuracy}")


# COMMAND ----------

metrics_log_reg = generate_classification_report(log_reg, preprocessor.transform(X_test), y_test)
metrics_best_model = generate_classification_report(best_model, X_test, y_test)
metrics_df = pd.DataFrame([metrics_log_reg, metrics_best_model], index=['log_reg', 'lightgbm']).round(3).reset_index()
display(metrics_df)

# COMMAND ----------

#lgb_classifier._Booster.dump_model()["tree_info"]
#lgb.create_tree_digraph(lgb_classifier.booster_)

tree_df = lgb_classifier.booster_.trees_to_dataframe()
t1 = tree_df[tree_df["tree_index"] == 1]
lgb_classifier.booster_.num_trees()

print(dir(lgb_classifier.booster_))
#lgb_classifier._Booster.dump_model()['tree_info']
#dir(lgb_classifier)
display(lgb_classifier.booster_.trees_to_dataframe())


# COMMAND ----------



# COMMAND ----------

import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges from t1 DataFrame
for _, row in t1.iterrows():
    G.add_node(row['node_index'], value=row['value'])
    if (row['parent_index'] != -1) and (row['parent_index'] is not None): # -1 indicates no parent
        G.add_edge(row['parent_index'], row['node_index'])

# Display the graph
display(G)

nx.draw(G, with_labels=True) 


# COMMAND ----------

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True) 


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit Final Model
# MAGIC Once the optimal model approach and hyperparameters have been identified a final model is fit on the entire training set 

# COMMAND ----------



# COMMAND ----------


