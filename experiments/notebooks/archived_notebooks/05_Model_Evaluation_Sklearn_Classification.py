# Databricks notebook source
# MAGIC %md
# MAGIC # PBC - Performance Metrics Template

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook template is used for evaluating model performance. 
# MAGIC * Performance metrics
# MAGIC * Confusion matrix 
# MAGIC * Precision / Recall Curve 
# MAGIC * Lift charts / double lift charts 
# MAGIC * AvsE charts 
# MAGIC *Histogram of errors 
# MAGIC * Hosmer plots
# MAGIC * Feature importance / SHAP beeswarm plots
# MAGIC * PDPs
# MAGIC * Feature interaction strength 
# MAGIC * Translate model performance into expected benefit to business
# MAGIC * Playback to stakeholders
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Import libraries
# MAGIC

# COMMAND ----------


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql.functions import *

from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.inspection import partial_dependence
from collections import Counter


# COMMAND ----------

# MAGIC %md
# MAGIC #Get predictions

# COMMAND ----------

import mlflow
import pandas as pd

#run_id = "4bcd9e4ff12541ecad68a71cc402ce0c"
run_id = "0282be423ffc4d30b1cf7399a6ccaa94"
logged_model = f'runs:/{run_id}/model'
artifact_path = "test_data/X_test.csv"

# Download the artifact
local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

numeric_features = ['annual_mileage', 'areasDamagedHeavy', 'areasDamagedMedium', 'areasDamagedMinimal', 'areasDamagedSevere', 'areasDamagedTotal', 'business_mileage', 'min_claim_driver_age', 'claim_to_policy_end', 'damageScore', 'impact_speed', 'inception_to_claim', 'incidentDayOfWeekC', 'incidentHourC', 'manufacture_yr_claim', 'max_cars_in_household', 'min_licence_length_years', 'min_years_resident_in_uk', 'policyholder_ncd_years', 'veh_age', 'vehicle_value', 'voluntary_amount']

categorical_features = ['C10_claim_to_policy_end', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C9_policy_within_30_days', 'assessment_category', 'doors_open', 'first_party_confirmed_tp_notified_claim', 'front_bonnet_severity', 'front_severity', 'impact_speed_range', 'incidentMonthC', 'incident_cause', 'incident_day_of_week', 'incident_sub_cause', 'is_crime_reference_provided', 'is_police_attendance', 'is_reported_monday', 'ncd_protected_flag', 'notification_method', 'policy_cover_type', 'policy_type', 'right_rear_wheel_severity', 'vehicle_overnight_location_id', 'checks_max']


['age_at_policy_start_date_1', 'licence_length_years_1', 'max_age_at_policy_start_date', 'max_licence_length_years', 'max_years_resident_in_uk', 'min_age_at_policy_start_date', 'years_resident_in_uk_1']

# Load the CSV file
df = pd.read_csv(local_path)

def set_types(raw_df):
    # Recast data types & check schema
    cols_groups = {
        "float": numeric_features,
        "string": categorical_features 
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

df =  set_types(df)

# Load model
pipeline = mlflow.sklearn.load_model(logged_model)
y_pred = pipeline.predict(df[numeric_features + categorical_features])
y_prob = pipeline.predict_proba(df[numeric_features + categorical_features])[:,1]
y_test = df['y_test']

model = pipeline.steps[1][1]
preprocessor = pipeline.named_steps['preprocessor']

X_test = preprocessor.transform(df[numeric_features + categorical_features]).toarray()
X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out()) 
print(list(X_test.columns)) 

display(X_test)

# COMMAND ----------

preprocessor.get_feature_names_out()

#print(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #Confusion metrics

# COMMAND ----------


cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix: ', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['0', '1'] )
disp.plot()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Classification Report

# COMMAND ----------

print(classification_report(y_test, y_pred))


# COMMAND ----------

# MAGIC %md
# MAGIC # Gini Score

# COMMAND ----------

def gini(y_true, y_pred):
  # Sort actual and predicted values by predicted values
  assert len(y_true) == len(y_pred)
  
  # Sort by predicted values, shows the ranking order your model has given
  all_data = np.array(list(zip(y_pred, y_true)))
  all_data = all_data[all_data[:, 0].argsort()]
  
  # Get the sorted predicted and actual values
  sorted_predicted = all_data[:, 0]
  sorted_actual = all_data[:, 1]
  
  # Cumulative sum of actual values
  cumulative_actual = np.cumsum(sorted_actual)
  cumulative_actual_normalized = cumulative_actual / cumulative_actual[-1]
  # this array shows the proportional representation of where the actual volumes sit on your predicted specturm
  
  # Gini formula
  gini_score = 1 - 2 * np.sum(cumulative_actual_normalized) / len(y_true) + 1 / len(y_true)
    
  return gini_score
  
def gini_normalized(y_true, y_pred):
  # we normalize the gini score to account for the fact that the actual distribution may not be uniform
  # so by dividing through by that, we see the proportional impact that our predictions have on the order
  return gini(y_true, y_pred) / gini(y_true, y_true)

# COMMAND ----------

print("Gini Score: " + str(gini_normalized(y_test, y_prob)))

# COMMAND ----------

# MAGIC %md
# MAGIC #Precision / Recall Curve

# COMMAND ----------

PrecisionRecallDisplay.from_predictions( y_test, y_prob, plot_chance_level=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #Lift charts / double lift charts

# COMMAND ----------

#add this script to the workspace and import the relevant function
#https://dev.azure.com/firstcentralsg/Data%20Science/_git/CodingToolkit?path=/model_evaluate/lift_chart_plots.py. Sample function call is below

#plot_lift_chart_regression(y_pred, y_test, n_bins=10, weights=None)

# COMMAND ----------

# MAGIC %md
# MAGIC #AvsE charts

# COMMAND ----------

#sample code to be added

# COMMAND ----------

# MAGIC %md
# MAGIC #Histogram of errors

# COMMAND ----------

y_errors = y_pred - y_test

plt.hist(y_errors)

# COMMAND ----------

# TBC

# COMMAND ----------

# MAGIC %md
# MAGIC #In-built Feature importance

# COMMAND ----------

# function that calculates features importance for all features in a dataset
def get_feature_importance_table(model, df):
  """
  Parameters:
  ----------
  model : trained model
      Function will accept any model that works with the partial dependence function from sklearn.inspection
      This includes most tree based models

  df : pandas dataframe
      X_test holdout dataset

  Returns:
  -------
  pandas dataframe

  Dataframe will have a row per feature, with columns to represent the total importance, as well as the porportional importance

  """
  # create dataframe with row per feature
  df_imp = pd.DataFrame(df.columns, columns=['feature'])
  # add in the feature importances
  # if statements to support varying model formats
  if hasattr(model, 'feature_importances_'):
    df_imp['importance'] = model.feature_importances_
  elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_importance'):
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type=importance_type)
    df_imp['importance'] = [importance_dict.get(f, 0) for f in df.columns]
  elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_importance'):
    df_imp['importance'] = model.booster_.feature_importance(importance_type='gain')
  else:
    raise ValueError("Unsupported model type. Must be scikit-learn, XGBoost, or LightGBM model.")

  # calculate relative proportion of gain explained by each feature
  df_imp['importance_pc'] = 100*df_imp['importance']/df_imp['importance'].sum()
  return df_imp.head(20)

def plot_importance(df_imp, num_features=0):
  """
  Parameters:
  ----------
  df_imp : pandas dataframe
      Designed to take output of get_feature_importance_table function
      Require columns for feature and importance_pc

  num_features : int
      Positive non-zero integer to state the number of most important features to plot in chart

  Returns:
  -------
  null (Will plot feature importance chart)

  """
  # sort table by importance, ascending fixes order of chart
  df = df_imp.sort_values('importance_pc',ascending=True)
  # apply cap to number of features plotted
  if num_features > 0:  
    # use tail as values sorted in ascending order
    df.tail(num_features).plot(kind='barh', x='feature', y='importance_pc', color='skyblue', legend=False)
  else:
    df.plot(kind='barh', x='feature', y='importance_pc', color='skyblue', legend=False)

  plt.xlabel('Feature Importance % (Gain)')
  plt.ylabel('Feature')
  plt.title('Feature Importance (% Gain explained)')

  # Show the plot
  plt.show()

# COMMAND ----------

plot_importance(get_feature_importance_table(model, X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC #SHAP beeswarm plots

# COMMAND ----------

import shap

# Initialize the SHAP TreeExplainer with the model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test dataset
shap_values = explainer.shap_values(X_test)

# Generate a summary plot of the SHAP values
shap.summary_plot(shap_values, X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #PDPs

# COMMAND ----------

# Define function to plot partial dependence for a list of features, given a model and dataset

def plot_pdps(model, df, feature_list=[]):
  """
  Plots PDPs with volume distributions, for a given set of features
  for a given model and dataset

  Parameters:
  ----------
  model : trained model
      Function will accept any model that works with the partial dependence function from sklearn.inspection
      This includes most tree based models

  df : pandas dataframe
      Dataset which you wish to calculate PDPs for

  feature_list (optional) : list of strings
      denoting names of features from the dataset for which you wish to plot partial dependence

  Returns:
  -------
  null (Will plot to cell output the required charts)

  """
  # by default, use all features in dataset
  if feature_list == []:
    feature_list = df.columns
  else:
    # check if passed features are in dataset, throw error if not
    for feature in feature_list:
      if feature not in df.columns:
        raise ValueError("Feature '" + str(feature) + "' not in dataset")

  # iterate through features, plot chart for each
  for feature in feature_list:
    # calculate partial dependence
    pdp_result = partial_dependence(model, df, [feature], kind="average")

    # We want to plot on same graph the distribution of the feature, so next section of code get the counts for 
    # each value evauluated as part of the partial dependence calculations
    
    # Get list of analysed grid values of feature
    grid_values = pdp_result["values"][0]
    #grid_values = pdp_result["grid_values"][0]
    
    # Get list of raw values of feature
    feature_values = np.array(df[feature])
    # Count how many cases fall into each grid value
    counts = Counter(np.digitize(feature_values, grid_values))
    # Look up by index to get ordered volume for each grid value
    freq = [counts[i+1] for i in range(len(grid_values))]

    # Plot the PDPs and Feature distributions on the same graph
    fig, ax1 = plt.subplots(figsize=(12,8))

    # Plot the PDP on the primary y-axis
    ax1.plot(grid_values, pdp_result['average'][0], label='Partial Dependence', marker='o', color='blue')
    ax1.set_xlabel('Value of ' + str(feature))
    ax1.set_ylabel('Partial dependence')
    ax1.tick_params(axis="y")

    grid_range = max(grid_values) - min(grid_values)

    # Create a second y-axis for the volume distribution
    ax2 = ax1.twinx() 
    ax2.bar(grid_values, freq, color="grey", alpha=0.4, width = grid_range/len(grid_values)*0.9, label="Volume")
    ax2.set_ylabel("Volume")
    ax2.tick_params(axis="y")

    fig.legend()

    plt.title('PDP for ' + str(feature))
    plt.show()
    
    # End of for loop for each feature

  # End of plot_pdps function

# COMMAND ----------

# Using example models built in template

plot_pdps(model, X_test, feature_list=X_test.columns[:10].tolist())
print([*X_test.columns][0:10])
print(type(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC #Translate model performance into expected benefit to business

# COMMAND ----------

# Here you write script to translate to expected benefits e.g. monetary value

#load claims data
table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"

# Read in dataset & display

check_cols = [
    'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age',
    'C14_contains_watchwords', 'C1_fri_sat_night', 'C2_reporting_delay',
    'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 'C6_no_commuting_but_rush_hour',
    'C7_police_attended_or_crime_reference', 'C9_policy_within_30_days'
]

other_cols = ['svi_risk', 'claim_number', 'reported_date'] + [x for x in check_cols if x not in categorical_features] 
raw_df = spark.table(table_path).withColumn('checks_max', greatest(*[col(c) for c in check_cols]))            

#add max outsourcing fees
svi_perf = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_performance")\
                .select("Outsourcing Fee", "Claim Number").coalesce(1).orderBy("Claim Number", "Outsourcing Fee", ascending=False)\
                .drop_duplicates()

raw_df = raw_df.select(numeric_features+categorical_features+other_cols).toPandas()

raw_df =  set_types(raw_df)

# Load model
y_pred = pipeline.predict(raw_df[numeric_features + categorical_features])
y_prob = pipeline.predict_proba(raw_df[numeric_features + categorical_features])[:,1]
y_test = raw_df['svi_risk']

raw_df['y_pred'] = y_pred
raw_df['y_prob'] = y_prob.round(4)
raw_df['y_test'] = raw_df['svi_risk']

raw_df['y_cmb']  = raw_df[["checks_max", "y_pred"]].astype('int').max(axis=1)
print("using ML only")
print(classification_report(y_test, y_pred))
print("using checks_max and ML")
print(classification_report(y_test, raw_df['checks_max'].astype(int)))

display(raw_df)

display(raw_df[['claim_number']].value_counts().reset_index())

#other_cols.append("Outsourcing Fee",)
#raw_df = raw_df.join(svi_perf, how="left", on=raw_df["claim_number"]==svi_perf["Claim Number"]).select(numeric_features+categorical_features+other_cols)
#FC/901119791, FC/9956681

# COMMAND ----------

from pyspark.sql.functions import array, col, lit, when

raw_df_spark = spark.createDataFrame(raw_df) 

checks_columns = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords",]

raw_df_spark = raw_df_spark.withColumn(
    "checks_list",
    array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in checks_columns])
)

raw_df_spark = raw_df_spark.withColumn(
    "checks_list",
    expr("filter(checks_list, x -> x is not null)")
)

display(raw_df_spark)

# COMMAND ----------

#uncomment to write to ADP aux catalog
'''
raw_df_spark.write \
    .mode("overwrite") \
    .format("delta")#.option("mergeSchema", "true")\
    .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_predictions")
'''

# COMMAND ----------

# MAGIC %md
# MAGIC #Playback to stakeholders

# COMMAND ----------

#Reminder: Powerpoint slides etc to be made and presented to stakeholders
