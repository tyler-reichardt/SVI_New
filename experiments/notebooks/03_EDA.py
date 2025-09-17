# Databricks notebook source
# MAGIC %md
# MAGIC ## EDA overview
# MAGIC
# MAGIC This notebook details the EDA for X
# MAGIC
# MAGIC EDA plan:
# MAGIC - Data ingestion
# MAGIC - Data cleansing
# MAGIC - Univariate analysis
# MAGIC - Feature Engineering
# MAGIC - Bivariate analysis
# MAGIC
# MAGIC Throughout the EDA process, generated plots will be logged as artifacts in mlflow and are visible in the Machine Learning Experiments databricks tab.

# COMMAND ----------

'''
!pip install bokeh
%pip install --no-deps /Volumes/prod_shared/libraries/lib/contourpy/1_2_1/contourpy-1.2.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
%pip install /Volumes/prod_shared/libraries/lib/bokeh/3_4_1/bokeh-3.4.1-py3-none-any.whl
%pip install fastparquet
'''

# COMMAND ----------




# COMMAND ----------

# Import Modules
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType, DateType
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.feature import Imputer
from builtins import min as py_min
from pyspark.sql import DataFrame as PySparkDataFrame
import seaborn as sns
import pandas as pd

'''
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, RangeTool, HoverTool
from bokeh.layouts import column
from bokeh.embed import components
from bokeh.models import NumeralTickFormatter
from bokeh.embed import file_html
from bokeh.resources import CDN
'''

import mlflow
mlflow.autolog(disable=True) 

# COMMAND ----------

# Start mlflow run
#mlflow_run_name = ""
#mlflow.start_run(run_name=mlflow_run_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Ingestion

# COMMAND ----------

# columns to fill using mean
mean_fills = [ "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", "veh_age", "business_mileage", "annual_mileage", "incidentHourC", "additional_vehicles_owned_1", "age_at_policy_start_date_1", "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", "max_additional_vehicles_owned", "min_additional_vehicles_owned", "max_age_at_policy_start_date", "min_age_at_policy_start_date", "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", "outstanding_finance_amount", "claim_to_policy_end"]

#boolean or damage columns with neg fills
neg_fills = ["vehicle_unattended","excesses_applied","is_first_party","first_party_confirmed_tp_notified_claim","is_air_ambulance_attendance","is_ambulance_attendance","is_fire_service_attendance","is_police_attendance","veh_age_more_than_10","damageScore","areasDamagedMinimal","areasDamagedMedium","areasDamagedHeavy","areasDamagedSevere","areasDamagedTotal","police_considering_actions","is_crime_reference_provided","ncd_protected_flag","boot_opens","doors_open","multiple_parties_involved",  "is_incident_weekend","is_reported_monday","driver_age_low_1","claim_driver_age_low","licence_low_1", ]

# fills with ones (rules variables, to trigger manual check)
one_fills = ["C1_fri_sat_night","C2_reporting_delay","C3_weekend_incident_reported_monday","C5_is_night_incident","C6_no_commuting_but_rush_hour","C7_police_attended_or_crime_reference","C9_policy_within_30_days", "C10_claim_to_policy_end", "C11_young_or_inexperienced", "C12_expensive_for_driver_age", "C14_contains_watchwords",]

#fill with word 'missing' (categoricals) 
string_cols = [
    'car_group', 'vehicle_overnight_location_id', 'incidentDayOfWeekC', 'incidentMonthC', 
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

other_cols = ['claim_number', 'svi_risk', 'policy_number', 'policy_transaction_id',  'start_date', 'policy_start_date', 'fa_risk', 'fraud_risk', 'tbg_risk']

useful_cols = mean_fills + neg_fills + one_fills + string_cols + other_cols

print(useful_cols) 



# COMMAND ----------

# Define file path
table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"

# Read in dataset & display
raw_df = spark.table(table_path).select(useful_cols)

display(raw_df.limit(100))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data cleaning and preparation - Schema

# COMMAND ----------

# Method to log schema in mlflow
def mflow_log_schema(raw_df, schema_name):
    # schema_name: string
    raw_df_schema_json = raw_df.schema.json()
    schema_file = schema_name
    with open(schema_file, "w") as f:
        f.write(raw_df_schema_json)
    mlflow.log_artifact(schema_file, schema_name)

# COMMAND ----------

# Log original Schema
#mflow_log_schema(raw_df, "Schema before data type recast")

# COMMAND ----------

# Examine the schema to identify columns with incorrect data types
raw_df.printSchema()

# Generate lists of columns to re-cast to different data type

# Function to recast datatypes
def recast_dtype(raw_df, column_list, dtype):
    # raw_df: current pyspark.sql.dataframe.DataFrame
    # column_list: list of columns that you want to cast to the same data type
    # dtype: dtype object from from pyspark.sql.types e.g. DoubleType()

    for column_name in column_list:
        raw_df = raw_df.withColumn(column_name, col(column_name).cast(dtype))
    return raw_df

# COMMAND ----------

# Recast data types & check schema

cols_groups = {
    "float": mean_fills,
    "integer": one_fills + neg_fills,
    "string": string_cols
}

for dtype, column_list in cols_groups.items():
    raw_df = recast_dtype(raw_df, column_list, dtype)

# COMMAND ----------

# Log updated Schema
#mflow_log_schema(raw_df, "Schema after data type recast")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Data cleaning and preparation - outliers

# COMMAND ----------

# Generate data profile - this should begin to answer questions on missing values, data distribution and outliers
#dbutils.data.summarize(raw_df)

# COMMAND ----------

# Minimum INC_TOT values defined as per BC5 https://1stcentral-my.sharepoint.com/:p:/r/personal/kian_azadi_first-central_com/_layouts/15/Doc2.aspx?action=edit&sourcedoc=%7B6418c3fb-dd6c-41c5-a9f7-345077a003b1%7D&wdOrigin=TEAMS-MAGLEV.undefined_ns.rwc&wdExp=TEAMS-TREATMENT&wdhostclicktime=1708418939721&web=1
column_mins_dict = {
    "INC_TOT_AD": 100,
    "INC_TOT_FIRE": 100,
    "INC_TOT_TPPD": 120,
    "INC_TOT_TPI": 200,
    "INC_TOT_THEFT": 20,
    "INC_TOT_WS": 0,
}

# Function to set minmum value
def set_min_value(raw_df, column, value):
    # value: clipping value
    raw_df = raw_df.withColumn(
        column, when(col(column) < value, value).otherwise(col(column))
    )
    return raw_df


# To use BC5 values
#for key, value in column_mins_dict.items():
#    raw_df = set_min_value(raw_df, key, value)

# COMMAND ----------

# Function to clip values above upper bound for a given feature
# Upper bound is determined using standard anomaly detection: Upper bound = Q3 + (IQR * 1.5)

def clip_upper_bound(raw_df, column_name, ignore_zeros=True):
    # ignore_zeros: if True, we ignore zeros when calculating outliers. This is useful for features with a large number of 0 values.
    if ignore_zeros:
        non_zero_df = raw_df.filter(col(column_name) != 0)
        quartiles = non_zero_df.approxQuantile(column_name, [0.25, 0.75], 0)
        Q1, Q3 = quartiles
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR * 1.5)
        raw_df = raw_df.withColumn(
            column_name + "_clipped",
            when(col(column_name) > upper_bound, lit(upper_bound)).otherwise(
                col(column_name)
            ),
        )

    else:
        quartiles = raw_df.approxQuantile(column_name, [0.25, 0.75], 0)
        Q1, Q3 = quartiles
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR * 1.5)
        raw_df = raw_df.withColumn(
            column_name + "_clipped",
            when(col(column_name) > upper_bound, lit(upper_bound)).otherwise(
                col(column_name)
            ),
        )

    return raw_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data cleaning and preparation - Imputation

# COMMAND ----------

# Histogram plotter to aid imputation decisions
def plot_feature_histogram(raw_df, column_name, num_bins, ml_flow=False, y_limit=False):
    # num_bins: number of histogram bins
    # mflow: True = logs histogram as an artifact in mlflow, False = no logging
    # y_limit: Set an upper limit for y-axis to visualise less common bins
    data_plot = raw_df.select(column_name).rdd.flatMap(lambda x: x).collect()
    hist, edges = np.histogram(data_plot, bins=num_bins)
    fig = figure(
        title=f"Histogram of {column_name}",
        tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,reset,save",
    )
    fig.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="navy",
        line_color="white",
    )
    fig.xaxis.axis_label = column_name
    fig.yaxis.axis_label = "Frequency"

    # Set the font size for the axis labels
    fig.xaxis.axis_label_text_font_size = "12pt"
    fig.yaxis.axis_label_text_font_size = "12pt"

    # Set the font size for the tick labels
    fig.xaxis.major_label_text_font_size = "10pt"
    fig.yaxis.major_label_text_font_size = "10pt"

    # Set the number format for the y-axis ticks
    fig.yaxis.formatter = NumeralTickFormatter(format="0,0")

    # Set the maximum y-value limit
    if y_limit != False:
        fig.y_range.end = y_limit

    # Generate HTML content of the plot
    html_content = file_html(fig, CDN, f"{column_name} histogram")

    # display this html
    displayHTML(html_content)

    # Define nam for html file
    html_file_name = (
        f"{column_name}plot_histogram.html"  # or any other short name you prefer
    )

    # Save the HTML content to a file
    with open(html_file_name, "w") as file:
        file.write(html_content)

    # Log the artifact if ml_flow is True
    if ml_flow:
        mlflow.log_artifact(html_file_name, "plots")

# COMMAND ----------

# Log any significant histograms pre imputation

# COMMAND ----------

# Median imputation
# Function to impute Median of given column
def impute_median(raw_df, cols_to_impute):
    imputer = Imputer(inputCols=cols_to_impute, outputCols=cols_to_impute).setStrategy(
        "median"
    )

    raw_df = imputer.fit(raw_df).transform(raw_df)

    return raw_df

# COMMAND ----------

# Mean imputation
# Function to impute Mean of given column
def impute_mean(raw_df, cols_to_impute):
    imputer = Imputer(inputCols=cols_to_impute, outputCols=cols_to_impute).setStrategy(
        "mean"
    )

    raw_df = imputer.fit(raw_df).transform(raw_df)

    return raw_df

# COMMAND ----------

raw_df = impute_mean(raw_df, mean_fills) 

##fillna columns
neg_fills_dict = {x:-1 for x in neg_fills}
one_fills_dict = {x:-1 for x in one_fills}
string_fills_dict = {x:'missing' for x in string_cols}
combined_fills = {**one_fills_dict, **neg_fills_dict, **string_fills_dict} 
print(combined_fills)
raw_df = raw_df.fillna(combined_fills) 

# COMMAND ----------

display(raw_df.select("policyholder_ncd_years"))

# COMMAND ----------

# Log histograms after imputation of anything significant

# COMMAND ----------

# MAGIC %md
# MAGIC ## Brief Summary of data cleansing
# MAGIC
# MAGIC - Steps that were taken
# MAGIC - Any particular issues

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

# Feature Engineering

# COMMAND ----------

# Data profile to summarise engineered features
#dbutils.data.summarize(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate analysis

# COMMAND ----------

# Plot and log histograms for interesting features
#plot_feature_histogram(raw_df, 'veh_age', 5, ml_flow=False, y_limit=False)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Plot histogram for 'licence_length_years_1'
def plot_histogram_seaborn(df, column_name, num_bins=50):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], bins=num_bins, kde=False)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

# "min_licence_length_years", "policyholder_ncd_years","damageScore","vehicle_value", "age_at_policy_start_date_1", "min_claim_driver_age", "licence_length_years_1"
# policy_cover_type_TPFT, C3_weekend_incident_reported_monday, C2_reporting_delay, num__is_reported_monday, incidentDayOfWeekC, incidentHourC, ncd_protected_flag, notification_method

raw_df_pd = raw_df.toPandas()
filter_num =["svi_risk", "min_licence_length_years", "policyholder_ncd_years","damageScore","vehicle_value", "age_at_policy_start_date_1", "min_claim_driver_age", "licence_length_years_1"]
filter_cat =["policy_cover_type", "C3_weekend_incident_reported_monday", "C2_reporting_delay", "is_reported_monday", "incidentDayOfWeekC", "incidentHourC", "ncd_protected_flag", "notification_method"]
for coln in filter_num: 
    plot_histogram_seaborn(raw_df_pd, coln)

# COMMAND ----------

import pandas as pd

def plot_categories(raw_df_pd, column_name, target):
    for column_name in filter_cat:
        # Group by 'policy_cover_type' and calculate the mean of 'svi_risk'
        results = raw_df_pd.groupby(column_name)[target].mean().reset_index()
        results.columns = [column_name, f'mean_{target}']
        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
        results.plot.bar(x=column_name, y=f'mean_{target}', ax=ax1, legend=False)
        sns.set_palette('PRGn')
        # Set the labels and titles
        ax1.set_ylabel('mean fraud rate')
        ax1.set_title('mean fraud rate')
        #results.plot.bar(x=column_name, y=['new', 'previous'], ax=ax2)
        #ax2.set_ylabel('Conversion rate')
        ax1.axhline(y=0, linewidth=0.8, color='orange')
        plt.xticks(rotation=45)
        plt.tight_layout() 

# Display the results
display(plot_categories(raw_df_pd, filter_cat, "svi_risk")) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bivariate analysis

# COMMAND ----------

# Function to plot correlation heatmap
def correlation_heatmap(
    raw_df,
    method,
    filename="",
    filter_columns=False,
    sample_fraction=False,
    figure_size=(150, 8),
    log_mlflow=False,
):
    # method: pearson : standard correlation coefficient, kendall : Kendall Tau correlation coefficient, spearman : Spearman rank correlation
    # filter_columns: list of columns that we want to include in a truncated heatmap
    # sample_fraction: with large datasets, specifying a sample fraction will randomly sample from dataframe
    if isinstance(raw_df, PySparkDataFrame):
        pandas_df = raw_df.toPandas()
    elif isinstance(raw_df, pd.DataFrame):
        pandas_df = raw_df.copy()

    if sample_fraction != False:
        pandas_df = pandas_df.sample(frac=sample_fraction)  # Adjust frac as needed

    full_correlation_matrix = pandas_df[filter_columns].corr(method=method)

    if filter_columns != False:
        filtered_correlation_matrix = full_correlation_matrix.loc[filter_columns]

    plt.figure(figsize=(20, 20))
    sns.heatmap(
        filtered_correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5
    )

    if log_mlflow:
        # Save the plot to a file
        plt.savefig(filename)
        # Log the plot file as an artifact
        mlflow.log_artifact(filename, "plots")

    plt.show()

# "min_licence_length_years", "policyholder_ncd_years","damageScore","vehicle_value", "age_at_policy_start_date_1", "min_claim_driver_age", "licence_length_years_1"
# policy_cover_type_TPFT, C3_weekend_incident_reported_monday, C2_reporting_delay, num__is_reported_monday, incidentDayOfWeekC, incidentHourC, ncd_protected_flag, notification_method

raw_df_pd = raw_df.toPandas()
filter_columns=["svi_risk", "min_licence_length_years", "policyholder_ncd_years","damageScore","vehicle_value", "age_at_policy_start_date_1", "min_claim_driver_age", "licence_length_years_1"]
correlation_heatmap(raw_df_pd , "pearson", filename="", filter_columns=filter_columns, sample_fraction=False, figure_size=(150, 8), log_mlflow=False )

# COMMAND ----------

old_final_features = ['C10_claim_to_policy_end','C2_reporting_delay','C3_weekend_incident_reported_monday','C5_is_night_incident','C9_policy_within_30_days','age_at_policy_start_date_1','annual_mileage','areasDamagedHeavy','areasDamagedMedium','areasDamagedMinimal','areasDamagedSevere','areasDamagedTotal','assessment_category','business_mileage','min_claim_driver_age','claim_driver_age_low','claim_to_policy_end','damageScore','doors_open','first_party_confirmed_tp_notified_claim','front_bonnet_severity','front_severity','impact_speed','impact_speed_range','inception_to_claim','incidentDayOfWeekC','incidentHourC','incidentMonthC','incident_cause','incident_day_of_week','incident_sub_cause','is_crime_reference_provided','is_police_attendance','is_reported_monday','licence_length_years_1','manufacture_yr_claim','max_age_at_policy_start_date','max_cars_in_household','max_licence_length_years','max_years_resident_in_uk','min_age_at_policy_start_date','min_licence_length_years','min_years_resident_in_uk','ncd_protected_flag','notification_method','policy_cover_type','policy_type','policyholder_ncd_years','right_rear_wheel_severity','veh_age','vehicle_overnight_location_id','vehicle_value','voluntary_amount','years_resident_in_uk_1', 'checks_max']

no_corr_cols =['age_at_policy_start_date_1', 'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'driver_age_low_1', 'claim_driver_age_low', 'licence_length_years_1', 'max_licence_length_years', 'licence_low_1', 'years_resident_in_uk_1', 'max_years_resident_in_uk'] + ["fa_risk", "fraud_risk", "tbg_risk"]

final_features = list(set(old_final_features) - set(no_corr_cols))
print(sorted(final_features))

# COMMAND ----------

# Calculate the correlation matrix
raw_df_pd[one_fills] = raw_df_pd[one_fills].astype(int)

correlation_matrix = raw_df_pd.drop(no_corr_cols,axis=1).corr(numeric_only=True)

# Specify the target variable
target_variable = 'svi_risk'

# Get the correlation of all features with the target variable
target_correlation = correlation_matrix[target_variable].abs().sort_values(ascending=False)

# Display the correlation values
display(target_correlation.reset_index().round(4))

plt.figure(figsize=(5, 10))
sns.heatmap(
        target_correlation.to_frame().head(20), annot=True, cmap="coolwarm", linewidths=0.5
    )

#'min_claim_driver_age', 'min_licence_length_years', 'min_years_resident_in_uk'

print([x for x in raw_df_pd.columns if ('age' in x) & ('amage' not in x)])

['age_at_policy_start_date_1', 'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 'veh_age_more_than_10', 'driver_age_low_1', 'claim_driver_age_low']

print([x for x in raw_df_pd.columns if ('resid' in x) ])


# COMMAND ----------

corr_cols = [ "svi_risk", "C2_reporting_delay", "policyholder_ncd_years", "claim_driver_age_low", "licence_low_1", "C11_young_or_inexperienced", "min_licence_length_years", "driver_age_low_1", "min_age_at_policy_start_date", "min_claim_driver_age", "ncd_protected_flag", "licence_length_years_1", "min_years_resident_in_uk", "age_at_policy_start_date_1", "claim_to_policy_end", "max_licence_length_years", "first_party_confirmed_tp_notified_claim", "years_resident_in_uk_1", "C5_is_night_incident", "max_age_at_policy_start_date", "C10_claim_to_policy_end", "max_years_resident_in_uk", "vehicle_value", "voluntary_amount", "max_cars_in_household", "cars_in_household_1", "min_cars_in_household", "is_incident_weekend", "C7_police_attended_or_crime_reference", "is_police_attendance" ]

correlation_matrix = raw_df_pd[corr_cols].corr(numeric_only=True)

# Flatten the correlation matrix
flat_corr_matrix = correlation_matrix.stack().reset_index()
flat_corr_matrix.columns = ['Row', 'Column', 'Value']
flat_corr_matrix["abs_value"] = flat_corr_matrix.Value.abs()
# Display the flat correlation matrix
display(flat_corr_matrix[(flat_corr_matrix.abs_value!=1) & (flat_corr_matrix.abs_value >= 0.5)].sort_values(by="abs_value", ascending=False))

display(correlation_matrix[corr_cols].reset_index().round(2))

plt.figure(figsize=(70, 40))

sns.heatmap(
        correlation_matrix.head(20), annot=True, cmap="coolwarm", linewidths=0.5
    )

# COMMAND ----------



# COMMAND ----------

#Function to plot 4x4 grid of scatter plots for bivariate analysis
#Example output: https://1stcentral-my.sharepoint.com/:i:/g/personal/david_brown1_first-central_com/EcySlb1nKZJJkJTaBGTeqS0Bv3puN8eiQE3klI0-rubRlg?email=David.Brown1%40first-central.com&e=kQGAKM 
def plot_bivariate_analysis(dataframe, x_columns, y_column, figsize=(20, 20), filename = '', log_mlflow = False):
  #dataframe: pandas DataFrame containing the data
  #x_columns: List of columns to be plotted as x
  #y_column: The y variable
  #figsize: Tuple for figure size

  if isinstance(raw_df, PySparkDataFrame):
    pandas_df = raw_df.toPandas()

  
  num_plots = py_min(len(x_columns), 16)
  
  num_rows = -(-num_plots // 4)  
  num_cols = 4 if num_plots > 4 else num_plots  

  fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharey=True)
  axes = axes.flatten()  

  for i, ax in enumerate(axes):
    if i < num_plots:
        sns.scatterplot(x=x_columns[i], y=y_column, data=dataframe, ax=ax)
        # Truncate column names to 40 characters for title and x-axis label
        truncated_col_name = x_columns[i][:40]
        ax.set_title(f'{truncated_col_name} vs {y_column}')
        ax.set_xlabel(truncated_col_name)
        ax.set_ylabel(y_column)
    else:
        ax.set_visible(False)  # Hide any unused subplot axes

    plt.tight_layout()

    if log_mlflow:
      # Save the plot to a file
      plt.savefig(filename)
      # Log the plot file as an artifact
      mlflow.log_artifact(filename, 'plots')

    plt.show()


#plot_bivariate_analysis(raw_df_pd, filter_columns, "svi_risk", figsize=(20, 20), filename = '', log_mlflow = False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA summary
# MAGIC Few line debrief to summarise key findings for someone else
