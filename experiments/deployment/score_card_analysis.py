# Databricks notebook source
from pyspark.sql.functions import *

result_df = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions").filter("model_version is not null and reported_date >= '2025-05-01' and reported_date <= '2025-05-31'").dropDuplicates()


result_df = result_df.withColumn(
    'new_score_card',
    ((col('delay_in_reporting') > 3) | (col('policyholder_ncd_years') < 2) | (~col('incidentHourC').between(5, 22))).cast('int'))

print('Count(high risk)', result_df.filter(col('y_cmb') == 1).count())
print('Count(new score_card)', result_df.filter(col('new_score_card') == 1).count())
print('Count(old score_card)', result_df.filter(col('score_card') == 1).count())
#what percentage of high risk claims are flagged by the score card
high_risk_percentage_new = (result_df.filter((col('new_score_card') == 1) & (col('y_cmb') == 1)).count() / result_df.filter(col('y_cmb') == 1).count()) * 100
print('high_risk_percentage_new: ', high_risk_percentage_new)

high_risk_percentage_old = (result_df.filter((col('score_card') == 1) & (col('y_cmb') == 1)).count() / result_df.filter(col('y_cmb') == 1).count()) * 100
print('high_risk_percentage: ', high_risk_percentage_old)

display(result_df)

# COMMAND ----------


'''
•	For SVIs that occurred in May, how many cases did the data science model flag as high-risk?
•	How many of the cases that were flagged as high-risk by the data science model, were also flagged as high risk by the FNOL scorecard?
•	How many of the cases that were deemed as high-risk by the data science model, were deemed as low-risk by the FNOL scorecard?
•	How many cases were deemed high-risk by the FNOL scorecard, that were deemed low-risk by the data science model?
'''

print('==============Old score card===================')
print('Total SVI Cases (only comprehensive cover): ', result_df.count())
print('All DS High Risk: ', result_df.filter(col('y_cmb') == 1).count())
print('All Flagged by Scorecard: ', result_df.filter(col('score_card') == 1).count())
print('High-risk DS Model and High-risk Scorecard: ', result_df.filter( (col('y_cmb') == 1) & (col('score_card') == 1)).count())
print('High-risk DS Model and Low-risk Scorecard: ', result_df.filter( (col('y_cmb') == 1) & (col('score_card') == 0)).count())
print('High-risk Scorecard and Low-risk DS Model: ', result_df.filter((col('score_card') == 1) & (col('y_cmb') == 0) ).count())

print('\n\n==============New score card===================')
print('Total SVI Cases (only comprehensive cover): ', result_df.count())
print('All DS High Risk: ', result_df.filter(col('y_cmb') == 1).count())
print('All Flagged by Scorecard: ', result_df.filter(col('new_score_card') == 1).count())
print('High-risk DS Model and High-risk Scorecard: ', result_df.filter( (col('y_cmb') == 1) & (col('new_score_card') == 1)).count())
print('High-risk DS Model and Low-risk Scorecard: ', result_df.filter( (col('y_cmb') == 1) & (col('new_score_card') == 0)).count())
print('High-risk Scorecard and Low-risk DS Model: ', result_df.filter((col('new_score_card') == 1) & (col('y_cmb') == 0) ).count())


# COMMAND ----------

total_loss_cases = result_df.filter(col('total_loss_flag') == 1).count()
print(total_loss_cases)
non_total_loss = result_df.filter(col('total_loss_flag') == 0).count()
print(non_total_loss)

# COMMAND ----------

display(result_df.filter( (col('y_cmb') == 1)).groupBy("assessment_category").count())

display(result_df.filter( (col('y_cmb') == 0)).groupBy("assessment_category").count())

# COMMAND ----------

dbutils.notebook.exit("Notebook stopped by user request.")

from pyspark.sql.functions import *

daily_claims_svi = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_claims_svi")
daily_svi_predictions = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions").filter("model_version is not null")

result_df = daily_svi_predictions.alias("predictions").drop("policyholder_ncd_years", "delay_in_reporting").join(
    daily_claims_svi.select("claim_number", "policyholder_ncd_years", "delay_in_reporting").alias("claims"),
    on="claim_number",
    how="left"
).select(
    "predictions.*",
    col("claims.policyholder_ncd_years").alias("policyholder_ncd_years"),
    col("claims.delay_in_reporting").alias("delay_in_reporting")
).fillna({"policyholder_ncd_years" : 0}).drop("model_version").dropDuplicates()

display(result_df)

# COMMAND ----------

result_df = result_df.withColumn(
    'score_card',
    ((col('delay_in_reporting') > 3) | (col('policyholder_ncd_years') < 2)).cast('int'))

#what percentage of high risk claims are flagged by the score card
high_risk_percentage = (result_df.filter((col('score_card') == 1) & (col('y_cmb') == 1)).count() / result_df.filter(col('y_cmb') == 1).count()) * 100
print('high_risk_percentage: ', high_risk_percentage)

#what percentage of score card claims are flagged by y_cmb
score_card_percentage = (result_df.filter((col('score_card') == 1) & (col('y_cmb') == 1)).count() / result_df.filter(col('score_card') == 1).count()) * 100
print('score_card_percentage: ', score_card_percentage)


# COMMAND ----------

result_df = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions").filter("model_version is not null").dropDuplicates()

oldest_reported_date = result_df.agg(min("reported_date")).collect()[0][0]
latest_reported_date = result_df.agg(max("reported_date")).collect()[0][0]

print('Oldest reported_date: ', oldest_reported_date)
print('Latest reported_date: ', latest_reported_date)

# COMMAND ----------



# COMMAND ----------

table_path = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_claims_svi"
raw_df = spark.table(table_path).toPandas()

raw_df['y_cmb'] = np.random.randint(0, 2, raw_df.shape[0])
import numpy as np
# override model predictions where desired

# No commuting on policy and customer travelling between the hours of 12am and 4am
late_night_no_commuting_condition = (raw_df['vehicle_use_quote'].astype('int')== 1) & (raw_df['incidentHourC'].between(1, 4))
raw_df['late_night_no_commuting'] = np.where(late_night_no_commuting_condition, 1, 0)
raw_df['y_cmb'] = np.where(raw_df['late_night_no_commuting']==1, raw_df['y_cmb'], 1)

# Add a column to check if Circumstances contains words to indicate unconsciousness
watch_words = "|".join(["pass out", "passed out", 'blackout', 'black out', 'blacked out',
                        'blacking out', "unconscious", "unconsciousness"])
raw_df['unconscious_flag'] = np.where(raw_df['Circumstances'].str.lower().str.contains(watch_words, na=False), 1, 0)
raw_df['y_cmb'] = np.where(raw_df['unconscious_flag']==1, raw_df['y_cmb'], 1)

#Filter out if reporting delay >30 days or claim is TPFT
raw_df = raw_df[(raw_df['delay_in_reporting'] <= 30) & (raw_df['cover_type'] != 'TPFT')]

display(raw_df[['claim_number', 'start_date', 'incidentHourC', 'vehicle_use_quote', 'late_night_no_commuting', 'unconscious_flag', 'Circumstances', ]])



# COMMAND ----------

claim_version_df = spark.table("prod_adp_certified.claim.claim_version")

for column in ['position_description', 'position_status']:
    distinct_values = claim_version_df.select(column).distinct()
    print(f"Distinct values for {column}:")
    display(distinct_values)

# COMMAND ----------

daily_svi_predictions = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions")

display(daily_svi_predictions)

from pyspark.sql.functions import min, max

reported_dates = daily_svi_predictions.select(min("reported_date").alias("earliest_date"), max("reported_date").alias("latest_date"))
display(reported_dates)

import pandas as pd

start_date = pd.to_datetime("2025-06-01")
end_date = pd.to_datetime("2025-03-01")
number_of_days = (start_date - end_date).days
number_of_days

# COMMAND ----------

test_df = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_predictions")
display(test_df.limit(20))
