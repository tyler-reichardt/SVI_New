# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring Module
# MAGIC
# MAGIC Production model monitoring for Single Vehicle Incident (SVI) fraud detection.
# MAGIC This module handles:
# MAGIC - Real-time monitoring of model predictions
# MAGIC - Feature drift detection and alerting
# MAGIC - Model performance tracking over time
# MAGIC - Business metrics dashboards
# MAGIC - Automated reporting and alerting

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# MAGIC %run ./DataPreprocessing

# COMMAND ----------

# MAGIC %run ./FeatureEngineering

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports and Configuration

# COMMAND ----------

# Import required libraries
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
extract_column_transformation_lists("/config_files/training.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# Get current environment
current_env = get_current_environment()
env_config = get_environment_config()

logger.info(f"Running model monitoring in environment: {current_env}")

# MLflow setup
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Monitoring Pipeline

# COMMAND ----------

class SVIModelMonitoring:
    """
    Comprehensive monitoring system for SVI fraud detection models.
    Tracks predictions, features, and model performance over time.
    """
    
    def __init__(self, spark):
        self.spark = spark
        self.env_config = get_environment_config()
        self.current_env = get_current_environment()
        
        # Model names
        self.desk_check_model_name = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_desk_check_lgbm"
        self.interview_model_name = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_interview_lgbm"
        
        # Monitoring tables
        self.predictions_table = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.model_predictions"
        self.feature_stats_table = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.feature_statistics"
        self.performance_metrics_table = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.performance_metrics"
        self.alerts_table = f"{self.env_config['mlstore_catalog']}.single_vehicle_incident_checks.monitoring_alerts"
        
        # Thresholds for monitoring
        self.thresholds = {
            'feature_drift': 0.2,  # KS statistic threshold
            'prediction_shift': 0.1,  # Relative change in prediction distribution
            'performance_degradation': 0.05,  # 5% drop in key metrics
            'null_rate_increase': 0.1,  # 10% increase in null values
            'volume_anomaly': 0.3  # 30% change in prediction volume
        }
        
    def log_predictions(self, predictions_df, model_type='combined'):
        """
        Log model predictions to monitoring table.
        
        Args:
            predictions_df: DataFrame with predictions
            model_type: 'desk_check', 'interview', or 'combined'
        """
        logger.info(f"Logging {model_type} predictions...")
        
        # Add metadata columns
        predictions_with_meta = predictions_df \
            .withColumn("prediction_timestamp", current_timestamp()) \
            .withColumn("model_type", lit(model_type)) \
            .withColumn("environment", lit(self.current_env)) \
            .withColumn("prediction_date", to_date(col("prediction_timestamp")))
        
        # Write to predictions table
        predictions_with_meta.write \
            .mode("append") \
            .partitionBy("prediction_date", "model_type") \
            .saveAsTable(self.predictions_table)
        
        logger.info(f"Logged {predictions_with_meta.count()} predictions")
        
    def calculate_feature_statistics(self, features_df, reference_df=None):
        """
        Calculate feature statistics for drift detection.
        """
        logger.info("Calculating feature statistics...")
        
        # Get numeric columns
        numeric_cols = [f.name for f in features_df.schema.fields 
                       if f.dataType in [IntegerType(), FloatType(), DoubleType()]]
        
        # Calculate current statistics
        current_stats = []
        
        for col_name in numeric_cols:
            if col_name not in ['claim_number', 'policy_number']:
                # Calculate statistics
                stats_df = features_df.select(
                    mean(col(col_name)).alias("mean"),
                    stddev(col(col_name)).alias("stddev"),
                    min(col(col_name)).alias("min"),
                    max(col(col_name)).alias("max"),
                    expr(f"percentile_approx({col_name}, 0.25)").alias("p25"),
                    expr(f"percentile_approx({col_name}, 0.50)").alias("p50"),
                    expr(f"percentile_approx({col_name}, 0.75)").alias("p75"),
                    count(when(col(col_name).isNull(), 1)).alias("null_count"),
                    count(col(col_name)).alias("total_count")
                ).collect()[0]
                
                current_stats.append({
                    'feature_name': col_name,
                    'mean': stats_df['mean'],
                    'stddev': stats_df['stddev'],
                    'min': stats_df['min'],
                    'max': stats_df['max'],
                    'p25': stats_df['p25'],
                    'p50': stats_df['p50'],
                    'p75': stats_df['p75'],
                    'null_rate': stats_df['null_count'] / stats_df['total_count'] if stats_df['total_count'] > 0 else 0,
                    'calculation_timestamp': datetime.now()
                })
        
        # Convert to DataFrame
        stats_spark_df = self.spark.createDataFrame(current_stats)
        
        # Calculate drift if reference data is provided
        if reference_df is not None:
            stats_spark_df = self._calculate_feature_drift(features_df, reference_df, stats_spark_df)
        
        # Save statistics
        stats_spark_df.write \
            .mode("append") \
            .saveAsTable(self.feature_stats_table)
        
        return stats_spark_df
    
    def _calculate_feature_drift(self, current_df, reference_df, stats_df):
        """
        Calculate feature drift using Kolmogorov-Smirnov test.
        """
        drift_results = []
        
        # Get numeric columns
        numeric_cols = [f.name for f in current_df.schema.fields 
                       if f.dataType in [IntegerType(), FloatType(), DoubleType()]]
        
        for col_name in numeric_cols:
            if col_name not in ['claim_number', 'policy_number']:
                # Get non-null values
                current_values = current_df.select(col_name).filter(col(col_name).isNotNull()).toPandas()[col_name].values
                reference_values = reference_df.select(col_name).filter(col(col_name).isNotNull()).toPandas()[col_name].values
                
                if len(current_values) > 0 and len(reference_values) > 0:
                    # Perform KS test
                    ks_statistic, p_value = stats.ks_2samp(current_values, reference_values)
                    
                    drift_results.append({
                        'feature_name': col_name,
                        'ks_statistic': ks_statistic,
                        'p_value': p_value,
                        'is_drifted': ks_statistic > self.thresholds['feature_drift']
                    })
        
        # Add drift results to stats
        drift_df = pd.DataFrame(drift_results)
        stats_pd = stats_df.toPandas()
        stats_with_drift = stats_pd.merge(drift_df, on='feature_name', how='left')
        
        return self.spark.createDataFrame(stats_with_drift)
    
    def track_model_performance(self, predictions_df, actuals_df=None):
        """
        Track model performance metrics over time.
        """
        logger.info("Tracking model performance...")
        
        # If we have actual outcomes, calculate performance metrics
        if actuals_df is not None:
            # Join predictions with actuals
            performance_df = predictions_df.join(
                actuals_df,
                on=['claim_number'],
                how='inner'
            )
            
            # Calculate metrics by date
            daily_metrics = performance_df.groupBy('prediction_date').agg(
                count('*').alias('prediction_count'),
                avg(when(col('final_decision') == 1, 1).otherwise(0)).alias('referral_rate'),
                avg('desk_check_proba').alias('avg_desk_check_score'),
                avg('interview_proba').alias('avg_interview_score')
            )
            
            # If we have actual fraud outcomes
            if 'actual_fraud' in actuals_df.columns:
                metrics_pd = performance_df.toPandas()
                
                # Calculate classification metrics
                accuracy = accuracy_score(metrics_pd['actual_fraud'], metrics_pd['final_decision'])
                precision = precision_score(metrics_pd['actual_fraud'], metrics_pd['final_decision'])
                recall = recall_score(metrics_pd['actual_fraud'], metrics_pd['final_decision'])
                f1 = f1_score(metrics_pd['actual_fraud'], metrics_pd['final_decision'])
                
                # Add to daily metrics
                daily_metrics = daily_metrics.withColumn('accuracy', lit(accuracy)) \
                    .withColumn('precision', lit(precision)) \
                    .withColumn('recall', lit(recall)) \
                    .withColumn('f1_score', lit(f1))
        else:
            # Just track prediction distributions
            daily_metrics = predictions_df.groupBy('prediction_date').agg(
                count('*').alias('prediction_count'),
                avg(when(col('final_decision') == 1, 1).otherwise(0)).alias('referral_rate'),
                avg('desk_check_proba').alias('avg_desk_check_score'),
                avg('interview_proba').alias('avg_interview_score'),
                stddev('desk_check_proba').alias('std_desk_check_score'),
                stddev('interview_proba').alias('std_interview_score')
            )
        
        # Add metadata
        daily_metrics = daily_metrics \
            .withColumn('environment', lit(self.current_env)) \
            .withColumn('calculation_timestamp', current_timestamp())
        
        # Save metrics
        daily_metrics.write \
            .mode("append") \
            .saveAsTable(self.performance_metrics_table)
        
        return daily_metrics
    
    def detect_anomalies(self, current_metrics_df, historical_metrics_df):
        """
        Detect anomalies in model behavior.
        """
        logger.info("Detecting anomalies...")
        
        alerts = []
        
        # Get latest metrics
        latest_metrics = current_metrics_df.orderBy(col('prediction_date').desc()).first()
        
        # Calculate historical statistics
        historical_stats = historical_metrics_df.agg(
            avg('prediction_count').alias('avg_volume'),
            stddev('prediction_count').alias('std_volume'),
            avg('referral_rate').alias('avg_referral_rate'),
            stddev('referral_rate').alias('std_referral_rate'),
            avg('avg_desk_check_score').alias('avg_desk_check_hist'),
            avg('avg_interview_score').alias('avg_interview_hist')
        ).collect()[0]
        
        # Volume anomaly detection
        if abs(latest_metrics['prediction_count'] - historical_stats['avg_volume']) > \
           self.thresholds['volume_anomaly'] * historical_stats['avg_volume']:
            alerts.append({
                'alert_type': 'VOLUME_ANOMALY',
                'severity': 'HIGH',
                'message': f"Prediction volume {latest_metrics['prediction_count']} deviates significantly from average {historical_stats['avg_volume']}",
                'metric_value': latest_metrics['prediction_count'],
                'threshold_value': historical_stats['avg_volume'] * (1 + self.thresholds['volume_anomaly']),
                'alert_timestamp': datetime.now()
            })
        
        # Referral rate shift detection
        if abs(latest_metrics['referral_rate'] - historical_stats['avg_referral_rate']) > \
           2 * historical_stats['std_referral_rate']:
            alerts.append({
                'alert_type': 'REFERRAL_RATE_SHIFT',
                'severity': 'MEDIUM',
                'message': f"Referral rate {latest_metrics['referral_rate']:.3f} is outside normal range",
                'metric_value': latest_metrics['referral_rate'],
                'threshold_value': historical_stats['avg_referral_rate'],
                'alert_timestamp': datetime.now()
            })
        
        # Prediction distribution shift
        if abs(latest_metrics['avg_desk_check_score'] - historical_stats['avg_desk_check_hist']) > \
           self.thresholds['prediction_shift']:
            alerts.append({
                'alert_type': 'PREDICTION_SHIFT',
                'severity': 'MEDIUM',
                'message': f"Desk check score distribution has shifted significantly",
                'metric_value': latest_metrics['avg_desk_check_score'],
                'threshold_value': historical_stats['avg_desk_check_hist'],
                'alert_timestamp': datetime.now()
            })
        
        # Save alerts
        if alerts:
            alerts_df = self.spark.createDataFrame(alerts)
            alerts_df.write.mode("append").saveAsTable(self.alerts_table)
            
        return alerts
    
    def generate_monitoring_dashboard(self, days_back=30):
        """
        Generate comprehensive monitoring dashboard.
        """
        logger.info(f"Generating monitoring dashboard for last {days_back} days...")
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Load recent predictions
        predictions_df = self.spark.table(self.predictions_table) \
            .filter(col('prediction_date') >= lit(start_date.date()))
        
        # Load recent metrics
        metrics_df = self.spark.table(self.performance_metrics_table) \
            .filter(col('prediction_date') >= lit(start_date.date()))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Daily prediction volume
        volume_data = metrics_df.select('prediction_date', 'prediction_count').toPandas()
        axes[0, 0].plot(volume_data['prediction_date'], volume_data['prediction_count'], marker='o')
        axes[0, 0].set_title('Daily Prediction Volume')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Predictions')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Referral rate trend
        referral_data = metrics_df.select('prediction_date', 'referral_rate').toPandas()
        axes[0, 1].plot(referral_data['prediction_date'], referral_data['referral_rate'] * 100, marker='o', color='green')
        axes[0, 1].set_title('Daily Referral Rate')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Referral Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Score distributions
        score_data = metrics_df.select('prediction_date', 'avg_desk_check_score', 'avg_interview_score').toPandas()
        axes[0, 2].plot(score_data['prediction_date'], score_data['avg_desk_check_score'], label='Desk Check', marker='o')
        axes[0, 2].plot(score_data['prediction_date'], score_data['avg_interview_score'], label='Interview', marker='s')
        axes[0, 2].set_title('Average Model Scores')
        axes[0, 2].set_xlabel('Date')
        axes[0, 2].set_ylabel('Average Score')
        axes[0, 2].legend()
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Performance metrics (if available)
        if 'f1_score' in metrics_df.columns:
            perf_data = metrics_df.select('prediction_date', 'precision', 'recall', 'f1_score').toPandas()
            axes[1, 0].plot(perf_data['prediction_date'], perf_data['precision'], label='Precision', marker='o')
            axes[1, 0].plot(perf_data['prediction_date'], perf_data['recall'], label='Recall', marker='s')
            axes[1, 0].plot(perf_data['prediction_date'], perf_data['f1_score'], label='F1 Score', marker='^')
            axes[1, 0].set_title('Model Performance Metrics')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No performance data available', ha='center', va='center')
        
        # 5. Hourly distribution
        hourly_data = predictions_df.groupBy(hour('prediction_timestamp').alias('hour')).count().toPandas()
        axes[1, 1].bar(hourly_data['hour'], hourly_data['count'])
        axes[1, 1].set_title('Predictions by Hour of Day')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Count')
        
        # 6. Alert summary
        try:
            alerts_df = self.spark.table(self.alerts_table) \
                .filter(col('alert_timestamp') >= lit(start_date))
            alert_counts = alerts_df.groupBy('alert_type').count().toPandas()
            
            if not alert_counts.empty:
                axes[1, 2].pie(alert_counts['count'], labels=alert_counts['alert_type'], autopct='%1.1f%%')
                axes[1, 2].set_title('Alert Distribution')
            else:
                axes[1, 2].text(0.5, 0.5, 'No alerts in period', ha='center', va='center')
        except:
            axes[1, 2].text(0.5, 0.5, 'No alert data available', ha='center', va='center')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = f"monitoring_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {dashboard_path}")
        
        # Generate summary statistics
        summary = {
            'monitoring_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'total_predictions': predictions_df.count(),
            'average_daily_volume': metrics_df.agg(avg('prediction_count')).collect()[0][0],
            'average_referral_rate': metrics_df.agg(avg('referral_rate')).collect()[0][0],
            'environment': self.current_env,
            'report_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary, dashboard_path
    
    def create_business_report(self, predictions_df, business_table_name):
        """
        Create business-friendly report table with predictions and features.
        """
        logger.info("Creating business report table...")
        
        # Select relevant columns for business users
        business_report = predictions_df.select(
            # Identifiers
            'claim_number',
            'policy_number',
            'claim_id',
            
            # Dates
            'start_date',
            'reported_date',
            'prediction_timestamp',
            
            # Risk scores and decisions
            'desk_check_proba',
            'interview_proba',
            'desk_check_decision',
            'interview_decision',
            'final_decision',
            
            # Key features for explanation
            'total_checks_failed',
            'damage_score',
            'areas_damaged',
            'vehicle_age',
            'min_claim_driver_age',
            'vehicle_value',
            'reporting_delay_days',
            'incident_cause_group',
            
            # Business flags
            'C1_fri_sat_night',
            'C2_reporting_delay',
            'C3_weekend_monday_report',
            'C4_total_loss',
            'C7_police_crime_ref',
            'C9_new_policy',
            'C11_young_inexperienced',
            'C14_watchwords'
        )
        
        # Add risk categorization
        business_report = business_report.withColumn(
            'risk_category',
            when(col('interview_proba') >= 0.8, 'Very High')
            .when(col('interview_proba') >= 0.6, 'High')
            .when(col('interview_proba') >= 0.4, 'Medium')
            .when(col('interview_proba') >= 0.2, 'Low')
            .otherwise('Very Low')
        )
        
        # Add explanation text
        business_report = business_report.withColumn(
            'risk_explanation',
            concat_ws(', ',
                when(col('C14_watchwords') == 1, lit('Contains fraud watchwords')),
                when(col('C4_total_loss') == 1, lit('Total loss claim')),
                when(col('C2_reporting_delay') == 1, lit('Significant reporting delay')),
                when(col('C11_young_inexperienced') == 1, lit('Young/inexperienced driver')),
                when(col('C7_police_crime_ref') == 1, lit('Police involvement')),
                when(col('total_checks_failed') >= 5, lit('Multiple risk factors'))
            )
        )
        
        # Save to business table
        table_path = get_table_path("single_vehicle_incident_checks", business_table_name, "mlstore")
        
        business_report.write \
            .mode("overwrite") \
            .partitionBy("prediction_date") \
            .saveAsTable(table_path)
        
        logger.info(f"Business report saved to {table_path}")
        
        return business_report

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Monitoring Pipeline

# COMMAND ----------

# Initialize monitor
monitor = SVIModelMonitoring(spark)

# Load recent predictions (example - replace with actual predictions)
# In production, this would come from your serving pipeline
predictions_table = get_table_path("single_vehicle_incident_checks", "model_predictions", "mlstore")

try:
    # Load predictions
    recent_predictions = spark.table(predictions_table) \
        .filter(col('prediction_date') >= date_sub(current_date(), 7))
    
    # Track model performance
    performance_metrics = monitor.track_model_performance(recent_predictions)
    
    # Generate monitoring dashboard
    summary, dashboard_path = monitor.generate_monitoring_dashboard(days_back=30)
    
    print("Monitoring Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
        
except Exception as e:
    logger.warning(f"No prediction data available yet: {e}")
    print("No prediction data available. Run model serving first to generate predictions.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Drift Detection

# COMMAND ----------

# Example of feature drift detection
# Load current and reference data
try:
    # Current features (last 7 days)
    current_features = spark.table(get_table_path("single_vehicle_incident_checks", "svi_features", "mlstore")) \
        .filter(col('dataset') == 'test') \
        .limit(1000)
    
    # Reference features (30 days ago)
    reference_features = spark.table(get_table_path("single_vehicle_incident_checks", "svi_features", "mlstore")) \
        .filter(col('dataset') == 'train') \
        .limit(1000)
    
    # Calculate feature statistics with drift detection
    feature_stats = monitor.calculate_feature_statistics(current_features, reference_features)
    
    # Show drifted features
    drifted_features = feature_stats.filter(col('is_drifted') == True)
    
    if drifted_features.count() > 0:
        print("⚠️ Feature Drift Detected:")
        drifted_features.select('feature_name', 'ks_statistic', 'p_value').show()
    else:
        print("✅ No significant feature drift detected")
        
except Exception as e:
    logger.warning(f"Could not perform drift detection: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Business Report

# COMMAND ----------

# Create business-friendly report table
try:
    # Load predictions with features
    predictions_with_features = spark.table(predictions_table) \
        .filter(col('prediction_date') == current_date())
    
    # Create business report
    business_report = monitor.create_business_report(
        predictions_with_features,
        "svi_predictions_business_report"
    )
    
    # Show sample
    print("Business Report Sample:")
    business_report.select(
        'claim_number',
        'risk_category',
        'interview_proba',
        'final_decision',
        'risk_explanation'
    ).show(10, truncate=False)
    
except Exception as e:
    logger.info(f"Business report will be created when predictions are available: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Automated Alerts

# COMMAND ----------

# Check for anomalies and generate alerts
try:
    # Load historical metrics
    historical_metrics = spark.table(monitor.performance_metrics_table) \
        .filter(col('prediction_date') < date_sub(current_date(), 1))
    
    # Load current metrics
    current_metrics = spark.table(monitor.performance_metrics_table) \
        .filter(col('prediction_date') == current_date())
    
    # Detect anomalies
    alerts = monitor.detect_anomalies(current_metrics, historical_metrics)
    
    if alerts:
        print(f"⚠️ {len(alerts)} alerts generated:")
        for alert in alerts:
            print(f"  - {alert['alert_type']}: {alert['message']}")
    else:
        print("✅ No anomalies detected")
        
except Exception as e:
    logger.info(f"Anomaly detection will run when sufficient data is available: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Summary

# COMMAND ----------

# Generate performance summary
print("""
Model Monitoring Dashboard
==========================

This notebook provides comprehensive monitoring for the SVI fraud detection models:

1. **Prediction Monitoring**: Tracks all model predictions with timestamps and metadata
2. **Feature Drift Detection**: Uses Kolmogorov-Smirnov test to detect distribution shifts
3. **Performance Tracking**: Monitors key metrics like precision, recall, and F1 score
4. **Anomaly Detection**: Alerts on unusual patterns in predictions or performance
5. **Business Reporting**: Creates user-friendly tables for business stakeholders

Key Tables Created:
- {env}.single_vehicle_incident_checks.model_predictions
- {env}.single_vehicle_incident_checks.feature_statistics
- {env}.single_vehicle_incident_checks.performance_metrics
- {env}.single_vehicle_incident_checks.monitoring_alerts
- {env}.single_vehicle_incident_checks.svi_predictions_business_report

The monitoring system automatically:
- Detects feature drift (KS statistic > 0.2)
- Alerts on prediction volume anomalies (>30% change)
- Tracks referral rate shifts (>2 standard deviations)
- Monitors model score distributions
- Generates visual dashboards
""".format(env=current_env))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Schedule this notebook to run daily/hourly
# MAGIC 2. Set up email alerts for critical anomalies
# MAGIC 3. Create Databricks dashboards from monitoring tables
# MAGIC 4. Integrate with model retraining pipeline when drift is detected
