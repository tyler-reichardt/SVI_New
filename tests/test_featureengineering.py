import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, TimestampType, DateType
)
from pyspark.sql.functions import col, lit, when, sum as spark_sum
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Mock the notebooks module structure
import sys
sys.path.append('..')

from functions.feature_engineering import *

@pytest.fixture(scope="session")
def spark_session():
    """Creates a Spark session for all tests in this file."""
    spark = SparkSession.builder.master("local[1]").appName("FeatureEngTests").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def mock_env_config():
    """Provides a mock environment configuration dictionary."""
    return {
        'auxiliary_catalog': 'mock_aux_catalog',
        'mlstore_catalog': 'mock_ml_catalog',
        'adp_catalog': 'mock_adp_catalog'
    }

@pytest.fixture
def feature_engineering(spark_session, mock_env_config):
    """Initializes the FeatureEngineering class with a real Spark session."""
    return FeatureEngineering(spark=spark_session, env_config=mock_env_config)

# --- Tests for Standalone Functions ---

def test_read_and_process_delta(spark_session):
    """Tests the delta reading and processing function with mocking."""
    mock_df = spark_session.createDataFrame([(1, "A"), (2, "B"), (3, "C")], ["id", "value"])
    
    with patch("pyspark.sql.readwriter.DataFrameReader.table") as mock_table:
        mock_table.return_value = mock_df
        
        # Test selecting useful columns
        df_select = read_and_process_delta(spark_session, "fake_path", useful_columns=["id"])
        assert df_select.columns == ["id"]
        
        # Test skiprows
        df_skip = read_and_process_delta(spark_session, "fake_path", skiprows=1)
        assert df_skip.count() == 2
        assert df_skip.collect()[0].id == 2

def test_recast_dtype(spark_session):
    """Tests the data type casting utility function."""
    input_df = spark_session.createDataFrame([("1", "2.0")], ["int_str", "float_str"])
    result_df = recast_dtype(input_df, ["int_str"], "integer")
    assert isinstance(result_df.schema["int_str"].dataType, IntegerType)

# --- Tests for FeatureEngineering Class ---

def test_get_table_path(feature_engineering):
    """Tests the construction of table paths."""
    assert feature_engineering.get_table_path("s", "t", "mlstore") == "mock_ml_catalog.s.t"
    with pytest.raises(ValueError):
        feature_engineering.get_table_path("s", "t", "invalid_type")

#def test_apply_damage_score_calculation(feature_engineering, spark_session):
#    """Tests the damage score and areas damaged calculations."""
#    all_damage_cols = [
#        'boot_opens', 'doors_open', 'front_severity', 'front_bonnet_severity',
#        'front_left_severity', 'front_right_severity', 'left_severity'
#    ]
#    input_data = [("C1", "Minimal", "Medium", "Heavy", "Severe", "Minimal", None)]
#    schema = ["claim_id"] + all_damage_cols
#    input_df = spark_session.createDataFrame(input_data, schema)
#
#    result_df = feature_engineering.apply_damage_score_calculation(input_df)
#    result = result_df.collect()[0]
#
#    # Minimal(2) + Medium(3) + Heavy(4) + Severe(5) + Minimal(2) = 16
#    assert result["damage_score"] == 16
#    assert result["areasDamagedMinimal"] == 2
#    assert result["areasDamagedMedium"] == 1
#    assert result["areasDamagedHeavy"] == 1
#    assert result["areasDamagedSevere"] == 1
#    assert result["areasDamagedTotal"] == 5

def test_create_time_based_features(feature_engineering, spark_session):
    """Tests the creation of time-based flags."""
    input_data = [("C1", datetime(2023, 1, 8)), ("C2", datetime(2023, 1, 9))] # Sunday, Monday
    input_df = spark_session.createDataFrame(input_data, ["claim_id", "latest_event_time"])

    result_df = feature_engineering.create_time_based_features(input_df)
    results = {r.claim_id: r for r in result_df.collect()}

    assert results["C1"]["incident_weekend"] == 1
    assert results["C1"]["reported_monday"] == 0
    assert results["C2"]["incident_weekend"] == 0
    assert results["C2"]["reported_monday"] == 1

def test_calculate_delays(feature_engineering, spark_session):
    """Tests calculation of date difference features."""
    input_data = [("C1", date(2023, 2, 10), date(2023, 1, 1), date(2024, 1, 1))]
    schema = ["claim_id", "start_date", "policy_start_date", "policy_renewal_date"]
    input_df = spark_session.createDataFrame(input_data, schema)
    
    result_df = feature_engineering.calculate_delays(input_df)
    result = result_df.collect()[0]
    
    assert result["inception_to_claim_days"] == 40
    assert result["claim_to_policy_end"] == 325

#def test_create_check_variables(feature_engineering, spark_session):
#    """Tests the creation of C-variables and their summary columns."""
#    input_data = [Row(
#        claim_id="C1", start_date=datetime(2023,1,15, 10), delay_in_reporting=5, 
#        incident_weekend=0, reported_monday=0, total_loss_flag=False, vehicle_use_quote="2", 
#        is_police_attendance=False, is_crime_reference_provided=False, vehicle_unattended=0, 
#        inception_to_claim_days=10, claim_to_policy_end=300, age_at_policy_start_date_1=40, 
#        licence_length_years_1=20, vehicle_value=10000, circumstances="police involved"
#    )]
#    input_df = spark_session.createDataFrame(input_data)
#    
#    result_df = feature_engineering.create_check_variables(input_df)
#    result = result_df.collect()[0]
#    
#    assert result["C2_reporting_delay"] == 1
#    assert result["C14_contains_watchwords"] == 1
#    assert result["num_failed_checks"] == 2
#    assert sorted(result["checks_list"]) == sorted(['C2_reporting_delay', 'C14_contains_watchwords'])

#def test_handle_missing_values(feature_engineering, spark_session):
#    """Tests filling missing values with representative data."""
#    input_data = [(1, None, None, None)]
#    schema = ["id", "damage_score", "C1_fri_sat_night", "inception_to_claim_days"]
#    input_df = spark_session.createDataFrame(input_data, schema)
#    
#    result_df = feature_engineering.handle_missing_values(input_df)
#    result = result_df.collect()[0]
#    
#    assert result["damage_score"] == 0
#    assert result["C1_fri_sat_night"] == 0
#    assert result["inception_to_claim_days"] == 0

def test_create_target_variable(feature_engineering, spark_session):
    """Tests creation of the target variable 'referred_to_tbg'."""
    input_data = [("C1", 1), ("C2", 0), ("C3", -1), ("C4", None)]
    input_df = spark_session.createDataFrame(input_data, ["claim_id", "tbg_risk"])
    
    result_df = feature_engineering.create_target_variable(input_df)
    results = {r.claim_id: r.referred_to_tbg for r in result_df.collect()}

    assert results["C1"] == 1
    assert results["C2"] == 1
    assert results["C3"] == 0
    assert results["C4"] == 0
