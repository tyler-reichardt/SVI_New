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
from pyspark.sql.functions import hour, dayofweek, month, datediff, year, greatest, least

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


# Tests for standalone feature engineering functions

def test_apply_damage_score_calculation(spark_session):
    """Test the apply_damage_score_calculation function"""
    # Create test data
    test_data = [
        ("claim1", 2, 1, 1, 1),  # 2 minimal, 1 medium, 1 severe, 1 heavy
        ("claim2", 1, 0, 2, 0),  # 1 minimal, 0 medium, 2 severe, 0 heavy
    ]
    columns = ["claim_number", "areasDamagedMinimal", "areasDamagedMedium", 
               "areasDamagedSevere", "areasDamagedHeavy"]
    df = spark_session.createDataFrame(test_data, columns)
    
    # Add damage severity columns for damage_sev_max calculation
    df = df.withColumn("front_severity", lit("Severe"))
    df = df.withColumn("rear_severity", lit("Medium"))
    
    result = apply_damage_score_calculation(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check areasDamagedRelative calculation
    # claim1: 2*1 + 1*2 + 1*3 + 1*4 = 2 + 2 + 3 + 4 = 11
    assert result_dict["claim1"]["areasDamagedRelative"] == 11
    
    # claim2: 1*1 + 0*2 + 2*3 + 0*4 = 1 + 0 + 6 + 0 = 7
    assert result_dict["claim2"]["areasDamagedRelative"] == 7
    
    # Check damage_sev_max (should be 4 for Severe)
    assert result_dict["claim1"]["damage_sev_max"] == 4


def test_create_time_based_features(spark_session):
    """Test the create_time_based_features function"""
    # Create test data
    test_data = [
        ("claim1", datetime(2024, 1, 5, 22, 0, 0), date(2024, 1, 8)),  # Friday night, Monday report
        ("claim2", datetime(2024, 1, 7, 14, 0, 0), date(2024, 1, 7)),  # Sunday afternoon, Sunday report
    ]
    columns = ["claim_number", "start_date", "reported_date"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = create_time_based_features(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check day of week features
    assert "incident_day_of_week" in result.columns
    assert "reported_day_of_week" in result.columns
    
    # Check weekend flags
    assert result_dict["claim2"]["incident_weekend"] == 1  # Sunday is weekend
    assert result_dict["claim2"]["is_incident_weekend"] == 1
    
    # Check Monday reporting
    assert result_dict["claim1"]["reported_monday"] == 1
    assert result_dict["claim1"]["is_reported_monday"] == 1
    
    # Check hour features
    assert result_dict["claim1"]["incidentHourC"] == 22
    assert result_dict["claim2"]["incidentHourC"] == 14
    
    # Check night incident
    assert result_dict["claim1"]["night_incident"] == 0  # 22:00 is not night (23-5)
    

def test_calculate_delays(spark_session):
    """Test the calculate_delays function"""
    # Create test data
    test_data = [
        ("claim1", date(2024, 1, 10), date(2024, 1, 8), date(2023, 12, 1), date(2024, 12, 31), 2020),
        ("claim2", date(2024, 6, 15), date(2024, 6, 20), date(2024, 1, 1), date(2024, 12, 31), 2015),
    ]
    columns = ["claim_number", "start_date", "reported_date", "policy_start_date", 
               "policy_renewal_date", "vehicle_year"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = calculate_delays(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check inception to claim delay
    # claim1: 2024-01-10 - 2023-12-01 = 40 days
    assert result_dict["claim1"]["inception_to_claim"] == 40
    assert result_dict["claim1"]["inception_to_claim_days"] == 40
    
    # Check claim to policy end
    # claim1: 2024-12-31 - 2024-01-10 = 356 days
    assert result_dict["claim1"]["claim_to_policy_end"] == 356
    
    # Check delay in reporting
    # claim2: 2024-06-20 - 2024-06-15 = 5 days
    assert result_dict["claim2"]["delay_in_reporting"] == 5
    
    # Check vehicle age
    # claim1: 2024 - 2020 = 4 years
    assert result_dict["claim1"]["veh_age"] == 4
    assert result_dict["claim1"]["manufacture_yr_claim"] == 2020
    assert result_dict["claim1"]["veh_age_more_than_10"] == 0
    
    # claim2: 2024 - 2015 = 9 years
    assert result_dict["claim2"]["veh_age"] == 9
    assert result_dict["claim2"]["veh_age_more_than_10"] == 0


def test_create_driver_features(spark_session):
    """Test the create_driver_features function"""
    # Create test data
    test_data = [
        ("claim1", 23, 22, 2),  # Young driver
        ("claim2", 45, 40, 20),  # Experienced driver
    ]
    columns = ["claim_number", "age_at_policy_start_date_1", "min_claim_driver_age", "licence_length_years_1"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = create_driver_features(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check driver age low flags
    assert result_dict["claim1"]["driver_age_low_1"] == 1  # 23 < 25
    assert result_dict["claim1"]["claim_driver_age_low"] == 1  # 22 < 25
    assert result_dict["claim1"]["licence_low_1"] == 1  # 2 <= 3
    
    assert result_dict["claim2"]["driver_age_low_1"] == 0  # 45 >= 25
    assert result_dict["claim2"]["claim_driver_age_low"] == 0  # 40 >= 25
    assert result_dict["claim2"]["licence_low_1"] == 0  # 20 > 3


def test_create_policy_features(spark_session):
    """Test the create_policy_features function"""
    # Create test data
    test_data = [
        ("claim1", "Social", "DriveableTotalLoss"),
        ("claim2", "Commuting", "Driveable"),
    ]
    columns = ["claim_number", "vehicle_use", "assessment_category"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = create_policy_features(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check vehicle use quote
    assert result_dict["claim1"]["vehicle_use_quote"] == 1  # Not commuting
    assert result_dict["claim2"]["vehicle_use_quote"] == 0  # Commuting
    
    # Check total loss flags
    assert result_dict["claim1"]["total_loss_flag"] == True
    assert result_dict["claim1"]["total_loss_new"] == 1
    
    assert result_dict["claim2"]["total_loss_flag"] == False
    assert result_dict["claim2"]["total_loss_new"] == 0


def test_handle_missing_values_fe(spark_session):
    """Test the handle_missing_values_fe function"""
    # Create test data with missing values
    test_data = [
        ("claim1", 10.0, 1, 1),
        ("claim2", None, None, None),
    ]
    columns = ["claim_number", "damage_score", "incident_weekend", "driver_age_low_1"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = handle_missing_values_fe(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check that missing values are filled
    assert result_dict["claim2"]["damage_score"] == 0  # Numeric fill with 0
    assert result_dict["claim2"]["incident_weekend"] == 0  # Boolean fill with 0
    assert result_dict["claim2"]["driver_age_low_1"] == 0  # Boolean fill with 0
    
    # Check that existing values are preserved
    assert result_dict["claim1"]["damage_score"] == 10.0
    assert result_dict["claim1"]["incident_weekend"] == 1
    assert result_dict["claim1"]["driver_age_low_1"] == 1


def test_create_aggregated_features(spark_session):
    """Test the create_aggregated_features function"""
    # Create test data with multiple driver columns
    test_data = [
        ("claim1", 2, 30, 1, 10),
        ("claim2", 0, 45, 2, 20),
    ]
    columns = ["claim_number", "additional_vehicles_owned_1", "age_at_policy_start_date_1",
               "additional_vehicles_owned_2", "age_at_policy_start_date_2"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = create_aggregated_features(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check that max/min aggregations are created
    assert result_dict["claim1"]["max_additional_vehicles_owned"] == 2  # max(2, 1) = 2
    assert result_dict["claim1"]["min_additional_vehicles_owned"] == 1  # min(2, 1) = 1
    
    assert result_dict["claim1"]["max_age_at_policy_start_date"] == 30  # max(30, 10) = 30
    assert result_dict["claim1"]["min_age_at_policy_start_date"] == 10  # min(30, 10) = 10


def test_select_modeling_features(spark_session):
    """Test the select_modeling_features function"""
    # Create test data with various feature types
    test_data = [
        ("claim1", 1, 0, 10.0, "Driveable", 1, "High"),
        ("claim2", 0, 1, 20.0, "TotalLoss", 0, "Low"),
    ]
    columns = ["claim_number", "svi_risk", "tbg_risk", "damage_score", 
               "assessment_category", "C1_fri_sat_night", "vehicle_use_quote"]
    df = spark_session.createDataFrame(test_data, columns)
    
    # Add dataset column for partitioning
    df = df.withColumn("dataset", lit("train"))
    
    result = select_modeling_features(df)
    
    # Check that key columns are selected
    assert "claim_number" in result.columns
    assert "svi_risk" in result.columns
    assert "damage_score" in result.columns
    assert "assessment_category" in result.columns
    assert "C1_fri_sat_night" in result.columns
    
    # Check that only existing columns are selected
    assert result.count() == 2
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
