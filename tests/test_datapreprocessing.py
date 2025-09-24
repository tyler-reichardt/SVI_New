import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, TimestampType, DateType, BooleanType
)
from pyspark.sql.functions import col, lit, when
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
from unittest.mock import patch, MagicMock, call

# Mock the notebooks module structure
import sys
sys.path.append('..')

from functions.data_processing import *


@pytest.fixture(scope="session")
def spark_session():
    """Creates a Spark session for all tests in this file."""
    spark = SparkSession.builder.master("local[1]").appName("DataPrepTests").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def mock_env_config():
    """Provides a mock environment configuration dictionary."""
    return {
        'mlstore_catalog': 'mock_ml_catalog',
        'auxiliary_catalog': 'mock_aux_catalog',
        'adp_catalog': 'mock_adp_catalog'
    }

@pytest.fixture
def data_preprocessor(spark_session, mock_env_config):
    """Fixture for the DataPreprocessing class."""
    # We pass a real spark session, but will mock its methods (like .sql) inside tests
    return DataPreprocessing(spark=spark_session, env_config=mock_env_config)

# --- Tests ---

def test_get_table_path(data_preprocessor):
    """Tests the construction of table paths."""
    assert data_preprocessor.get_table_path("s", "t", "auxiliary") == "mock_aux_catalog.s.t"
    with pytest.raises(ValueError):
        data_preprocessor.get_table_path("s", "t", "invalid")

#def test_load_claim_referral_log(data_preprocessor, spark_session):
#    """Tests loading claim referral log with a mocked spark.sql call."""
#    mock_data = [Row(
#        **{"Claim_Ref": "FC/123*", "Date_received": date(2023, 5, 10), "Final_Outcome_of_Claim": "Repudiated - Not challenged", "Outcome_of_Referral": "X", "Outcome_of_investigation": "Repudiated"}
#    )]
#    mock_df = spark_session.createDataFrame(mock_data)
#    
#    # Patch the spark.sql method on the instance to return our mock DataFrame
#    with patch.object(data_preprocessor.spark, 'sql', return_value=mock_df) as mock_sql:
#        result_df = data_preprocessor.load_claim_referral_log()
#        
#        mock_sql.assert_called_once()
#        assert result_df.count() == 1
#        assert result_df.collect()[0]['id'] == "FC/123"

def test_deduplicate_driver_data(data_preprocessor, spark_session):
    """Tests the driver data deduplication logic."""
    input_data = [
        ("C1", 25, "D1", "1998-01-01"),
        ("C1", 30, "D2", "1993-01-01"), # Same claim, different driver
        ("C2", 40, "D3", "1983-01-01"),
    ]
    schema = ["claim_number", "claim_driver_age", "driver_id", "claim_driver_dob"]
    input_df = spark_session.createDataFrame(input_data, schema)

    result_df = data_preprocessor.deduplicate_driver_data(input_df)

    assert result_df.count() == 2
    assert "min_claim_driver_age" in result_df.columns
    assert "claim_driver_age" not in result_df.columns
    result_data = {r.claim_number: r.min_claim_driver_age for r in result_df.collect()}
    assert result_data["C1"] == 25

def test_join_claim_and_policy_data(data_preprocessor, spark_session):
    """Tests the join logic and filtering for matched policies."""
    claim_data = [("P1", "C1"), ("P2", "C2"), ("P3", "C3")]
    claim_df = spark_session.createDataFrame(claim_data, ["policy_number", "claim_number"])

    policy_data = [("P1", "T1"), ("P2", "T2")] # Policy P3 is missing
    policy_df = spark_session.createDataFrame(policy_data, ["policy_number", "policy_transaction_id"])
    
    result_df = data_preprocessor.join_claim_and_policy_data(claim_df, policy_df)
    
    # Only claims with a matched policy should remain
    assert result_df.count() == 2
    assert "policy_transaction_id" in result_df.columns
    assert result_df.filter(col("claim_number") == "C3").count() == 0

def test_create_train_test_split(data_preprocessor, spark_session):
    """Tests the train/test split logic."""
    data = [("id_1", 0, "2024-01-01")] * 8 + [("id_2", 1, "2024-01-01")] * 2
    pdf = pd.DataFrame(data, columns=["claim_number", "svi_risk", "reported_date"])
    pdf["reported_date"] = pd.to_datetime(pdf["reported_date"])
    input_df = spark_session.createDataFrame(pdf)

    result_df = data_preprocessor.create_train_test_split(input_df)

    assert "dataset" in result_df.columns
    assert result_df.count() == 10
    assert result_df.filter("dataset = 'test'").count() == 3
    assert result_df.filter("dataset = 'train'").count() == 7

def test_fill_missing_values(data_preprocessor, spark_session):
    """Tests the logic for filling missing values with different strategies."""
    schema = spark_session.createDataFrame(
        [("a", 1.0, 1, 1, "x")], 
        ["id", "policyholder_ncd_years", "C1_fri_sat_night", "vehicle_unattended", "car_group"]
    ).schema

    input_data = [
        ("a", 10.0, 1, 1, "x"),
        ("b", 20.0, None, None, None),
        ("c", None, 0, 0, "y"),
    ]
    input_df = spark_session.createDataFrame(input_data, schema)
    
    result_df = data_preprocessor.fill_missing_values(input_df)
    results = {r.id: r for r in result_df.collect()}

    # Mean of 'policyholder_ncd_years' is (10+20)/2 = 15.0
    assert results["c"]["policyholder_ncd_years"] == 15.0
    # Rule variables fill with 1
    assert results["b"]["C1_fri_sat_night"] == 1
    # Boolean/flag variables fill with -1
    assert results["b"]["vehicle_unattended"] == -1
    # String variables fill with 'missing'
    assert results["b"]["car_group"] == "missing"
    # Existing values should be untouched
    assert results["a"]["policyholder_ncd_years"] == 10.0


# Tests for standalone functions

def test_get_referral_vertices(spark_session):
    """Test the get_referral_vertices function"""
    from functions.data_processing import get_referral_vertices
    
    # Create test data
    test_data = [
        ("FC/123*", "Repudiated - Not challenged", "X", "Repudiated"),
        ("FC/456", "Settled", "Y", "Repudiated"),
        ("FC/789", "Settled", "Z", "Settled")
    ]
    columns = ["Claim Ref", "Final_Outcome_of_Claim", "Outcome_of_Referral", "Outcome_of_investigation"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = get_referral_vertices(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check that claim references are cleaned
    assert "FC/123" in result_dict
    assert "FC/456" in result_dict
    
    # Check risk indicators
    assert result_dict["FC/123"]["tbg_risk"] == 1
    assert result_dict["FC/123"]["fa_risk"] == 1
    assert result_dict["FC/123"]["svi_risk"] == 1
    
    assert result_dict["FC/456"]["tbg_risk"] == 1  # Settled but investigation was Repudiated
    assert result_dict["FC/789"]["tbg_risk"] == 0  # Settled and investigation was Settled


def test_calculate_damage_score(spark_session):
    """Test the calculate_damage_score function"""
    from functions.data_processing import calculate_damage_score
    
    # Create test data with damage severity columns
    test_data = [
        ("claim1", "Minimal", "Medium", "Heavy", "Severe"),
        ("claim2", "Minimal", "Minimal", None, None),
        ("claim3", None, None, None, None)
    ]
    columns = ["claim_number", "front_severity", "rear_severity", "left_severity", "right_severity"]
    
    # Add remaining damage columns as None
    damage_cols = [
        'boot_opens', 'doors_open', 'front_bonnet_severity',
        'front_left_severity', 'front_right_severity',
        'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity',
        'left_rear_wheel_severity', 'left_underside_severity',
        'rear_left_severity', 'rear_right_severity',
        'rear_window_damage_severity', 'right_back_seat_severity',
        'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity',
        'right_roof_severity', 'right_underside_severity', 'roof_damage_severity',
        'underbody_damage_severity', 'windscreen_damage_severity'
    ]
    
    # Create DataFrame with all damage columns
    df = spark_session.createDataFrame(test_data, columns[:5])
    for col in damage_cols:
        df = df.withColumn(col, lit(None))
    
    result = calculate_damage_score(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check damage scores
    # claim1: Minimal(2) + Medium(3) + Heavy(4) + Severe(5) = 14
    assert result_dict["claim1"]["damage_score"] == 14
    assert result_dict["claim1"]["damageScore"] == 14  # Check alias
    
    # claim2: Minimal(2) + Minimal(2) = 4
    assert result_dict["claim2"]["damage_score"] == 4
    
    # claim3: All None = 0
    assert result_dict["claim3"]["damage_score"] == 0
    
    # Check area counts
    assert result_dict["claim1"]["areasDamagedMinimal"] == 1
    assert result_dict["claim1"]["areasDamagedMedium"] == 1
    assert result_dict["claim1"]["areasDamagedHeavy"] == 1
    assert result_dict["claim1"]["areasDamagedSevere"] == 1
    assert result_dict["claim1"]["areasDamagedTotal"] == 4


def test_create_check_variables(spark_session):
    """Test the create_check_variables function"""
    from functions.data_processing import create_check_variables
    from datetime import datetime, date
    
    # Create test data
    test_data = [
        ("claim1", datetime(2024, 1, 5, 22, 0, 0), date(2024, 1, 8), 3, 1, True, 1, 25, 20000),
        ("claim2", datetime(2024, 1, 1, 12, 0, 0), date(2024, 1, 1), 0, 0, False, 0, 30, 35000)
    ]
    columns = ["claim_number", "start_date", "reported_date", "delay_in_reporting", 
               "incident_weekend", "total_loss_flag", "vehicle_use_quote",
               "inception_to_claim_days", "vehicle_value"]
    
    df = spark_session.createDataFrame(test_data, columns)
    
    # Add required columns for checks
    df = df.withColumn("reported_monday", lit(0))
    df = df.withColumn("is_police_attendance", lit(False))
    df = df.withColumn("is_crime_reference_provided", lit(False))
    df = df.withColumn("vehicle_unattended", lit(0))
    df = df.withColumn("claim_to_policy_end", lit(100))
    df = df.withColumn("age_at_policy_start_date_1", lit(30))
    df = df.withColumn("licence_length_years_1", lit(5))
    df = df.withColumn("assessment_category", lit("Driveable"))
    df = df.withColumn("Circumstances", lit("Normal accident"))
    
    result = create_check_variables(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check C2: Reporting delay >= 3 days
    assert result_dict["claim1"]["C2_reporting_delay"] == 1
    assert result_dict["claim2"]["C2_reporting_delay"] == 0
    
    # Check C5: Night incident (22:00 is night)
    assert result_dict["claim1"]["C5_is_night_incident"] == 0  # 22:00 is not in 23-5 range
    
    # Check C9: Policy inception within 30 days
    assert result_dict["claim1"]["C9_policy_within_30_days"] == 1  # 25 days
    
    # Check num_failed_checks
    assert result_dict["claim1"]["num_failed_checks"] > 0


def test_daily_pipeline_functions(spark_session, mock_env_config):
    """Test daily pipeline data retrieval functions"""
    from functions.data_processing import (
        get_latest_policy_transactions, get_policy_data,
        get_vehicle_data, get_excess_data, get_driver_data
    )
    
    # Create mock table for testing
    test_data = [
        ("P001", "2024-01-01", "policy1"),
        ("P002", "2024-01-01", "policy2"),
        ("P003", "2024-01-02", "policy3")
    ]
    columns = ["policy_number", "transaction_date", "policy_id"]
    df = spark_session.createDataFrame(test_data, columns)
    df.createOrReplaceTempView("test_policies")
    
    # Test get_latest_policy_transactions
    result = get_latest_policy_transactions(spark_session, "test_policies", "2024-01-01")
    assert result.count() == 2
    
    # Test get_policy_data
    result = get_policy_data(spark_session, "test_policies", ["P001", "P002"])
    assert result.count() == 2


def test_dedup_driver_features(spark_session):
    """Test the dedup_driver_features function"""
    from functions.data_processing import dedup_driver_features
    
    # Create test data with multiple drivers per claim
    test_data = [
        ("claim1", 25, "1999-01-01", "driver1"),
        ("claim1", 30, "1994-01-01", "driver2"),
        ("claim2", 40, "1984-01-01", "driver3")
    ]
    columns = ["claim_number", "claim_driver_age", "claim_driver_dob", "driver_id"]
    df = spark_session.createDataFrame(test_data, columns)
    
    result = dedup_driver_features(df)
    result_dict = {r["claim_number"]: r for r in result.collect()}
    
    # Check that minimum age is selected for claim1
    assert result_dict["claim1"]["min_claim_driver_age"] == 25
    assert result_dict["claim2"]["min_claim_driver_age"] == 40
    
    # Check that original columns are removed
    assert "claim_driver_age" not in result.columns


def test_aggregate_driver_columns(spark_session):
    """Test the aggregate_driver_columns function"""
    from functions.data_processing import aggregate_driver_columns
    
    # Create test data with driver columns
    test_data = [
        ("claim1", 2, 30, 1, 10, 25),
        ("claim2", 0, 45, 2, 20, 40)
    ]
    columns = ["claim_number", "additional_vehicles_owned_1", "age_at_policy_start_date_1",
               "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1"]
    df = spark_session.createDataFrame(test_data, columns)
    
    # Add _2 columns for testing
    df = df.withColumn("additional_vehicles_owned_2", lit(1))
    df = df.withColumn("age_at_policy_start_date_2", lit(35))
    
    result = aggregate_driver_columns(df)
    
    # Check that aggregation columns are created
    assert "min_additional_vehicles_owned" in result.columns
    assert "max_additional_vehicles_owned" in result.columns