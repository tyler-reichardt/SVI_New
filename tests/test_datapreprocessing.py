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