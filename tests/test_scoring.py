import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, array, expr

# Import the functions to test
from functions.scoring import (
    load_claims_data,
    do_fills_pd_scoring,
    apply_business_rules,
    score_claims_batch,
    save_predictions,
    load_models_from_mlflow,
    get_feature_lists,
    get_mean_dict,
    print_scoring_summary
)


@pytest.fixture(scope="session")
def spark_session():
    """Creates a Spark session for all tests in this file."""
    spark = SparkSession.builder.master("local[1]").appName("ScoringTests").getOrCreate()
    yield spark
    spark.stop()


def test_do_fills_pd_scoring():
    """Test the do_fills_pd_scoring function"""
    # Create test data with missing values
    test_data = pd.DataFrame({
        'policyholder_ncd_years': [5.0, np.nan, 10.0],
        'damageScore': [10, np.nan, 20],
        'C1_fri_sat_night': [1, np.nan, 0],
        'vehicle_unattended': [0, np.nan, 1],
        'assessment_category': ['Driveable', np.nan, 'TotalLoss'],
        'num_failed_checks': [2, np.nan, 0]
    })
    
    mean_dict = {'policyholder_ncd_years': 7.5, 'num_failed_checks': 1}
    
    result = do_fills_pd_scoring(test_data, mean_dict)
    
    # Check that missing values are filled correctly
    assert result['policyholder_ncd_years'].iloc[1] == 7.5  # Filled with mean
    assert result['damageScore'].iloc[1] == -1  # Damage cols filled with -1
    assert result['C1_fri_sat_night'].iloc[1] == '1'  # Rule variable filled with 1 and converted to string
    assert result['vehicle_unattended'].iloc[1] == '-1'  # Boolean filled with -1 and converted to string
    assert result['assessment_category'].iloc[1] == 'missing'  # String filled with 'missing'
    assert result['num_failed_checks'].iloc[1] == 1  # Filled with mean
    
    # Check that existing values are preserved
    assert result['policyholder_ncd_years'].iloc[0] == 5.0
    assert result['damageScore'].iloc[0] == 10.0


def test_apply_business_rules():
    """Test the apply_business_rules function"""
    # Create test data
    test_data = pd.DataFrame({
        'fa_pred': [0.6, 0.3, 0.8],
        'y_prob2': [0.7, 0.2, 0.9],
        'num_failed_checks': [2, 0, 1],
        'vehicle_use_quote': [1, 0, 1],
        'incidentHourC': [2, 10, 3],
        'Circumstances': ['Normal accident', 'Driver passed out', 'No issues']
    })
    
    result = apply_business_rules(test_data, fa_threshold=0.5, interview_threshold=0.5)
    
    # Check model predictions
    assert result['y_pred'].iloc[0] == 1  # fa_pred > 0.5
    assert result['y_pred'].iloc[1] == 0  # fa_pred < 0.5
    
    assert result['y_pred2'].iloc[0] == 1  # y_prob2 > 0.5
    assert result['y_pred2'].iloc[1] == 0  # y_prob2 < 0.5
    
    # Check combined prediction (both models must flag AND at least one check must fail)
    assert result['y_cmb'].iloc[0] == 1  # Both models flagged and num_failed_checks > 0
    assert result['y_cmb'].iloc[1] == 0  # Model didn't flag
    assert result['y_cmb'].iloc[2] == 1  # Both models flagged and num_failed_checks > 0
    
    # Check late night no commuting override
    assert result['late_night_no_commuting'].iloc[0] == 1  # vehicle_use_quote=1 and hour between 1-4
    assert result['late_night_no_commuting'].iloc[1] == 0
    
    # Check unconscious flag
    assert result['unconscious_flag'].iloc[1] == 1  # "passed out" in circumstances
    assert result['unconscious_flag'].iloc[0] == 0
    
    # Check labels and scores
    assert 'y_cmb_label' in result.columns
    assert 'y_rank_prob' in result.columns
    assert 'model_version' in result.columns


def test_score_claims_batch(spark_session):
    """Test the score_claims_batch function"""
    # Create test data
    test_data = pd.DataFrame({
        'policyholder_ncd_years': [5.0, 10.0],
        'min_claim_driver_age': [30, 45],
        'assessment_category': ['Driveable', 'TotalLoss'],
        'C1_fri_sat_night': [0, 1],
        'voluntary_amount': [100, 200],
        'left_severity': ['Minimal', 'Heavy'],
        'C9_policy_within_30_days': [0, 1],
        'checks_max': [0, 1],
        'total_loss_flag': [False, True],
        'num_failed_checks': [1, 2],
        'vehicle_use_quote': [0, 1],
        'incidentHourC': [10, 2],
        'Circumstances': ['Normal', 'Issue']
    })
    
    # Add all required columns with dummy values
    numeric_features = ['policyholder_ncd_years', 'min_claim_driver_age', 'voluntary_amount']
    categorical_features = ['assessment_category', 'C1_fri_sat_night', 'left_severity', 
                           'C9_policy_within_30_days', 'checks_max', 'total_loss_flag']
    num_interview = ['voluntary_amount', 'policyholder_ncd_years']
    cat_interview = ['assessment_category', 'left_severity', 'C9_policy_within_30_days', 
                     'checks_max', 'total_loss_flag']
    
    # Create mock models
    mock_fa_model = MagicMock()
    mock_fa_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
    
    mock_interview_model = MagicMock()
    mock_interview_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
    
    mean_dict = {'policyholder_ncd_years': 7.5, 'min_claim_driver_age': 35, 'voluntary_amount': 150}
    
    result = score_claims_batch(
        spark_session, test_data, mock_fa_model, mock_interview_model,
        numeric_features, categorical_features,
        num_interview, cat_interview, mean_dict
    )
    
    assert 'fa_pred' in result.columns
    assert 'y_prob2' in result.columns
    assert 'y_cmb' in result.columns
    assert len(result) == 2


def test_save_predictions(spark_session):
    """Test the save_predictions function"""
    # Create test predictions
    predictions_df = pd.DataFrame({
        'claim_number': ['C1', 'C2'],
        'fa_pred': [0.7, 0.3],
        'y_prob2': [0.8, 0.2],
        'y_cmb': [1, 0],
        'flagged_by_model': [1, 0],
        'unconscious_flag': [0, 0],
        'late_night_no_commuting': [0, 0],
        'vehicle_use_quote': [1.0, 0.0]
    })
    
    with patch.object(spark_session, 'createDataFrame') as mock_create:
        mock_spark_df = MagicMock()
        mock_create.return_value = mock_spark_df
        mock_spark_df.withColumn.return_value = mock_spark_df
        mock_spark_df.write.format.return_value.option.return_value.mode.return_value.saveAsTable = MagicMock()
        mock_spark_df.count.return_value = 2
        
        count = save_predictions(spark_session, predictions_df, 'test_table')
        
        assert count == 2
        mock_create.assert_called_once()


def test_load_models_from_mlflow():
    """Test the load_models_from_mlflow function"""
    with patch('mlflow.sklearn.load_model') as mock_load:
        mock_fa_model = MagicMock()
        mock_interview_model = MagicMock()
        mock_load.side_effect = [mock_fa_model, mock_interview_model]
        
        fa_model, interview_model = load_models_from_mlflow('run_id_1', 'run_id_2')
        
        assert fa_model == mock_fa_model
        assert interview_model == mock_interview_model
        assert mock_load.call_count == 2
        mock_load.assert_any_call('runs:/run_id_1/model')
        mock_load.assert_any_call('runs:/run_id_2/model')


def test_get_feature_lists():
    """Test the get_feature_lists function"""
    numeric_features, categorical_features, num_interview, cat_interview = get_feature_lists()
    
    # Check that feature lists are returned
    assert isinstance(numeric_features, list)
    assert isinstance(categorical_features, list)
    assert isinstance(num_interview, list)
    assert isinstance(cat_interview, list)
    
    # Check some expected features
    assert 'policyholder_ncd_years' in numeric_features
    assert 'assessment_category' in categorical_features
    assert 'voluntary_amount' in num_interview
    assert 'total_loss_flag' in cat_interview
    
    # Check list sizes are reasonable
    assert len(numeric_features) > 10
    assert len(categorical_features) > 20
    assert len(num_interview) > 10
    assert len(cat_interview) > 5


def test_get_mean_dict():
    """Test the get_mean_dict function"""
    mean_dict = get_mean_dict()
    
    # Check that dictionary is returned
    assert isinstance(mean_dict, dict)
    
    # Check some expected keys
    assert 'policyholder_ncd_years' in mean_dict
    assert 'min_claim_driver_age' in mean_dict
    assert 'vehicle_value' in mean_dict
    assert 'num_failed_checks' in mean_dict
    
    # Check that values are numeric
    assert isinstance(mean_dict['policyholder_ncd_years'], (int, float))
    assert mean_dict['policyholder_ncd_years'] > 0
    
    # Check dictionary size
    assert len(mean_dict) > 20


def test_print_scoring_summary(capsys):
    """Test the print_scoring_summary function"""
    # Create test predictions
    predictions_df = pd.DataFrame({
        'y_cmb': [1, 0, 1, 0, 0],
        'flagged_by_model': [1, 0, 1, 0, 0],
        'late_night_no_commuting': [0, 0, 0, 1, 0],
        'unconscious_flag': [0, 0, 1, 0, 0],
        'fa_pred': [0.7, 0.3, 0.8, 0.4, 0.2],
        'y_prob2': [0.6, 0.2, 0.9, 0.3, 0.1],
        'y_rank_prob': [0.65, 0.25, 0.85, 0.35, 0.15]
    })
    
    this_day = '2024-01-15'
    
    print_scoring_summary(predictions_df, this_day)
    
    # Capture printed output
    captured = capsys.readouterr()
    
    # Check that summary contains expected information
    assert 'SCORING SUMMARY' in captured.out
    assert '2024-01-15' in captured.out
    assert 'Total claims scored: 5' in captured.out
    assert 'High risk claims: 2' in captured.out
    assert '40.00%' in captured.out  # 2 out of 5
    assert 'Flagged by model: 2' in captured.out
    assert 'Late night no commuting: 1' in captured.out
    assert 'Unconscious flag: 1' in captured.out
    assert 'FA model:' in captured.out
    assert 'Interview model:' in captured.out
    assert 'Combined rank:' in captured.out


def test_apply_business_rules_edge_cases():
    """Test apply_business_rules with edge cases"""
    # Test with all NaN predictions
    test_data = pd.DataFrame({
        'fa_pred': [np.nan, np.nan],
        'y_prob2': [np.nan, np.nan],
        'num_failed_checks': [0, 1],
        'vehicle_use_quote': [np.nan, np.nan],
        'incidentHourC': [np.nan, np.nan],
        'Circumstances': [np.nan, '']
    })
    
    result = apply_business_rules(test_data)
    
    # Check that function handles NaN gracefully
    assert 'y_cmb' in result.columns
    assert 'y_rank_prob' in result.columns
    
    # y_rank_prob should handle NaN fa_pred and y_prob2
    # sqrt(100 * 100) = 100 when both are NaN (filled with 100)
    assert result['y_rank_prob'].iloc[0] == 100.0
    
    # Check that unconscious_flag handles NaN/empty circumstances
    assert result['unconscious_flag'].iloc[0] == 0
    assert result['unconscious_flag'].iloc[1] == 0


def test_do_fills_pd_scoring_comprehensive():
    """Test do_fills_pd_scoring with comprehensive column coverage"""
    # Create test data with all column types
    test_data = pd.DataFrame({
        # Mean fills
        'policyholder_ncd_years': [np.nan],
        'inception_to_claim': [np.nan],
        'min_claim_driver_age': [np.nan],
        # Damage cols
        'damageScore': [np.nan],
        'areasDamagedMinimal': [np.nan],
        # Bool cols
        'vehicle_unattended': [np.nan],
        'is_first_party': [np.nan],
        'total_loss_flag': [np.nan],
        # One fills (rule variables)
        'C1_fri_sat_night': [np.nan],
        'C2_reporting_delay': [np.nan],
        # String cols
        'assessment_category': [np.nan],
        'incident_type': [np.nan],
        'vehicle_overnight_location_id': [np.nan]
    })
    
    mean_dict = {
        'policyholder_ncd_years': 5.0,
        'inception_to_claim': 100.0,
        'min_claim_driver_age': 35.0
    }
    
    result = do_fills_pd_scoring(test_data, mean_dict)
    
    # Check mean fills
    assert result['policyholder_ncd_years'].iloc[0] == 5.0
    assert result['inception_to_claim'].iloc[0] == 100.0
    assert result['min_claim_driver_age'].iloc[0] == 35.0
    
    # Check damage cols (filled with -1)
    assert float(result['damageScore'].iloc[0]) == -1
    assert float(result['areasDamagedMinimal'].iloc[0]) == -1
    
    # Check bool cols (filled with -1, converted to string)
    assert result['vehicle_unattended'].iloc[0] == '-1'
    assert result['is_first_party'].iloc[0] == '-1'
    assert result['total_loss_flag'].iloc[0] == '-1'
    
    # Check one fills (filled with 1, converted to string)
    assert result['C1_fri_sat_night'].iloc[0] == '1'
    assert result['C2_reporting_delay'].iloc[0] == '1'
    
    # Check string cols (filled with 'missing')
    assert result['assessment_category'].iloc[0] == 'missing'
    assert result['incident_type'].iloc[0] == 'missing'
    assert result['vehicle_overnight_location_id'].iloc[0] == 'missing'