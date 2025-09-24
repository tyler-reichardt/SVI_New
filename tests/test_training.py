import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve, auc

# Import the functions to test
from functions.training import (
    do_fills_pd,
    simple_classification_report,
    generate_classification_report,
    pr_auc,
    prepare_training_data,
    create_preprocessing_pipeline,
    train_desk_check_model,
    train_interview_model,
    plot_feature_importance
)


def test_do_fills_pd():
    """Test the do_fills_pd function"""
    # Create test data with missing values
    test_data = pd.DataFrame({
        'policyholder_ncd_years': [5.0, np.nan, 10.0],
        'C1_fri_sat_night': [1, np.nan, 0],
        'vehicle_unattended': [0, np.nan, 1],
        'assessment_category': ['Driveable', np.nan, 'TotalLoss'],
        'damageScore': [10, np.nan, 20]
    })
    
    mean_dict = {'policyholder_ncd_years': 7.5}
    
    result = do_fills_pd(test_data, mean_dict)
    
    # Check that missing values are filled correctly
    assert result['policyholder_ncd_years'].iloc[1] == 7.5  # Filled with mean
    assert result['C1_fri_sat_night'].iloc[1] == 1  # Rule variable filled with 1
    assert result['vehicle_unattended'].iloc[1] == -1  # Boolean filled with -1
    assert result['assessment_category'].iloc[1] == 'missing'  # String filled with 'missing'
    assert result['damageScore'].iloc[1] == -1  # Damage score filled with -1
    
    # Check that existing values are preserved
    assert result['policyholder_ncd_years'].iloc[0] == 5.0
    assert result['C1_fri_sat_night'].iloc[0] == 1


def test_simple_classification_report():
    """Test the simple_classification_report function"""
    y_prob = np.array([0.2, 0.6, 0.8, 0.3])
    y_true = np.array([0, 1, 1, 0])
    
    result = simple_classification_report(y_prob, y_true, threshold=0.5)
    
    # Check that results are returned
    assert 'accuracy' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1' in result
    assert 'confusion_matrix' in result
    
    # Check confusion matrix shape
    assert result['confusion_matrix'].shape == (2, 2)


def test_generate_classification_report():
    """Test the generate_classification_report function"""
    y_prob = np.array([0.2, 0.6, 0.8, 0.3])
    y_true = np.array([0, 1, 1, 0])
    
    report = generate_classification_report(y_prob, y_true, threshold=0.5)
    
    # Check that report is a string
    assert isinstance(report, str)
    assert 'precision' in report
    assert 'recall' in report


def test_pr_auc():
    """Test PR-AUC calculation"""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    
    pr_auc_score = pr_auc(y_true, y_scores)
    
    # Calculate expected PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    expected_auc = auc(recall, precision)
    
    assert abs(pr_auc_score - expected_auc) < 0.001


def test_prepare_training_data():
    """Test prepare_training_data function"""
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'fa_risk': [0, 1, 0, 1, 0],
        'tbg_risk': [1, 0, 1, 0, 1],
        'svi_risk': [0, 0, 1, 1, 0],
        'claim_number': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'dataset': ['train', 'test', 'train', 'test', 'train']
    })
    
    # Test with default target
    X, y = prepare_training_data(data)
    assert 'fa_risk' not in X.columns
    assert 'tbg_risk' not in X.columns
    assert 'svi_risk' not in X.columns
    assert 'claim_number' not in X.columns
    assert 'dataset' not in X.columns
    assert len(y) == 5
    
    # Test with different target column
    X, y = prepare_training_data(data, target_col='svi_risk')
    assert y.tolist() == [0, 0, 1, 1, 0]


def test_create_preprocessing_pipeline():
    """Test preprocessing pipeline creation"""
    numeric_features = ['age', 'income']
    categorical_features = ['gender', 'city']
    
    pipeline = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    assert isinstance(pipeline, ColumnTransformer)
    assert len(pipeline.transformers) == 2
    assert pipeline.transformers[0][0] == 'num'
    assert pipeline.transformers[1][0] == 'cat'


def test_train_desk_check_model():
    """Test desk check model training"""
    # Create sample data
    np.random.seed(42)
    X_train = pd.DataFrame({
        'policyholder_ncd_years': np.random.rand(100) * 10,
        'min_claim_driver_age': np.random.randint(18, 65, 100),
        'assessment_category': np.random.choice(['Driveable', 'TotalLoss'], 100),
        'C1_fri_sat_night': np.random.choice([0, 1], 100)
    })
    y_train = np.random.choice([0, 1], 100)
    X_test = X_train.iloc[:20].copy()
    y_test = y_train[:20]
    
    numeric_features = ['policyholder_ncd_years', 'min_claim_driver_age']
    categorical_features = ['assessment_category', 'C1_fri_sat_night']
    
    with patch('mlflow.start_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.sklearn.log_model'):
        
        model, metrics = train_desk_check_model(
            X_train, y_train, X_test, y_test,
            numeric_features, categorical_features,
            run_hyperparameter_tuning=False
        )
        
        assert model is not None
        assert isinstance(model, Pipeline)
        assert metrics is not None
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics


def test_train_interview_model():
    """Test interview model training"""
    # Create sample data
    np.random.seed(42)
    X_train = pd.DataFrame({
        'voluntary_amount': np.random.rand(100) * 1000,
        'policyholder_ncd_years': np.random.randint(0, 10, 100),
        'assessment_category': np.random.choice(['Driveable', 'TotalLoss'], 100),
        'left_severity': np.random.choice(['Minimal', 'Medium', 'Heavy'], 100),
        'C9_policy_within_30_days': np.random.choice([0, 1], 100),
        'checks_max': np.random.choice([0, 1], 100),
        'total_loss_flag': np.random.choice([True, False], 100)
    })
    y_train = np.random.choice([0, 1], 100)
    X_test = X_train.iloc[:20].copy()
    y_test = y_train[:20]
    
    numeric_features = ['voluntary_amount', 'policyholder_ncd_years']
    categorical_features = ['assessment_category', 'left_severity', 'C9_policy_within_30_days', 
                           'checks_max', 'total_loss_flag']
    
    with patch('mlflow.start_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.sklearn.log_model'):
        
        model, metrics = train_interview_model(
            X_train, y_train, X_test, y_test,
            numeric_features, categorical_features
        )
        
        assert model is not None
        assert isinstance(model, Pipeline)
        assert metrics is not None
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics


def test_plot_feature_importance():
    """Test feature importance plotting"""
    # Create a mock model with feature importances
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.35])
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
    
    with patch('matplotlib.pyplot.show'):
        # Should not raise any errors
        plot_feature_importance(mock_model, feature_names, top_n=3)


def test_prepare_training_data_with_different_targets():
    """Test prepare_training_data with different target columns"""
    # Create sample data with multiple target columns
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'fa_risk': [0, 1, 0, 1, 0],
        'tbg_risk': [1, 0, 1, 0, 1],
        'svi_risk': [0, 0, 1, 1, 0]
    })
    
    # Test with fa_risk
    X, y = prepare_training_data(data, target_col='fa_risk')
    assert y.tolist() == [0, 1, 0, 1, 0]
    assert 'fa_risk' not in X.columns
    assert 'tbg_risk' not in X.columns
    assert 'svi_risk' not in X.columns
    
    # Test with tbg_risk
    X, y = prepare_training_data(data, target_col='tbg_risk')
    assert y.tolist() == [1, 0, 1, 0, 1]
    
    # Test with svi_risk
    X, y = prepare_training_data(data, target_col='svi_risk')
    assert y.tolist() == [0, 0, 1, 1, 0]


def test_do_fills_pd_edge_cases():
    """Test do_fills_pd with edge cases"""
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    mean_dict = {'col1': 10.0}
    result = do_fills_pd(empty_df, mean_dict)
    assert result.empty
    
    # Test with all NaN values
    nan_df = pd.DataFrame({
        'policyholder_ncd_years': [np.nan, np.nan],
        'C1_fri_sat_night': [np.nan, np.nan],
        'vehicle_unattended': [np.nan, np.nan],
        'assessment_category': [np.nan, np.nan]
    })
    
    mean_dict = {'policyholder_ncd_years': 5.0}
    result = do_fills_pd(nan_df, mean_dict)
    
    # Check fills are applied correctly
    assert result['policyholder_ncd_years'].iloc[0] == 5.0
    assert result['C1_fri_sat_night'].iloc[0] == 1  # one_fills
    assert result['vehicle_unattended'].iloc[0] == -1  # bool_cols
    assert result['assessment_category'].iloc[0] == 'missing'  # string_cols


def test_simple_classification_report_edge_cases():
    """Test simple_classification_report with edge cases"""
    # Test with all predictions being the same class
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    y_true = np.array([0, 0, 1, 1])
    
    result = simple_classification_report(y_prob, y_true, threshold=0.5)
    
    # All predictions should be 0 (below threshold)
    assert result['precision'] == 0.0 or np.isnan(result['precision'])  # No positive predictions
    
    # Test with perfect predictions
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])
    
    result = simple_classification_report(y_prob, y_true, threshold=0.5)
    assert result['accuracy'] == 1.0
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0


def test_generate_classification_report_formatting():
    """Test that generate_classification_report produces properly formatted output"""
    y_prob = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    y_true = np.array([0, 1, 0, 1, 0])
    
    report = generate_classification_report(y_prob, y_true, threshold=0.5)
    
    # Check that report contains expected sections
    assert 'Classification Report' in report
    assert 'accuracy' in report.lower()
    assert 'macro avg' in report or 'weighted avg' in report