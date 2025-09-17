#import pytest
#from pyspark.sql import SparkSession, Row
#from pyspark.sql.types import (
#    StructType, StructField, StringType, IntegerType, DoubleType, 
#    BooleanType
#)
#from pyspark.sql.functions import col, lit, when
#from unittest.mock import MagicMock, patch, Mock, ANY
#import pandas as pd
#import numpy as np
#from sklearn.datasets import make_classification
#from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline as SklearnPipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.compose import ColumnTransformer
#from sklearn.metrics import accuracy_score, roc_auc_score
#from lightgbm import LGBMClassifier
#import mlflow
#from mlflow.models import infer_signature
#
## Mock the notebooks module structure
#import sys
#sys.path.append('..')
#from functions.training import *
#
#@pytest.fixture(scope="session")
#def spark_session():
#    """Creates a Spark session for all tests in this file."""
#    spark = SparkSession.builder.master("local[1]").appName("TrainingTests").getOrCreate()
#    yield spark
#    spark.stop()
#
#@pytest.fixture
#def mock_env_config():
#    """Provides a mock environment configuration dictionary."""
#    return {
#        'mlstore_catalog': 'mock_ml_catalog',
#        'auxiliary_catalog': 'mock_aux_catalog',
#        'adp_catalog': 'mock_adp_catalog'
#    }
#
#@pytest.fixture
#def svi_model_trainer(spark_session, mock_env_config, mocker):
#    """Initializes SVIModelTraining with a mocked MLflow client."""
#    mocker.patch('your_module_name.mlflow.tracking.MlflowClient', return_value=MagicMock())
#    trainer = SVIModelTraining(spark=spark_session, env_config=mock_env_config)
#    return trainer
#
#@pytest.fixture
#def sample_pandas_df():
#    """Provides a sample pandas DataFrame for feature preparation tests."""
#    data = {
#        'claim_number': ['C1', 'C2', 'C3'], 'dataset': ['train', 'train', 'test'],
#        'svi_risk': [1, 0, 1], 'referred_to_tbg': [1, 0, 1],
#        'policyholder_ncd_years': [5, 2, np.nan], 'min_claim_driver_age': [25, 40, 30],
#        'veh_age': [2, 10, 5], 'inception_to_claim': [100, 200, 50],
#        'assessment_category': ['CAT_A', 'CAT_B', 'CAT_A'],
#        'C9_policy_within_30_days': [0, 0, 1], 'checks_max': [0, 0, 1],
#        'total_loss_flag': [False, True, False], 'desk_check_pred': [0.8, 0.1, 0.75]
#    }
#    df = pd.DataFrame(data)
#    df['total_loss_flag'] = df['total_loss_flag'].astype(bool)
#    return df
#
## --- Tests ---
#
#def test_init(svi_model_trainer, mock_env_config):
#    """Tests if the class initializes correctly."""
#    assert isinstance(svi_model_trainer.client, MagicMock)
#    expected_desk_model = f"{mock_env_config['mlstore_catalog']}.single_vehicle_incident_checks.svi_desk_check_lgbm"
#    assert svi_model_trainer.desk_check_model_name == expected_desk_model
#
#def test_load_training_data(svi_model_trainer, spark_session):
#    """Tests data loading and splitting into pandas DataFrames."""
#    mock_data = [('C1', 'train', 1), ('C2', 'train', 0), ('C3', 'test', 1)]
#    mock_spark_df = spark_session.createDataFrame(mock_data, ["id", "dataset", "val"])
#    
#    with patch.object(svi_model_trainer.spark, 'table', return_value=mock_spark_df):
#        train_df, test_df = svi_model_trainer.load_training_data()
#        assert len(train_df) == 2
#        assert len(test_df) == 1
#
#def test_prepare_desk_check_features(svi_model_trainer, sample_pandas_df):
#    """Tests the feature list and preprocessor creation for the desk check model."""
#    features, preprocessor = svi_model_trainer.prepare_desk_check_features(sample_pandas_df)
#    assert isinstance(features, list)
#    assert 'min_claim_driver_age' in features
#    assert isinstance(preprocessor, ColumnTransformer)
#    transformed_X = preprocessor.fit_transform(sample_pandas_df[features])
#    assert transformed_X.shape[0] == len(sample_pandas_df)
#
#def test_prepare_interview_features(svi_model_trainer, sample_pandas_df):
#    """Tests the feature list and preprocessor creation for the interview model."""
#    features, preprocessor = svi_model_trainer.prepare_interview_features(sample_pandas_df)
#    assert 'desk_check_pred' in features
#    assert isinstance(preprocessor, ColumnTransformer)
#    transformed_X = preprocessor.fit_transform(sample_pandas_df[features])
#    assert transformed_X.shape[0] == len(sample_pandas_df)
#
#@patch('your_module_name.mlflow.sklearn.log_model')
#@patch('your_module_name.mlflow.log_metric')
#@patch('your_module_name.mlflow.log_params')
#@patch('your_module_name.GridSearchCV')
#def test_train_desk_check_model(mock_grid, mock_log_params, mock_log_metric, mock_log_model, svi_model_trainer, sample_pandas_df):
#    """Tests the desk check model training flow with robust mocking."""
#    mock_estimator = MagicMock()
#    mock_grid_instance = mock_grid.return_value
#    mock_grid_instance.best_estimator_ = mock_estimator
#    mock_grid_instance.best_params_ = {'classifier__n_estimators': 30}
#    svi_model_trainer.get_latest_model_version = MagicMock(return_value="1")
#    svi_model_trainer.plot_confusion_matrix = MagicMock()
#    svi_model_trainer.log_feature_importance = MagicMock()
#    svi_model_trainer.simple_classification_report = MagicMock()
#
#    features, preprocessor = svi_model_trainer.prepare_desk_check_features(sample_pandas_df)
#    X, y = sample_pandas_df, sample_pandas_df['referred_to_tbg']
#    
#    svi_model_trainer.train_desk_check_model(X, y, X, y, features, preprocessor)
#    
#    mock_grid_instance.fit.assert_called_once()
#    mock_log_params.assert_called_once_with({'classifier__n_estimators': 30})
#    mock_log_metric.assert_any_call("recall", ANY)
#    mock_log_model.assert_called_once()
#
#@patch('your_module_name.mlflow.sklearn.log_model')
#@patch('your_module_name.GridSearchCV')
#def test_train_interview_model(mock_grid, mock_log_model, svi_model_trainer, sample_pandas_df):
#    """Tests the interview model training flow."""
#    mock_estimator = MagicMock()
#    mock_grid_instance = mock_grid.return_value
#    mock_grid_instance.best_estimator_ = mock_estimator
#    svi_model_trainer.get_latest_model_version = MagicMock(return_value="1")
#    svi_model_trainer.plot_confusion_matrix = MagicMock()
#    svi_model_trainer.log_feature_importance = MagicMock()
#    svi_model_trainer.simple_classification_report = MagicMock()
#
#    features, preprocessor = svi_model_trainer.prepare_interview_features(sample_pandas_df)
#    X, y = sample_pandas_df, sample_pandas_df['svi_risk']
#    desk_preds = np.random.rand(len(X))
#    
#    svi_model_trainer.train_interview_model(X, y, X, y, desk_preds, desk_preds, features, preprocessor)
#    
#    mock_grid_instance.fit.assert_called_once()
#    mock_log_model.assert_called_once()
#
#def test_get_latest_model_version(svi_model_trainer):
#    """Tests the logic for finding the latest model version from a mocked client."""
#    mock_version_1 = MagicMock()
#    mock_version_1.version = "1"
#    mock_version_2 = MagicMock()
#    mock_version_2.version = "10"
#    mock_version_3 = MagicMock()
#    mock_version_3.version = "2"
#    
#    svi_model_trainer.client.search_model_versions.return_value = [mock_version_1, mock_version_2, mock_version_3]
#    
#    latest_version = svi_model_trainer.get_latest_model_version("any_model")
#    
#    assert latest_version == 10
#    svi_model_trainer.client.search_model_versions.assert_called_with("name='any_model'")
#
#@patch('your_module_name.SVIModelTraining.train_interview_model')
#@patch('your_module_name.SVIModelTraining.train_desk_check_model')
#@patch('your_module_name.SVIModelTraining.load_training_data')
#def test_run_training_pipeline(mock_load, mock_train_desk, mock_train_int, svi_model_trainer, sample_pandas_df):
#    """Tests the end-to-end orchestration of the training pipeline."""
#    mock_train_df = sample_pandas_df[sample_pandas_df['dataset'] == 'train']
#    mock_test_df = sample_pandas_df[sample_pandas_df['dataset'] == 'test']
#    mock_load.return_value = (mock_train_df, mock_test_df)
#    
#    mock_desk_model = MagicMock()
#    mock_desk_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.9, 0.1]])
#    mock_train_desk.return_value = (mock_desk_model, np.array([0.7]), np.array([1]))
#
#    mock_train_int.return_value = (MagicMock(), MagicMock(), MagicMock())
#    
#    svi_model_trainer.run_training_pipeline()
#    
#    mock_load.assert_called_once()
#    mock_train_desk.assert_called_once()
#    mock_train_int.assert_called_once()
#    
#    # Verify that the desk check predictions were passed correctly to the interview model
#    args, kwargs = mock_train_int.call_args
#    desk_preds_for_train = kwargs['desk_check_pred_train']
#    assert isinstance(desk_preds_for_train, np.ndarray)
#    assert desk_preds_for_train.shape == (2,)
