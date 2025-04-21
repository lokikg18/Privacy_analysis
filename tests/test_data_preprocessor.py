from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_models.data_preprocessor import (DataPreprocessor, prepare_test_data,
                                         prepare_training_data)


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    return pd.DataFrame(
        {
            "device_type": ["camera", "sensor", "camera", "sensor"],
            "data_type": ["video", "temperature", "video", "humidity"],
            "location_type": ["public", "private", "public", "private"],
            "access_frequency": [10, 5, 15, 3],
            "network_security_level": [3, 4, 2, 5],
            "data_sensitivity": [4, 2, 5, 1],
            "encryption_level": [3, 4, 2, 5],
            "retention_period": [30, 90, 15, 180],
            "data_volume": [1000, 100, 2000, 50],
            "last_audit_days": [30, 15, 60, 7],
            "access_pattern": ["continuous", "periodic", "continuous", "periodic"],
            "user_consent": [True, True, False, True],
            "data_anonymization": [True, False, True, False],
            "data_pseudonymization": [False, True, False, True],
            "data_minimization": [True, True, False, True],
            "purpose_limitation": [True, False, True, False],
            "storage_duration": [30, 90, 15, 180],
            "data_sharing": ["none", "limited", "none", "extensive"],
            "compliance_status": [
                "compliant",
                "non_compliant",
                "compliant",
                "non_compliant",
            ],
            "security_incidents": [0, 1, 2, 0],
            "privacy_impact_assessment": [True, False, True, False],
            "data_protection_officer": [True, True, False, True],
            "risk_level": [3, 2, 4, 1],
        }
    )


def test_data_preprocessor_initialization():
    """Test DataPreprocessor initialization."""
    preprocessor = DataPreprocessor()
    assert isinstance(preprocessor.scaler, type(preprocessor.scaler))
    assert isinstance(preprocessor.label_encoders, dict)
    assert isinstance(preprocessor.feature_columns, list)
    assert preprocessor.target_column == "risk_level"


def test_convert_boolean_features(sample_data):
    """Test boolean feature conversion."""
    preprocessor = DataPreprocessor()
    df_converted = preprocessor._convert_boolean_features(sample_data)

    boolean_columns = [
        "user_consent",
        "data_anonymization",
        "data_pseudonymization",
        "data_minimization",
        "purpose_limitation",
        "privacy_impact_assessment",
        "data_protection_officer",
    ]

    for col in boolean_columns:
        assert df_converted[col].dtype == np.int64
        assert set(df_converted[col].unique()).issubset({0, 1})


def test_encode_categorical_features(sample_data):
    """Test categorical feature encoding."""
    preprocessor = DataPreprocessor()
    df_encoded = preprocessor._encode_categorical_features(sample_data)

    categorical_columns = [
        "device_type",
        "data_type",
        "location_type",
        "access_pattern",
        "data_sharing",
        "compliance_status",
    ]

    for col in categorical_columns:
        assert df_encoded[col].dtype == np.int64
        assert len(df_encoded[col].unique()) == len(sample_data[col].unique())


def test_scale_numerical_features(sample_data):
    """Test numerical feature scaling."""
    preprocessor = DataPreprocessor()
    df_scaled = preprocessor._scale_numerical_features(sample_data)

    numerical_columns = [
        "access_frequency",
        "network_security_level",
        "data_sensitivity",
        "encryption_level",
        "retention_period",
        "data_volume",
        "last_audit_days",
        "storage_duration",
        "security_incidents",
    ]

    for col in numerical_columns:
        assert np.isclose(df_scaled[col].mean(), 0, atol=1e-7)
        assert np.isclose(df_scaled[col].std(), 1, atol=0.2)


def test_preprocess_data(sample_data):
    """Test complete data preprocessing."""
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(sample_data)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)
    assert X.shape[1] == len(preprocessor.feature_columns)


def test_preprocess_new_data(sample_data):
    """Test preprocessing of new data."""
    preprocessor = DataPreprocessor()
    # First fit the preprocessor
    preprocessor.preprocess_data(sample_data)

    # Then preprocess new data
    new_data = sample_data.iloc[:2]
    X_new = preprocessor.preprocess_new_data(new_data)

    assert isinstance(X_new, np.ndarray)
    assert X_new.shape[0] == len(new_data)
    assert X_new.shape[1] == len(preprocessor.feature_columns)


def test_save_and_load_preprocessor(sample_data, tmp_path):
    """Test saving and loading preprocessor."""
    preprocessor = DataPreprocessor()
    preprocessor.model_path = tmp_path / "preprocessor.joblib"

    # Fit and save preprocessor
    preprocessor.preprocess_data(sample_data)
    preprocessor.save_preprocessor()

    # Create new preprocessor and load
    new_preprocessor = DataPreprocessor()
    new_preprocessor.model_path = tmp_path / "preprocessor.joblib"
    new_preprocessor.load_preprocessor()

    # Test if loaded preprocessor works
    X, y = new_preprocessor.preprocess_data(sample_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_prepare_training_data():
    """Test training data preparation."""
    # Test data preparation
    X_train, X_val, y_train, y_val = prepare_training_data()

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_val, np.ndarray)

    # Check shapes
    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_val.shape[0] > 0
    assert X_train.shape[0] + X_val.shape[0] == y_train.shape[0] + y_val.shape[0]


def test_prepare_test_data():
    """Test test data preparation."""
    # First prepare training data to fit preprocessor
    prepare_training_data()

    # Test test data preparation
    X_test, y_test = prepare_test_data()

    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert X_test.shape[0] > 0
    assert y_test.shape[0] > 0
    assert X_test.shape[0] == y_test.shape[0]
