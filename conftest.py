import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml_models.privacy_risk_classifier import PrivacyRiskClassifier
from models.domain_models import ComplianceStatus
from ontology_handlers.privacy_ontology import PrivacyOntologyHandler

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Configure pytest-asyncio
def pytest_configure(config):
    config.option.asyncio_mode = "strict"
    config.option.asyncio_default_fixture_loop_scope = "function"
    config.option.asyncio_default_test_loop_scope = "function"


@pytest.fixture(autouse=True)
def mock_classifier(monkeypatch):
    """Mock the privacy risk classifier for testing."""
    mock = MagicMock(spec=PrivacyRiskClassifier)
    mock.predict.return_value = np.array([3])  # Return medium risk level
    mock.predict_proba.return_value = np.array(
        [[0.1, 0.2, 0.4, 0.2, 0.1]]
    )  # Return probability distribution

    # Patch the classifier in the main module
    monkeypatch.setattr("api.main.classifier", mock)
    return mock


@pytest.fixture(autouse=True)
def mock_ontology_handler(monkeypatch):
    """Mock the privacy ontology handler for testing."""
    mock = MagicMock(spec=PrivacyOntologyHandler)

    # Mock all methods used in the API
    mock.get_mitigation_strategies.return_value = ["Test mitigation strategy"]
    mock.get_personal_data_types.return_value = ["video", "audio"]
    mock.get_risk_levels.return_value = {"test_risk": 3}
    mock.get_risk.return_value = None
    mock.get_risks.return_value = []
    mock.get_policy.return_value = None
    mock.add_policy.return_value = None
    mock.add_personal_data.return_value = None
    mock.add_risk.return_value = None
    mock.save_ontology.return_value = None

    # Create mock methods for compliance and risk analysis
    mock.check_compliance = MagicMock(return_value=ComplianceStatus.COMPLIANT)
    mock.analyze_risk_factors = MagicMock(
        return_value={
            "data_sensitivity": 0.8,
            "user_consent": 0.2,
            "location_risk": 0.5,
            "device_type_risk": 0.6,
        }
    )

    # Patch the ontology handler in the main module
    monkeypatch.setattr("api.main.ontology_handler", mock)
    return mock
