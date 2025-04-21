from datetime import datetime

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import app, devices_db, policies_db, risk_history, users_db
from models.domain_models import DataType, DeviceType, LocationType

# Create a test client
client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_database():
    """Clear all in-memory databases before each test."""
    users_db.clear()
    devices_db.clear()
    policies_db.clear()
    risk_history.clear()
    yield


@pytest.fixture
def test_user_data():
    return {
        "name": "test_user",
        "email": "test@example.com",
        "password": "test_password",
        "role": "user",
    }


@pytest.fixture
def test_device_data():
    return {
        "name": "Test Camera",
        "device_type": DeviceType.CAMERA.value,
        "location": "Living Room",
        "location_type": LocationType.PRIVATE_SPACE.value,
        "data_types": [DataType.VIDEO.value, DataType.AUDIO.value],
        "network_security_level": 3,
        "description": "Test security camera",
    }


@pytest.fixture
def test_risk_data():
    return {
        "device_id": "test_device_id",
        "user_consent": True,
        "data_sensitivity": 4,
        "additional_context": {
            "location": LocationType.PRIVATE_SPACE.value,
            "data_types": [DataType.VIDEO.value, DataType.AUDIO.value],
        },
    }


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert all(
        component in data["components"]
        for component in ["classifier", "ontology_handler", "database"]
    )
    assert all(
        data["components"][component] == "operational"
        for component in data["components"]
    )


def test_user_registration(test_user_data):
    """Test user registration endpoint"""
    response = client.post("/users", json=test_user_data)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == test_user_data["name"]
    assert data["email"] == test_user_data["email"]


def test_user_login(test_user_data):
    """Test user login endpoint"""
    # First register the user
    response = client.post("/users", json=test_user_data)
    assert response.status_code == 200

    # Then try to login
    response = client.post(
        "/token",
        data={
            "username": test_user_data["email"],
            "password": test_user_data["password"],
            "grant_type": "password",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data


def test_device_registration(test_user_data, test_device_data):
    """Test device registration endpoint"""
    # First get a token
    response = client.post("/users", json=test_user_data)
    assert response.status_code == 200

    response = client.post(
        "/token",
        data={
            "username": test_user_data["email"],
            "password": test_user_data["password"],
            "grant_type": "password",
        },
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    # Then register the device
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/devices", json=test_device_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == test_device_data["name"]


def test_risk_assessment(test_user_data, test_device_data, test_risk_data):
    """Test risk assessment endpoint"""
    # First get a token and register a device
    response = client.post("/users", json=test_user_data)
    assert response.status_code == 200

    response = client.post(
        "/token",
        data={
            "username": test_user_data["email"],
            "password": test_user_data["password"],
            "grant_type": "password",
        },
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    # Register device
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/devices", json=test_device_data, headers=headers)
    assert response.status_code == 200
    device = response.json()

    # Update risk data with actual device ID
    test_risk_data["device_id"] = device["id"]

    # Perform risk assessment
    response = client.post("/assess_risk", json=test_risk_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "risk_level" in data
    assert "mitigation_suggestions" in data


def test_risk_history(test_user_data, test_device_data):
    """Test risk history endpoint"""
    # First get a token and register a device
    response = client.post("/users", json=test_user_data)
    assert response.status_code == 200

    response = client.post(
        "/token",
        data={
            "username": test_user_data["email"],
            "password": test_user_data["password"],
            "grant_type": "password",
        },
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    # Register device
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/devices", json=test_device_data, headers=headers)
    assert response.status_code == 200
    device = response.json()

    # Get risk history
    response = client.get(f"/risk_history/{device['id']}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_policy_management(test_user_data):
    """Test policy management endpoints"""
    # First get a token
    response = client.post("/users", json=test_user_data)
    assert response.status_code == 200

    response = client.post(
        "/token",
        data={
            "username": test_user_data["email"],
            "password": test_user_data["password"],
            "grant_type": "password",
        },
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    # Create a policy
    policy_data = {
        "name": "Test Policy",
        "description": "Test privacy policy",
        "data_types": [DataType.VIDEO.value],
        "retention_period": 30,
        "access_control_rules": [{"role": "user", "access": "read"}],
        "compliance_requirements": ["GDPR", "CCPA"],
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/policies", json=policy_data, headers=headers)
    assert response.status_code == 200
    policy = response.json()

    # Get the policy
    response = client.get(f"/policies/{policy['id']}", headers=headers)
    assert response.status_code == 200
    assert response.json()["name"] == policy_data["name"]

    # Update the policy
    updated_policy = {**policy_data, "retention_period": 60}
    response = client.put(
        f"/policies/{policy['id']}", json=updated_policy, headers=headers
    )
    assert response.status_code == 200
    assert response.json()["retention_period"] == 60


def test_error_handling(test_user_data):
    """Test error handling"""
    # Test invalid token
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get("/devices", headers=headers)
    assert response.status_code == 401

    # Test invalid device ID
    response = client.post("/users", json=test_user_data)
    assert response.status_code == 200

    response = client.post(
        "/token",
        data={
            "username": test_user_data["email"],
            "password": test_user_data["password"],
            "grant_type": "password",
        },
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/risk_history/invalid_id", headers=headers)
    assert response.status_code == 404

    # Test invalid request data
    invalid_data = {"invalid": "data"}
    response = client.post("/devices", json=invalid_data, headers=headers)
    assert response.status_code == 422
