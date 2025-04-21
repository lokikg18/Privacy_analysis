from datetime import datetime

import pytest

from models.domain_models import (DataType, DeviceType, IoTDevice,
                                  LocationType, PersonalData, PrivacyPolicy,
                                  PrivacyRisk, PrivacyRiskLevel, User)


@pytest.fixture
def sample_user():
    return User(
        id="user1",
        name="John Doe",
        email="john@example.com",
        role="citizen",
        consent_status={"location": True, "video": False},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_device():
    return IoTDevice(
        id="device1",
        name="Test Camera",
        device_type=DeviceType.CAMERA.value,
        location="Street A",
        location_type=LocationType.PUBLIC_SPACE.value,
        data_types=[DataType.VIDEO.value, DataType.LOCATION.value],
        network_security_level=3,
        description="Test security camera",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_personal_data():
    return PersonalData(
        id="data1",
        type=DataType.LOCATION,
        owner_id="user1",
        device_id="device1",
        sensitivity_level=3,
        collection_timestamp=datetime.now(),
        retention_period=30,
        is_encrypted=True,
        access_log=[],
    )


@pytest.fixture
def sample_privacy_risk():
    return PrivacyRisk(
        id="risk1",
        level=PrivacyRiskLevel.HIGH,
        description="Unauthorized access attempt",
        affected_data_types=[DataType.LOCATION],
        affected_devices=["device1"],
        mitigation_strategy="Increase encryption level",
        detected_at=datetime.now(),
        resolved_at=None,
        status="active",
    )


@pytest.fixture
def sample_privacy_policy():
    return PrivacyPolicy(
        id="policy1",
        name="Location Data Policy",
        description="Policy for handling location data",
        data_types=[DataType.LOCATION.value],
        retention_period=30,
        access_control_rules=[{"role": "administrator", "access": "full"}],
        compliance_requirements=["GDPR"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def test_user_consent_update(sample_user):
    sample_user.update_consent("video", True)
    assert sample_user.consent_status["video"] is True
    assert sample_user.updated_at > sample_user.created_at


def test_device_data_type_addition(sample_device):
    initial_types = len(sample_device.data_types)
    sample_device.data_types.append(DataType.AUDIO.value)
    assert len(sample_device.data_types) == initial_types + 1
    assert DataType.AUDIO.value in sample_device.data_types


def test_device_security_level_update(sample_device):
    # Test that we can update to a valid security level
    sample_device.network_security_level = 4
    assert sample_device.network_security_level == 4

    # Test that creating a device with an invalid security level raises ValueError
    with pytest.raises(ValueError):
        IoTDevice(
            id="device2",
            name="Test Camera",
            device_type=DeviceType.CAMERA.value,
            location="Street A",
            location_type=LocationType.PUBLIC_SPACE.value,
            data_types=[DataType.VIDEO.value, DataType.LOCATION.value],
            network_security_level=6,  # Invalid security level
            description="Test security camera",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )


def test_personal_data_access_log(sample_personal_data):
    initial_log_length = len(sample_personal_data.access_log)
    sample_personal_data.add_access_log("user2")
    assert len(sample_personal_data.access_log) == initial_log_length + 1
    assert sample_personal_data.access_log[-1]["user_id"] == "user2"


def test_privacy_risk_resolution(sample_privacy_risk):
    assert sample_privacy_risk.status == "active"
    assert sample_privacy_risk.resolved_at is None

    sample_privacy_risk.resolve()
    assert sample_privacy_risk.status == "resolved"
    assert sample_privacy_risk.resolved_at is not None


def test_privacy_policy_updates(sample_privacy_policy):
    initial_rules = len(sample_privacy_policy.access_control_rules)
    sample_privacy_policy.add_access_rule({"role": "analyst", "access": "read"})
    assert len(sample_privacy_policy.access_control_rules) == initial_rules + 1

    sample_privacy_policy.update_retention_period(60)
    assert sample_privacy_policy.retention_period == 60
    assert sample_privacy_policy.updated_at > sample_privacy_policy.created_at
