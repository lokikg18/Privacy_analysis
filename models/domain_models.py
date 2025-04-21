from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DataType(Enum):
    LOCATION = "location"
    VIDEO = "video"
    AUDIO = "audio"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    HEALTH = "health"
    IDENTIFICATION = "identification"


class DeviceType(Enum):
    CAMERA = "camera"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    WEARABLE = "wearable"


class LocationType(Enum):
    PUBLIC_SPACE = "public_space"
    PRIVATE_SPACE = "private_space"
    SEMI_PUBLIC = "semi_public"
    RESTRICTED = "restricted"


class PrivacyRiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4
    CRITICAL = 5


class AccessPattern(Enum):
    REGULAR = "regular"
    IRREGULAR = "irregular"
    BURST = "burst"


class DataSharing(Enum):
    NONE = "none"
    INTERNAL = "internal"
    EXTERNAL = "external"


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"


class User(BaseModel):
    id: str
    name: str
    email: str
    role: str
    consent_status: Dict[str, bool] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    def update_consent(self, data_type: str, status: bool) -> None:
        """Update consent status for a specific data type."""
        self.consent_status[data_type] = status
        self.updated_at = datetime.now()


class IoTDevice(BaseModel):
    id: str
    name: str
    device_type: str
    location: str
    location_type: str
    data_types: List[str]
    network_security_level: int = Field(ge=1, le=5)
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@dataclass
class PersonalData:
    id: str
    type: DataType
    owner_id: str
    device_id: str
    sensitivity_level: int  # 1-5
    collection_timestamp: datetime
    retention_period: Optional[int]  # in days
    is_encrypted: bool
    access_log: List[Dict[str, datetime]]

    def add_access_log(self, user_id: str) -> None:
        """Log access to the personal data."""
        self.access_log.append({"user_id": user_id, "timestamp": datetime.now()})


@dataclass
class PrivacyRisk:
    id: str
    level: PrivacyRiskLevel
    description: str
    affected_data_types: List[DataType]
    affected_devices: List[str]
    mitigation_strategy: str
    detected_at: datetime
    resolved_at: Optional[datetime]
    status: str  # active, mitigated, resolved

    def resolve(self) -> None:
        """Mark the risk as resolved."""
        self.status = "resolved"
        self.resolved_at = datetime.now()


class PrivacyPolicy(BaseModel):
    id: str
    name: str
    description: str
    data_types: List[str]
    retention_period: int
    access_control_rules: List[Dict]
    compliance_requirements: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    def update_retention_period(self, days: int) -> None:
        """Update the retention period for data."""
        self.retention_period = days
        self.updated_at = datetime.now()

    def add_access_rule(self, rule: Dict) -> None:
        """Add a new access control rule."""
        self.access_control_rules.append(rule)
        self.updated_at = datetime.now()
