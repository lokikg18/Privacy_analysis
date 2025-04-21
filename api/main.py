import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_import_path():
    """Setup the import path."""
    try:
        # Get the current directory
        current_dir = Path.cwd()
        logger.info(f"Current directory: {current_dir}")

        # In Colab, we need to add the project root directory
        # Try to find the project root by looking for key directories
        project_root = current_dir
        while (
            not (project_root / "models").exists()
            and project_root.parent != project_root
        ):
            project_root = project_root.parent

        if not (project_root / "models").exists():
            raise ImportError(
                "Could not find project root directory containing 'models' folder"
            )

        # Add the project root to Python path
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
            logger.info(f"Added to Python path: {project_root}")

        return project_root
    except Exception as e:
        logger.error(f"Error setting up import path: {e}")
        raise


# Setup import path
project_root = setup_import_path()
logger.info(f"Project root: {project_root}")

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from ml_models.privacy_risk_classifier import PrivacyRiskClassifier
# Now import the required modules
from models.domain_models import (AccessPattern, ComplianceStatus, DataSharing,
                                  DataType, DeviceType, IoTDevice,
                                  LocationType, PersonalData, PrivacyPolicy,
                                  PrivacyRisk, PrivacyRiskLevel, User)
from ontology_handlers.privacy_ontology import PrivacyOntologyHandler

# Initialize FastAPI app
app = FastAPI(
    title="Privacy Risk Assessment API",
    description="API for assessing and managing privacy risks in IoT systems",
    version="1.0.0",
)

# Initialize components
classifier = PrivacyRiskClassifier()
ontology_handler = PrivacyOntologyHandler()

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# In-memory storage (replace with database in production)
users_db = {}
devices_db = {}
policies_db = {}
risk_history = []


# Pydantic models
class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str = "user"


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    consent_status: Dict[str, bool] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DeviceCreate(BaseModel):
    name: str
    device_type: DeviceType
    location: str
    location_type: LocationType
    data_types: List[DataType]
    network_security_level: int
    description: Optional[str] = None


class DeviceResponse(BaseModel):
    id: str
    name: str
    device_type: DeviceType
    location: str
    location_type: LocationType
    data_types: List[DataType]
    network_security_level: int
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class PolicyCreate(BaseModel):
    name: str
    description: str
    data_types: List[DataType]
    retention_period: int
    access_control_rules: List[Dict]
    compliance_requirements: List[str]


class PolicyResponse(BaseModel):
    id: str
    name: str
    description: str
    data_types: List[DataType]
    retention_period: int
    access_control_rules: List[Dict]
    compliance_requirements: List[str]
    created_at: datetime
    updated_at: datetime


class RiskAssessmentRequest(BaseModel):
    device_id: str
    user_consent: bool
    data_sensitivity: int
    additional_context: Optional[Dict] = None


class RiskAssessmentResponse(BaseModel):
    risk_level: int
    risk_probabilities: List[float]
    mitigation_suggestions: List[str]
    compliance_status: ComplianceStatus
    risk_factors: Dict[str, float]


class RiskHistoryResponse(BaseModel):
    id: str
    device_id: str
    risk_level: int
    timestamp: datetime
    mitigation_suggestions: List[str]
    resolved: bool


# Helper functions
def get_current_user(token: str = Depends(oauth2_scheme)):
    if token not in users_db:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )
    return users_db[token]


# API endpoints
@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    """Create a new user."""
    try:
        # Check if user already exists
        if any(u.email == user.email for u in users_db.values()):
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = str(uuid.uuid4())
        new_user = User(
            id=user_id,
            name=user.name,
            email=user.email,
            role=user.role,
            consent_status={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        users_db[user_id] = new_user
        return new_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/devices", response_model=DeviceResponse)
async def create_device(
    device: DeviceCreate, current_user: User = Depends(get_current_user)
):
    """Create a new IoT device."""
    try:
        device_id = str(uuid.uuid4())
        new_device = IoTDevice(
            id=device_id,
            name=device.name,
            device_type=device.device_type,
            location=device.location,
            location_type=device.location_type,
            data_types=device.data_types,
            network_security_level=device.network_security_level,
            description=device.description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        devices_db[device_id] = new_device
        return new_device
    except Exception as e:
        logger.error(f"Error creating device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/policies", response_model=PolicyResponse)
async def create_policy(
    policy: PolicyCreate, current_user: User = Depends(get_current_user)
):
    """Create a new privacy policy."""
    try:
        policy_id = str(uuid.uuid4())
        new_policy = PrivacyPolicy(
            id=policy_id,
            name=policy.name,
            description=policy.description,
            data_types=policy.data_types,
            retention_period=policy.retention_period,
            access_control_rules=policy.access_control_rules,
            compliance_requirements=policy.compliance_requirements,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        policies_db[policy_id] = new_policy
        return new_policy
    except Exception as e:
        logger.error(f"Error creating policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policies/{policy_id}", response_model=PolicyResponse)
async def get_policy(policy_id: str, current_user: User = Depends(get_current_user)):
    """Get a privacy policy by ID."""
    try:
        if policy_id not in policies_db:
            raise HTTPException(status_code=404, detail="Policy not found")
        return policies_db[policy_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/policies/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: str, policy: PolicyCreate, current_user: User = Depends(get_current_user)
):
    """Update a privacy policy."""
    try:
        if policy_id not in policies_db:
            raise HTTPException(status_code=404, detail="Policy not found")

        updated_policy = PrivacyPolicy(
            id=policy_id,
            name=policy.name,
            description=policy.description,
            data_types=policy.data_types,
            retention_period=policy.retention_period,
            access_control_rules=policy.access_control_rules,
            compliance_requirements=policy.compliance_requirements,
            created_at=policies_db[policy_id].created_at,
            updated_at=datetime.now(),
        )
        policies_db[policy_id] = updated_policy
        return updated_policy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess_risk", response_model=RiskAssessmentResponse)
async def assess_risk(
    request: RiskAssessmentRequest, current_user: User = Depends(get_current_user)
):
    """Assess privacy risk for a device."""
    try:
        device = devices_db.get(request.device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Prepare data for classifier
        input_data = [
            {
                "device_type": device.device_type,
                "data_type": device.data_types[0],  # Use first data type for now
                "location_type": device.location_type,
                "access_frequency": 1,  # Default value
                "user_consent": request.user_consent,
                "network_security_level": device.network_security_level,
                "data_sensitivity": request.data_sensitivity,
            }
        ]

        # Get risk assessment
        risk_level = classifier.predict(input_data)[0]
        probabilities = classifier.predict_proba(input_data)[0]

        # Get mitigation suggestions from ontology
        mitigation_suggestions = ontology_handler.get_mitigation_strategies(risk_level)

        # Check compliance status
        compliance_status = ontology_handler.check_compliance(
            device, request.user_consent
        )

        # Get risk factors
        risk_factors = ontology_handler.analyze_risk_factors(
            device, request.user_consent
        )

        # Store risk assessment in history
        risk_history.append(
            {
                "id": str(uuid.uuid4()),
                "device_id": request.device_id,
                "risk_level": risk_level,
                "timestamp": datetime.now(),
                "mitigation_suggestions": mitigation_suggestions,
                "resolved": False,
            }
        )

        return RiskAssessmentResponse(
            risk_level=risk_level,
            risk_probabilities=probabilities.tolist(),
            mitigation_suggestions=mitigation_suggestions,
            compliance_status=compliance_status,
            risk_factors=risk_factors,
        )
    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk_history/{device_id}", response_model=List[RiskHistoryResponse])
async def get_risk_history(
    device_id: str, current_user: User = Depends(get_current_user)
):
    """Get risk assessment history for a device."""
    try:
        if device_id not in devices_db:
            raise HTTPException(status_code=404, detail="Device not found")
        device_history = [
            risk for risk in risk_history if risk["device_id"] == device_id
        ]
        return device_history
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving risk history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/risks/{risk_id}/resolve")
async def resolve_risk(risk_id: str, current_user: User = Depends(get_current_user)):
    """Mark a risk assessment as resolved."""
    try:
        for risk in risk_history:
            if risk["id"] == risk_id:
                risk["resolved"] = True
                risk["resolved_at"] = datetime.now()
                risk["resolved_by"] = current_user.id
                return {"message": "Risk resolved successfully"}
        raise HTTPException(status_code=404, detail="Risk not found")
    except Exception as e:
        logger.error(f"Error resolving risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/devices", response_model=List[DeviceResponse])
async def list_devices(current_user: User = Depends(get_current_user)):
    """List all devices."""
    try:
        return list(devices_db.values())
    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "classifier": "operational",
            "ontology_handler": "operational",
            "database": "operational",
        },
    }


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Generate access token for user login."""
    try:
        # Find user by email (username in form_data)
        user = next(
            (u for u in users_db.values() if u.email == form_data.username), None
        )
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # In a real application, verify password hash here
        # For demo, we just check if passwords match
        if form_data.password != "test_password":  # Hardcoded for testing
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Generate token
        token = str(uuid.uuid4())
        users_db[token] = user  # Store token in users_db for simplicity

        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
