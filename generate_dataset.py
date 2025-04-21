import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_import_path():
    """Setup the import path."""
    try:
        # Get the current directory
        current_dir = Path.cwd()
        logger.info(f"Current directory: {current_dir}")

        # Add the current directory to Python path
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))

        # Import domain models
        from models.domain_models import DataType, DeviceType, LocationType

        logger.info("Successfully imported domain models")

        return current_dir
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Current sys.path:")
        for path in sys.path:
            logger.error(f"  {path}")
        raise
    except Exception as e:
        logger.error(f"Error setting up import path: {e}")
        raise


def main():
    """Main function to generate and save the dataset."""
    try:
        # Setup import path and get domain models
        current_dir = setup_import_path()
        from models.domain_models import DataType, DeviceType, LocationType

        # Generate the dataset
        logger.info("Generating device data...")
        df = generate_device_data(num_samples=1000)

        # Save the dataset
        output_dir = current_dir / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save both train and test datasets
        train_file = output_dir / "train_dataset.csv"
        test_file = output_dir / "test_dataset.csv"

        # Split the dataset (80% train, 20% test)
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        logger.info(f"Train dataset saved to {train_file}")
        logger.info(f"Test dataset saved to {test_file}")

        # Print dataset statistics
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        logger.info(f"Features: {', '.join(df.columns)}")
        logger.info("\nRisk level distribution:")
        logger.info(df["risk_level"].value_counts().sort_index())

        return df

    except Exception as e:
        logger.error(f"Error in dataset generation: {e}")
        raise


def generate_device_data(num_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic device data for training the privacy risk classifier."""
    from models.domain_models import DataType, DeviceType, LocationType

    # Define possible values for categorical features
    device_types = [dt.value for dt in DeviceType]
    data_types = [dt.value for dt in DataType]
    location_types = [lt.value for lt in LocationType]

    # Generate random data
    data = {
        "device_id": [f"device_{i}" for i in range(num_samples)],
        "device_type": np.random.choice(device_types, num_samples),
        "data_type": np.random.choice(data_types, num_samples),
        "location_type": np.random.choice(location_types, num_samples),
        "access_frequency": np.random.randint(1, 100, num_samples),
        "user_consent": np.random.choice([True, False], num_samples, p=[0.7, 0.3]),
        "network_security_level": np.random.randint(1, 6, num_samples),
        "data_sensitivity": np.random.randint(1, 6, num_samples),
        "encryption_level": np.random.randint(1, 4, num_samples),
        "retention_period": np.random.randint(1, 365, num_samples),
        "data_volume": np.random.randint(1, 1000, num_samples),  # in MB
        "access_pattern": np.random.choice(
            ["regular", "irregular", "burst"], num_samples
        ),
        "last_audit_days": np.random.randint(1, 90, num_samples),
        "data_anonymization": np.random.choice(
            [True, False], num_samples, p=[0.6, 0.4]
        ),
        "data_pseudonymization": np.random.choice(
            [True, False], num_samples, p=[0.5, 0.5]
        ),
        "data_minimization": np.random.choice([True, False], num_samples, p=[0.7, 0.3]),
        "purpose_limitation": np.random.choice(
            [True, False], num_samples, p=[0.8, 0.2]
        ),
        "storage_duration": np.random.randint(1, 365, num_samples),
        "data_sharing": np.random.choice(
            ["none", "internal", "external"], num_samples, p=[0.4, 0.4, 0.2]
        ),
        "compliance_status": np.random.choice(
            ["compliant", "partially_compliant", "non_compliant"],
            num_samples,
            p=[0.6, 0.3, 0.1],
        ),
        "security_incidents": np.random.randint(0, 5, num_samples),
        "privacy_impact_assessment": np.random.choice(
            [True, False], num_samples, p=[0.7, 0.3]
        ),
        "data_protection_officer": np.random.choice(
            [True, False], num_samples, p=[0.8, 0.2]
        ),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate risk level based on features
    df["risk_level"] = calculate_risk_level(df)

    return df


def calculate_risk_level(df: pd.DataFrame) -> List[int]:
    """Calculate risk level based on various factors."""
    risk_levels = []

    for _, row in df.iterrows():
        risk_score = 0

        # Device type risk
        device_risk = {
            "camera": 4,
            "sensor": 2,
            "actuator": 3,
            "gateway": 4,
            "wearable": 5,
        }
        risk_score += device_risk[row["device_type"]]

        # Data type risk
        data_risk = {
            "location": 5,
            "video": 5,
            "audio": 4,
            "temperature": 1,
            "humidity": 1,
            "pressure": 1,
            "health": 5,
            "identification": 5,
        }
        risk_score += data_risk[row["data_type"]]

        # Location type risk
        location_risk = {
            "public_space": 3,
            "private_space": 4,
            "semi_public": 3,
            "restricted": 5,
        }
        risk_score += location_risk[row["location_type"]]

        # Other factors
        risk_score += 6 - row["network_security_level"]  # Inverse of security level
        risk_score += row["data_sensitivity"]
        risk_score += 4 - row["encryption_level"]  # Inverse of encryption level

        if not row["user_consent"]:
            risk_score += 2

        if row["access_pattern"] == "irregular":
            risk_score += 1
        elif row["access_pattern"] == "burst":
            risk_score += 2

        if row["last_audit_days"] > 30:
            risk_score += 1

        # New risk factors
        if not row["data_anonymization"]:
            risk_score += 2
        if not row["data_pseudonymization"]:
            risk_score += 1
        if not row["data_minimization"]:
            risk_score += 2
        if not row["purpose_limitation"]:
            risk_score += 2

        if row["data_sharing"] == "external":
            risk_score += 3
        elif row["data_sharing"] == "internal":
            risk_score += 1

        if row["compliance_status"] == "non_compliant":
            risk_score += 3
        elif row["compliance_status"] == "partially_compliant":
            risk_score += 1

        risk_score += row["security_incidents"]

        if not row["privacy_impact_assessment"]:
            risk_score += 2
        if not row["data_protection_officer"]:
            risk_score += 1

        # Normalize to 1-5 scale
        risk_level = min(5, max(1, int(risk_score / 8)))
        risk_levels.append(risk_level)

    return risk_levels


if __name__ == "__main__":
    try:
        df = main()
        print("\nDataset generation complete!")
        print(f"Total samples: {len(df)}")
        print(f"Features: {', '.join(df.columns)}")
        print("\nRisk level distribution:")
        print(df["risk_level"].value_counts().sort_index())
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
