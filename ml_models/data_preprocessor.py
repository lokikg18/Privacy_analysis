from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.target_column: str = "risk_level"
        self.model_path = Path("ml_models/saved_models/preprocessor.joblib")

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        categorical_columns = [
            "device_type",
            "data_type",
            "location_type",
            "access_pattern",
            "data_sharing",
            "compliance_status",
        ]

        df_encoded = df.copy()
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col])
            df_encoded[col] = self.label_encoders[col].transform(df[col])

        return df_encoded

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
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

        df_scaled = df.copy()
        df_scaled[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df_scaled

    def _convert_boolean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean features to numerical values."""
        boolean_columns = [
            "user_consent",
            "data_anonymization",
            "data_pseudonymization",
            "data_minimization",
            "purpose_limitation",
            "privacy_impact_assessment",
            "data_protection_officer",
        ]

        df_converted = df.copy()
        for col in boolean_columns:
            df_converted[col] = df_converted[col].astype(int)

        return df_converted

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the dataset for model training."""
        # Convert boolean features
        df_processed = self._convert_boolean_features(df)

        # Encode categorical features
        df_processed = self._encode_categorical_features(df_processed)

        # Scale numerical features
        df_processed = self._scale_numerical_features(df_processed)

        # Define feature columns
        self.feature_columns = [
            col for col in df_processed.columns if col != self.target_column
        ]

        # Split into features and target
        X = df_processed[self.feature_columns].values
        y = df_processed[self.target_column].values

        return X, y

    def preprocess_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess new data for prediction."""
        if not self.label_encoders or not self.feature_columns:
            raise ValueError("Preprocessor must be fitted with training data first")

        # Convert boolean features
        df_processed = self._convert_boolean_features(df)

        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            df_processed[col] = encoder.transform(df_processed[col])

        # Scale numerical features
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
        df_processed[numerical_columns] = self.scaler.transform(
            df_processed[numerical_columns]
        )

        return df_processed[self.feature_columns].values

    def save_preprocessor(self) -> None:
        """Save the preprocessor to disk."""
        preprocessor_data = {
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
        }
        joblib.dump(preprocessor_data, self.model_path)

    def load_preprocessor(self) -> None:
        """Load a preprocessor from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved preprocessor found at {self.model_path}")

        preprocessor_data = joblib.load(self.model_path)
        self.scaler = preprocessor_data["scaler"]
        self.label_encoders = preprocessor_data["label_encoders"]
        self.feature_columns = preprocessor_data["feature_columns"]
        self.target_column = preprocessor_data["target_column"]


def prepare_training_data(
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training and validation data."""
    # Load dataset
    train_df = pd.read_csv("data/processed/train_dataset.csv")

    # Initialize and fit preprocessor
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(train_df)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=42
    )

    # Save preprocessor
    preprocessor.save_preprocessor()

    return X_train, X_val, y_train, y_val


def prepare_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Prepare test data."""
    # Load dataset and preprocessor
    test_df = pd.read_csv("data/processed/test_dataset.csv")
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor()

    # Preprocess test data
    X_test = preprocessor.preprocess_new_data(test_df)
    y_test = test_df["risk_level"].values

    return X_test, y_test


if __name__ == "__main__":
    # Example usage
    print("Preparing training data...")
    X_train, X_val, y_train, y_val = prepare_training_data()

    print("Preparing test data...")
    X_test, y_test = prepare_test_data()

    print("\nData shapes:")
    print(f"Training features: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Validation labels: {y_val.shape}")
    print(f"Test features: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")
