import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivacyRiskClassifier:
    def __init__(self, model_dir: str = "ml_models/saved_models"):
        """Initialize the Privacy Risk Classifier.

        Args:
            model_dir (str): Directory to save/load models
        """
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.label_encoders = {
            "device_type": LabelEncoder(),
            "data_type": LabelEncoder(),
            "location_type": LabelEncoder(),
        }

        # Create model directory if it doesn't exist
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "privacy_risk_model.joblib"
        logger.info(f"Model will be saved to: {self.model_path}")

    def _encode_features(self, X: List[Dict[str, str]]) -> np.ndarray:
        """Encode categorical features.

        Args:
            X (List[Dict]): List of feature dictionaries

        Returns:
            np.ndarray: Encoded features

        Raises:
            ValueError: If required features are missing
        """
        try:
            # First, collect all unique values for each categorical feature
            for feature in self.label_encoders.keys():
                unique_values = list(set(sample[feature] for sample in X))
                self.label_encoders[feature].fit(unique_values)

            # Then encode the features
            encoded = []
            for sample in X:
                encoded_sample = []
                for feature, encoder in self.label_encoders.items():
                    if feature in sample:
                        encoded_sample.append(encoder.transform([sample[feature]])[0])
                    else:
                        raise ValueError(f"Missing required feature: {feature}")

                # Add numerical features
                encoded_sample.extend(
                    [
                        sample.get("access_frequency", 0),
                        1 if sample.get("user_consent", False) else 0,
                        sample.get("network_security_level", 0),
                    ]
                )

                encoded.append(encoded_sample)

            return np.array(encoded)

        except Exception as e:
            logger.error(f"Error encoding features: {e}")
            raise

    def train(self, X: List[Dict[str, str]], y: List[int]) -> None:
        """Train the privacy risk classifier.

        Args:
            X (List[Dict]): List of feature dictionaries
            y (List[int]): List of risk levels (1-5)
        """
        X_encoded = self._encode_features(X)
        self.classifier.fit(X_encoded, y)

    def predict(self, X: List[Dict[str, str]]) -> List[int]:
        """Predict risk levels for new data.

        Args:
            X (List[Dict]): List of feature dictionaries

        Returns:
            List[int]: Predicted risk levels
        """
        X_encoded = self._encode_features(X)
        return self.classifier.predict(X_encoded)

    def predict_proba(self, X: List[Dict[str, str]]) -> List[List[float]]:
        """Predict risk level probabilities for new data.

        Args:
            X (List[Dict]): List of feature dictionaries

        Returns:
            List[List[float]]: Predicted probabilities for each risk level
        """
        X_encoded = self._encode_features(X)
        return self.classifier.predict_proba(X_encoded)

    def save_model(self) -> None:
        """Save the trained model and encoders."""
        try:
            model_data = {
                "classifier": self.classifier,
                "label_encoders": self.label_encoders,
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self) -> None:
        """Load the trained model and encoders."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            model_data = joblib.load(self.model_path)
            self.classifier = model_data["classifier"]
            self.label_encoders = model_data["label_encoders"]
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    classifier = PrivacyRiskClassifier()

    # Example training data
    training_data = [
        {
            "device_type": "camera",
            "data_type": "video",
            "location_type": "public_space",
            "access_frequency": 10,
            "user_consent": True,
            "network_security_level": 3,
        },
        {
            "device_type": "sensor",
            "data_type": "temperature",
            "location_type": "private_space",
            "access_frequency": 5,
            "user_consent": False,
            "network_security_level": 2,
        },
    ]

    # Example labels (risk levels)
    labels = [4, 2]  # High risk for camera, low risk for temperature sensor

    # Train the model
    classifier.train(training_data, labels)

    # Make predictions
    new_data = [
        {
            "device_type": "camera",
            "data_type": "video",
            "location_type": "private_space",
            "access_frequency": 15,
            "user_consent": False,
            "network_security_level": 1,
        }
    ]

    risk_level = classifier.predict(new_data)
    risk_probabilities = classifier.predict_proba(new_data)

    print(f"Predicted Risk Level: {risk_level}")
    print(f"Risk Level Probabilities: {risk_probabilities}")

    # Save the model
    classifier.save_model()
