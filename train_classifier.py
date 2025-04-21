import logging
import sys
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

        return current_dir
    except Exception as e:
        logger.error(f"Error setting up import path: {e}")
        raise


def convert_to_dict_format(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of dictionaries format expected by the classifier."""
    return df.to_dict("records")


def main():
    try:
        # Setup import path
        setup_import_path()

        # Import required modules
        from ml_models.data_preprocessor import prepare_training_data
        from ml_models.privacy_risk_classifier import PrivacyRiskClassifier

        # Load and prepare training data
        logger.info("Loading and preparing training data...")
        train_df = pd.read_csv("data/processed/train_dataset.csv")

        # Convert DataFrame to list of dictionaries
        X_train_list = convert_to_dict_format(train_df)
        y_train = train_df["risk_level"].values

        # Initialize and train classifier
        logger.info("Initializing classifier...")
        classifier = PrivacyRiskClassifier()

        logger.info("Training classifier...")
        classifier.train(X_train_list, y_train)

        # Save the model
        logger.info("Saving trained model...")
        classifier.save_model()

        # Load validation data
        logger.info("Loading validation data...")
        val_df = pd.read_csv("data/processed/test_dataset.csv")
        X_val_list = convert_to_dict_format(val_df)
        y_val = val_df["risk_level"].values

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        y_pred = classifier.predict(X_val_list)
        accuracy = np.mean(y_pred == y_val)
        logger.info(f"Validation accuracy: {accuracy:.4f}")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
