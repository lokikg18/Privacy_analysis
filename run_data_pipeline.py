import logging
import os
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_current_directory():
    """Get the current directory."""
    try:
        current_dir = Path.cwd()
        logger.info(f"Current directory: {current_dir}")
        return current_dir
    except Exception as e:
        logger.error(f"Error getting current directory: {e}")
        raise


def setup_environment():
    """Setup the environment."""
    try:
        current_dir = get_current_directory()

        # Create necessary directories
        data_dir = current_dir / "data"
        processed_dir = data_dir / "processed"
        models_dir = current_dir / "models"

        for dir_path in [data_dir, processed_dir, models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

        return current_dir
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise


def run_pipeline():
    """Run the complete data pipeline."""
    try:
        current_dir = setup_environment()

        # Set PYTHONPATH for subprocesses
        env = os.environ.copy()
        env["PYTHONPATH"] = str(current_dir)

        # Run dataset generation
        logger.info("Generating dataset...")
        generate_script = current_dir / "generate_dataset.py"
        if not generate_script.exists():
            raise FileNotFoundError(
                f"Generate dataset script not found at {generate_script}"
            )

        # Use the current Python interpreter
        python_executable = sys.executable

        subprocess.run(
            [python_executable, str(generate_script)],
            check=True,
            env=env,
            cwd=str(current_dir),
        )

        # Run dataset analysis
        logger.info("Analyzing dataset...")
        analyze_script = current_dir / "analyze_dataset.py"
        if not analyze_script.exists():
            raise FileNotFoundError(
                f"Analyze dataset script not found at {analyze_script}"
            )

        subprocess.run(
            [python_executable, str(analyze_script)],
            check=True,
            env=env,
            cwd=str(current_dir),
        )

        logger.info("Data pipeline completed successfully!")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error in pipeline execution: {e}")
        logger.error(f"Command: {e.cmd}")
        logger.error(f"Return code: {e.returncode}")
        if e.output:
            logger.error(f"Output: {e.output.decode()}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr.decode()}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
