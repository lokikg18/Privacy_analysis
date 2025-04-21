# wsgi.py
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("wsgi.py: Starting import...")
try:
    # Assuming dashboard.py is in the same directory (root)
    from dashboard import app
    server = app.server
    logger.info("wsgi.py: Successfully imported app and server.")
except ImportError as e:
    logger.error(f"wsgi.py: Failed to import 'dashboard' or 'app'. Error: {e}", exc_info=True)
    raise  # Re-raise the exception to make the failure clear
except AttributeError as e:
    logger.error(f"wsgi.py: Found 'app' but failed to access 'app.server'. Error: {e}", exc_info=True)
    raise
except Exception as e:
    logger.error(f"wsgi.py: An unexpected error occurred during import. Error: {e}", exc_info=True)
    raise

logger.info("wsgi.py: Import complete. Server object ready.")