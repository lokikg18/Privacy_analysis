# wsgi.py
import logging
import sys

# Basic logging setup to see output
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("wsgi.py: Attempting to import from dashboard...")
try:
    # Import the app instance from your existing dashboard file
    from dashboard import app

    # Explicitly get the server attribute
    server = app.server
    logger.info("wsgi.py: Successfully imported 'app' and accessed 'app.server'.")

except ImportError as e:
    logger.error(f"wsgi.py: Failed to import 'dashboard' or find 'app'. Error: {e}", exc_info=True)
    raise
except AttributeError as e:
    logger.error(f"wsgi.py: Imported 'app' but failed to access 'app.server'. Error: {e}", exc_info=True)
    raise
except Exception as e:
    logger.error(f"wsgi.py: An unexpected error occurred during import/access. Error: {e}", exc_info=True)
    raise

logger.info("wsgi.py: 'server' object is ready for Gunicorn.")
