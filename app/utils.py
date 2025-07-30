import logging

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)