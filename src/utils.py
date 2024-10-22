# src/utils.py

import os
import logging
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    return True

def configure_logging(log_level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
