"""
Groq API client setup.
Reads GROQ_API_KEY from the environment (via .env) and returns a configured client.
No API keys are ever hardcoded here.

Why Groq?
  - Free tier: 14,400 requests/day, 6,000 tokens/min with llama-3.3-70b-versatile
  - No billing setup required — works immediately after signup
  - Direct REST API via the official groq-python SDK (no orchestration framework)
"""

import os

from groq import Groq
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

# Load .env if present
load_dotenv()

_MODEL_NAME = "llama-3.3-70b-versatile"


def get_client() -> Groq:
    """
    Initialises and returns a configured Groq client.
    Raises EnvironmentError if GROQ_API_KEY is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key, or set the variable directly. "
            "Get a free key at https://console.groq.com/keys"
        )

    client = Groq(api_key=api_key)
    logger.debug("Groq client initialised with model '%s'.", _MODEL_NAME)
    return client


def get_model_name() -> str:
    return _MODEL_NAME
