"""
Gemini API client setup.
Reads GEMINI_API_KEY from the environment (via .env) and configures the SDK.
No API keys are ever hardcoded here.
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

# Load .env if present (ignored silently in production where env vars are set directly)
load_dotenv()

_MODEL_NAME = "gemini-1.5-flash"


def get_model() -> genai.GenerativeModel:
    """
    Initialises and returns a configured Gemini GenerativeModel instance.
    Raises EnvironmentError if GEMINI_API_KEY is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key, or set the environment variable directly."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=_MODEL_NAME,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,          # low temperature → consistent structured output
            max_output_tokens=1024,
        ),
    )
    logger.debug("Gemini model '%s' initialised.", _MODEL_NAME)
    return model
