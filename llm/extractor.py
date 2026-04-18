"""
Structured extraction from text chunks using the Groq API (llama-3.3-70b-versatile).

For each chunk, asks the LLM to return a JSON object with:
  - summary       : 2-3 sentence summary of the text
  - entities      : {people: [], places: [], organizations: []}
  - sentiment     : {label: "positive"|"neutral"|"negative", confidence: 0.0-1.0}
  - questions     : [str, str, str]  — 3 important questions the text raises

Retry strategy (via tenacity):
  - Up to 5 attempts
  - Exponential backoff: 2 -> 4 -> 8 -> 16 -> 32 seconds (capped at 60s)
  - Retries on any Exception (covers rate limits, timeouts, server errors)
  - Malformed JSON is caught and repaired via regex; if unfixable, chunk is skipped.
"""

import json
import re
from typing import Optional

from groq import Groq
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm.client import get_model_name
from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL = get_model_name()

# -------------------------------------------------------------------------
# Prompt template
# -------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a precise information extraction assistant. "
    "You always respond with valid JSON only — no markdown fences, no extra text, no explanation."
)

_USER_TEMPLATE = """Analyse the following text and return ONLY a valid JSON object with exactly these keys:

{{
  "summary": "<2 to 3 sentence summary of the text>",
  "entities": {{
    "people": ["<name>", ...],
    "places": ["<name>", ...],
    "organizations": ["<name>", ...]
  }},
  "sentiment": {{
    "label": "<positive|neutral|negative>",
    "confidence": <float between 0.0 and 1.0>
  }},
  "questions": [
    "<question 1>",
    "<question 2>",
    "<question 3>"
  ]
}}

TEXT:
\"\"\"
{text}
\"\"\"
"""


# -------------------------------------------------------------------------
# Retry-decorated API call
# -------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, log_level=20),  # 20 = logging.INFO
    reraise=True,
)
def _call_api(client: Groq, prompt: str) -> str:
    """
    Calls the Groq API and returns the raw text response.
    Tenacity handles retries — do NOT catch exceptions here.
    """
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,        # low temperature -> consistent structured output
        max_tokens=1024,
        response_format={"type": "json_object"},  # Groq JSON mode
    )
    return response.choices[0].message.content


# -------------------------------------------------------------------------
# Public extraction function
# -------------------------------------------------------------------------
def extract_from_chunk(chunk: dict, client: Groq) -> Optional[dict]:
    """
    Extracts structured information from a single chunk dict.

    Args:
        chunk : dict with at least "text", "source", "chunk_index", "total_chunks"
        client: configured Groq client instance

    Returns:
        A result dict merging chunk metadata with extracted fields, or None on failure.
    """
    source = chunk.get("source", "unknown")
    chunk_index = chunk.get("chunk_index", 0)
    text = chunk.get("text", "")

    if not text.strip():
        logger.warning("Empty text for chunk %d from '%s' -- skipping.", chunk_index, source)
        return None

    prompt = _USER_TEMPLATE.format(text=text[:6000])  # hard safety trim

    raw_json: Optional[str] = None
    try:
        raw_json = _call_api(client, prompt)
        logger.debug(
            "Raw LLM response for chunk %d from '%s': %s",
            chunk_index, source, raw_json[:200],
        )
    except Exception as exc:
        logger.error(
            "Groq API permanently failed for chunk %d from '%s' after retries: %s",
            chunk_index, source, exc,
        )
        return None

    extracted = _parse_json(raw_json, source, chunk_index)
    if extracted is None:
        return None

    extracted = _normalise(extracted)

    return {
        "source": source,
        "source_type": chunk.get("source_type", "unknown"),
        "chunk_index": chunk_index,
        "total_chunks": chunk.get("total_chunks", 1),
        "token_count": chunk.get("token_count", 0),
        **extracted,
    }


# -------------------------------------------------------------------------
# JSON parsing with fallback repair
# -------------------------------------------------------------------------
def _parse_json(raw: str, source: str, chunk_index: int) -> Optional[dict]:
    """
    Attempts to parse raw LLM output as JSON.
    Tries direct parse first, then regex extraction of a JSON block.
    Returns None + logs error if both attempts fail.
    """
    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.debug(
            "Direct JSON parse failed for chunk %d from '%s'; trying regex repair.",
            chunk_index, source,
        )

    # Attempt 2: extract the first {...} block via regex
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning(
                "Regex-extracted JSON also invalid for chunk %d from '%s': %s",
                chunk_index, source, exc,
            )

    logger.error(
        "Could not parse JSON from LLM response for chunk %d from '%s'. Skipping.\nRaw: %s",
        chunk_index, source, raw[:500],
    )
    return None


# -------------------------------------------------------------------------
# Structure normalisation
# -------------------------------------------------------------------------
def _normalise(data: dict) -> dict:
    """Ensures all required keys exist with sensible defaults."""
    if not isinstance(data.get("summary"), str):
        data["summary"] = ""

    entities = data.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}
    data["entities"] = {
        "people": _ensure_str_list(entities.get("people")),
        "places": _ensure_str_list(entities.get("places")),
        "organizations": _ensure_str_list(entities.get("organizations")),
    }

    sentiment = data.get("sentiment", {})
    if not isinstance(sentiment, dict):
        sentiment = {}
    label = sentiment.get("label", "neutral")
    if label not in ("positive", "neutral", "negative"):
        label = "neutral"
    confidence = sentiment.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    data["sentiment"] = {"label": label, "confidence": confidence}

    data["questions"] = _ensure_str_list(data.get("questions"))[:3]

    return data


def _ensure_str_list(value) -> list:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []
