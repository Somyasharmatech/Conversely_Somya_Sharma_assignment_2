"""Writes all extraction results to a JSON file."""

import json
from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


def write_json(results: List[dict], output_path: str) -> None:
    """
    Writes results list to a pretty-printed JSON file.

    Args:
        results: list of result dicts (one per chunk)
        output_path: file path to write (will be created/overwritten)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("JSON results written to: %s (%d records)", path, len(results))
    except Exception as exc:
        logger.error("Failed to write JSON to %s: %s", path, exc)
        raise
