"""Writes all extraction results to a CSV (and Excel) file using pandas."""

from pathlib import Path
from typing import List

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def write_csv(results: List[dict], output_path: str) -> None:
    """
    Flattens nested result dicts and writes to CSV + Excel.
    One row per chunk.

    Nested fields are flattened:
      - entities.people / entities.places / entities.organizations → comma-separated strings
      - sentiment.label / sentiment.confidence → separate columns
      - questions → three separate columns
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_flatten(r) for r in results]

    if not rows:
        logger.warning("No results to write to CSV.")
        return

    df = pd.DataFrame(rows)

    # Write CSV
    csv_path = path
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compat
        logger.info("CSV results written to: %s (%d rows)", csv_path, len(df))
    except Exception as exc:
        logger.error("Failed to write CSV to %s: %s", csv_path, exc)
        raise

    # Write Excel alongside the CSV
    excel_path = path.with_suffix(".xlsx")
    try:
        df.to_excel(excel_path, index=False, engine="openpyxl")
        logger.info("Excel results written to: %s", excel_path)
    except Exception as exc:
        logger.error("Failed to write Excel to %s: %s", excel_path, exc)
        # Excel failure is non-fatal — CSV already written


def _flatten(result: dict) -> dict:
    """Flattens a nested result dict into a flat dict suitable for a DataFrame row."""
    entities = result.get("entities", {})
    sentiment = result.get("sentiment", {})
    questions = result.get("questions", [])

    # Pad questions list to always have 3 entries
    while len(questions) < 3:
        questions.append("")

    return {
        "source": result.get("source", ""),
        "source_type": result.get("source_type", ""),
        "chunk_index": result.get("chunk_index", 0),
        "total_chunks": result.get("total_chunks", 1),
        "token_count": result.get("token_count", 0),
        "summary": result.get("summary", ""),
        "entities_people": ", ".join(entities.get("people", [])),
        "entities_places": ", ".join(entities.get("places", [])),
        "entities_organizations": ", ".join(entities.get("organizations", [])),
        "sentiment_label": sentiment.get("label", ""),
        "sentiment_confidence": sentiment.get("confidence", 0.0),
        "question_1": questions[0],
        "question_2": questions[1],
        "question_3": questions[2],
    }
