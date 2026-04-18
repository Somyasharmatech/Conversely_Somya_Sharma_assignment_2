"""Generates a plain-text aggregated summary report across all pipeline results."""

from collections import Counter
from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


def write_report(results: List[dict], output_path: str, failed_sources: List[str] = None) -> None:
    """
    Writes a plain-text summary report aggregating findings across all chunks.

    Args:
        results: list of extraction result dicts
        output_path: path to write the report text file
        failed_sources: list of source identifiers that failed processing
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    failed_sources = failed_sources or []

    report_lines = _build_report(results, failed_sources)

    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report_lines))
        logger.info("Summary report written to: %s", path)
    except Exception as exc:
        logger.error("Failed to write summary report to %s: %s", path, exc)
        raise


def _build_report(results: List[dict], failed_sources: List[str]) -> List[str]:
    lines = []
    sep = "=" * 72

    lines.append(sep)
    lines.append("  LLM DATA PIPELINE — SUMMARY REPORT")
    lines.append(sep)
    lines.append("")

    # ---- High-level stats ------------------------------------------------
    total_chunks = len(results)
    unique_sources = sorted(set(r["source"] for r in results))
    sentiment_counts = Counter(r["sentiment"]["label"] for r in results)

    all_people = []
    all_places = []
    all_orgs = []
    for r in results:
        entities = r.get("entities", {})
        all_people.extend(entities.get("people", []))
        all_places.extend(entities.get("places", []))
        all_orgs.extend(entities.get("organizations", []))

    top_people = [name for name, _ in Counter(all_people).most_common(10)]
    top_places = [name for name, _ in Counter(all_places).most_common(10)]
    top_orgs = [name for name, _ in Counter(all_orgs).most_common(10)]

    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total sources processed : {len(unique_sources)}")
    lines.append(f"Total chunks extracted  : {total_chunks}")
    lines.append(f"Failed sources          : {len(failed_sources)}")
    lines.append("")

    lines.append("SENTIMENT DISTRIBUTION")
    lines.append("-" * 40)
    for label in ("positive", "neutral", "negative"):
        count = sentiment_counts.get(label, 0)
        pct = (count / total_chunks * 100) if total_chunks else 0
        lines.append(f"  {label.capitalize():<10}: {count:>4} chunks  ({pct:.1f}%)")
    lines.append("")

    lines.append("TOP ENTITIES (across all sources)")
    lines.append("-" * 40)
    lines.append(f"  People        : {', '.join(top_people) or 'none detected'}")
    lines.append(f"  Places        : {', '.join(top_places) or 'none detected'}")
    lines.append(f"  Organizations : {', '.join(top_orgs) or 'none detected'}")
    lines.append("")

    # ---- Per-source summaries --------------------------------------------
    lines.append(sep)
    lines.append("PER-SOURCE DETAILS")
    lines.append(sep)
    lines.append("")

    for source in unique_sources:
        source_chunks = [r for r in results if r["source"] == source]
        lines.append(f"SOURCE: {source}")
        lines.append(f"  Type   : {source_chunks[0].get('source_type', 'unknown')}")
        lines.append(f"  Chunks : {len(source_chunks)}")

        # Combined sentiment across chunks
        src_sentiment = Counter(r["sentiment"]["label"] for r in source_chunks)
        dominant = src_sentiment.most_common(1)[0][0]
        lines.append(f"  Dominant sentiment: {dominant}")

        # First chunk summary as the representative
        lines.append(f"  Summary (chunk 0):")
        summary = source_chunks[0].get("summary", "").strip()
        for sline in _wrap(summary, width=68, indent="    "):
            lines.append(sline)

        # Questions from first chunk
        questions = source_chunks[0].get("questions", [])
        if questions:
            lines.append(f"  Key questions:")
            for i, q in enumerate(questions, 1):
                for qline in _wrap(f"{i}. {q}", width=68, indent="     "):
                    lines.append(qline)

        lines.append("")

    # ---- Failed sources --------------------------------------------------
    if failed_sources:
        lines.append(sep)
        lines.append("FAILED / SKIPPED SOURCES")
        lines.append("-" * 40)
        for src in failed_sources:
            lines.append(f"  [SKIPPED] {src}")
        lines.append("")

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    return lines


def _wrap(text: str, width: int = 72, indent: str = "") -> List[str]:
    """Very simple word-wrap helper."""
    if not text:
        return [indent]
    words = text.split()
    lines = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = indent + word
        else:
            current = (current + " " + word).lstrip()
            current = indent + current.lstrip()
    if current.strip():
        lines.append(current)
    return lines or [indent]
