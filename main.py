"""
LLM Integration & Data Pipeline — Main Entry Point

Usage:
    python main.py --file sample_inputs/sample.txt --urls sample_inputs/urls.txt --output output/

The pipeline:
  1. Ingest file (txt/pdf) and/or URLs
  2. Clean and chunk each source
  3. Call Gemini API for structured extraction on each chunk
  4. Store results as JSON, CSV, and a plain-text summary report
  5. Log all failures; never crash on a single bad input
"""

import argparse
import sys
from pathlib import Path
from typing import List

from ingestion.file_ingestor import ingest_file
from ingestion.url_ingestor import ingest_urls_from_file
from llm.client import get_model
from llm.extractor import extract_from_chunk
from preprocessing.cleaner import clean_and_chunk
from storage.csv_writer import write_csv
from storage.json_writer import write_json
from storage.report_writer import write_report
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Data Pipeline: ingest text/PDF files and URLs, extract structured data."
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Path to a .txt or .pdf file to process.",
    )
    parser.add_argument(
        "--urls",
        metavar="PATH",
        help="Path to a text file containing one URL per line.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default="output",
        help="Output directory for results (default: output/).",
    )
    args = parser.parse_args()

    if not args.file and not args.urls:
        parser.error("Provide at least one of --file or --urls.")

    return args


def run_pipeline(file_path: str = None, urls_path: str = None, output_dir: str = "output") -> None:
    logger.info("=" * 60)
    logger.info("Pipeline starting.")
    logger.info("  File   : %s", file_path or "none")
    logger.info("  URLs   : %s", urls_path or "none")
    logger.info("  Output : %s", output_dir)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Initialise LLM model (fail fast if API key missing)
    # ------------------------------------------------------------------
    try:
        model = get_model()
    except EnvironmentError as exc:
        logger.critical("Cannot start pipeline: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Ingest all sources
    # ------------------------------------------------------------------
    raw_documents: List[dict] = []

    if file_path:
        doc = ingest_file(file_path)
        if doc:
            raw_documents.append(doc)
        else:
            logger.warning("File ingestion failed for '%s'; skipping.", file_path)

    if urls_path:
        url_docs = ingest_urls_from_file(urls_path)
        raw_documents.extend(url_docs)

    if not raw_documents:
        logger.critical("No documents were successfully ingested. Exiting.")
        sys.exit(1)

    logger.info("Total sources ingested: %d", len(raw_documents))

    # ------------------------------------------------------------------
    # Step 3: Clean and chunk each document
    # ------------------------------------------------------------------
    all_chunks: List[dict] = []
    failed_sources: List[str] = []

    for doc in raw_documents:
        source = doc["source"]
        try:
            chunks = clean_and_chunk(doc["raw_text"], source)
            # Attach source_type to each chunk
            for chunk in chunks:
                chunk["source_type"] = doc.get("source_type", "unknown")
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("Chunking failed for '%s': %s — skipping.", source, exc)
            failed_sources.append(source)

    if not all_chunks:
        logger.critical("No chunks available for LLM processing. Exiting.")
        sys.exit(1)

    logger.info("Total chunks to process: %d", len(all_chunks))

    # ------------------------------------------------------------------
    # Step 4: Extract structured data from each chunk via LLM
    # ------------------------------------------------------------------
    results: List[dict] = []

    for i, chunk in enumerate(all_chunks):
        source = chunk["source"]
        chunk_idx = chunk["chunk_index"]
        logger.info(
            "Processing chunk %d/%d — source: '%s' (chunk %d/%d)",
            i + 1, len(all_chunks), source, chunk_idx + 1, chunk["total_chunks"],
        )
        try:
            result = extract_from_chunk(chunk, model)
            if result:
                results.append(result)
            else:
                logger.warning("No result for chunk %d from '%s'.", chunk_idx, source)
                if source not in failed_sources:
                    failed_sources.append(source)
        except Exception as exc:
            # extract_from_chunk already handles retries and logs; this is a safety net
            logger.error("Unexpected error processing chunk %d from '%s': %s", chunk_idx, source, exc)
            if source not in failed_sources:
                failed_sources.append(source)

    logger.info("Extraction complete: %d results, %d failed/skipped sources.", len(results), len(failed_sources))

    # ------------------------------------------------------------------
    # Step 5: Store results
    # ------------------------------------------------------------------
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if results:
        write_json(results, str(out / "results.json"))
        write_csv(results, str(out / "results.csv"))
        write_report(results, str(out / "summary_report.txt"), failed_sources=failed_sources)
    else:
        logger.warning("No results to store. Check pipeline.log for details.")

    logger.info("Pipeline finished. Outputs in: %s", out.resolve())


def main() -> None:
    args = parse_args()
    run_pipeline(
        file_path=args.file,
        urls_path=args.urls,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
