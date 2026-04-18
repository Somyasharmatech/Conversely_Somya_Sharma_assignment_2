"""Ingests .txt and .pdf files, returning raw text with source metadata."""

from pathlib import Path
from typing import Optional

import chardet

from utils.logger import get_logger

logger = get_logger(__name__)


def ingest_file(file_path: str) -> Optional[dict]:
    """
    Reads a .txt or .pdf file and returns a dict with:
      - source: file path string
      - source_type: "txt" or "pdf"
      - raw_text: extracted text content

    Returns None if the file cannot be read, logging the error.
    """
    path = Path(file_path)

    if not path.exists():
        logger.error("File not found: %s", file_path)
        return None

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _read_txt(path)
    elif suffix == ".pdf":
        return _read_pdf(path)
    else:
        logger.error("Unsupported file type '%s' for file: %s", suffix, file_path)
        return None


def _read_txt(path: Path) -> Optional[dict]:
    """Read a plain-text file, auto-detecting encoding."""
    try:
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        logger.debug("Detected encoding '%s' for %s", encoding, path)

        text = raw_bytes.decode(encoding, errors="replace")
        logger.info("Ingested TXT file: %s (%d chars)", path.name, len(text))
        return {"source": str(path), "source_type": "txt", "raw_text": text}
    except Exception as exc:
        logger.error("Failed to read TXT file %s: %s", path, exc)
        return None


def _read_pdf(path: Path) -> Optional[dict]:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader  # imported here to keep the module optional-friendly

        reader = PdfReader(str(path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append(text)
            except Exception as page_exc:
                logger.warning("Could not extract page %d from %s: %s", i, path.name, page_exc)

        full_text = "\n".join(pages_text)
        logger.info(
            "Ingested PDF file: %s (%d pages, %d chars)", path.name, len(reader.pages), len(full_text)
        )
        return {"source": str(path), "source_type": "pdf", "raw_text": full_text}
    except Exception as exc:
        logger.error("Failed to read PDF file %s: %s", path, exc)
        return None
