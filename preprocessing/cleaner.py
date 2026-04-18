"""
Text cleaning and token-aware chunking for LLM context limits.

Steps applied to every raw text:
  1. Normalize encoding / replace non-printable characters
  2. Remove boilerplate patterns (excessive whitespace, repeated lines)
  3. Split into token-bounded chunks with overlap
"""

import re
import unicodedata
from typing import List

import tiktoken

from utils.logger import get_logger

logger = get_logger(__name__)

# Gemini flash supports ~1M tokens, but we target small chunks for quality extraction
_MAX_TOKENS = 2000
_OVERLAP_TOKENS = 200
_ENCODING_NAME = "cl100k_base"  # compatible token counting for most modern LLMs


def clean_and_chunk(raw_text: str, source: str) -> List[dict]:
    """
    Cleans raw_text and splits it into chunks.

    Returns a list of chunk dicts:
      {
        "source": str,
        "chunk_index": int,
        "total_chunks": int,
        "text": str,
        "token_count": int,
      }
    """
    cleaned = _clean(raw_text)
    if not cleaned.strip():
        logger.warning("Text from '%s' is empty after cleaning — skipping.", source)
        return []

    chunks = _chunk(cleaned)
    logger.info("'%s' -> %d chunk(s) after cleaning.", source, len(chunks))

    return [
        {
            "source": source,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk_text,
            "token_count": _count_tokens(chunk_text),
        }
        for i, chunk_text in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Applies all cleaning steps sequentially."""
    text = _normalize_unicode(text)
    text = _remove_control_characters(text)
    text = _collapse_whitespace(text)
    text = _remove_repeated_lines(text)
    return text.strip()


def _normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC and replace common broken encodings."""
    # Normalize unicode form
    text = unicodedata.normalize("NFC", text)
    # Replace Windows smart quotes and dashes with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u00a0": " ",  # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _remove_control_characters(text: str) -> str:
    """Remove non-printable control characters except newlines and tabs."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple blank lines to at most two, and multiple spaces to one."""
    # Collapse inline whitespace (spaces/tabs) to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _remove_repeated_lines(text: str, threshold: int = 3) -> str:
    """Remove lines that appear more than `threshold` times (boilerplate indicator)."""
    lines = text.split("\n")
    from collections import Counter
    line_counts = Counter(line.strip() for line in lines if line.strip())

    filtered = [
        line for line in lines
        if not line.strip() or line_counts[line.strip()] < threshold
    ]
    return "\n".join(filtered)


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding)."""
    try:
        enc = tiktoken.get_encoding(_ENCODING_NAME)
        return len(enc.encode(text))
    except Exception as exc:
        logger.warning("Token counting failed: %s — estimating by word count.", exc)
        return len(text.split())


def _chunk(text: str) -> List[str]:
    """
    Splits text into chunks of at most MAX_TOKENS tokens,
    with OVERLAP_TOKENS of overlap between adjacent chunks.
    """
    try:
        enc = tiktoken.get_encoding(_ENCODING_NAME)
        tokens = enc.encode(text)
    except Exception as exc:
        logger.warning("tiktoken encoding failed (%s); falling back to word-based chunking.", exc)
        return _word_chunk(text)

    if len(tokens) <= _MAX_TOKENS:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + _MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - _OVERLAP_TOKENS  # overlap

    return chunks


def _word_chunk(text: str) -> List[str]:
    """Fallback chunker that splits by approximate word count."""
    words = text.split()
    approx_words_per_chunk = _MAX_TOKENS  # rough 1:1 token-to-word ratio
    chunks = []
    for i in range(0, len(words), approx_words_per_chunk):
        chunks.append(" ".join(words[i: i + approx_words_per_chunk]))
    return chunks or [text]
