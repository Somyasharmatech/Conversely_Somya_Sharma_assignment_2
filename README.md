# LLM Integration & Data Pipeline

A modular Python CLI application that ingests unstructured text from `.txt`/`.pdf` files and URLs, processes it through Google Gemini, and outputs structured JSON, CSV, and a plain-text summary report.

---

## Features

- **Multi-source ingestion**: `.txt` files, `.pdf` files, and web URLs in a single run
- **Smart preprocessing**: encoding repair, boilerplate removal, token-aware chunking
- **Structured LLM extraction** via Gemini API (no LangChain):
  - 2–3 sentence summary
  - Named entities (people, places, organizations)
  - Sentiment with confidence score
  - 3 important questions the text raises
- **Robust error handling**: retry with exponential backoff (tenacity), malformed JSON repair, per-input failure isolation
- **Dual output**: `results.json` + `results.csv/xlsx` + `summary_report.txt`
- **Full logging** to both console and `pipeline.log`

---

## LLM Choice: Google Gemini (gemini-1.5-flash)

- **Free tier** available via `google-generativeai` SDK
- **Native JSON mode** (`response_mime_type="application/json"`) makes structured output highly reliable
- **Large context window** (1M tokens) minimises unnecessary chunking
- **Speed**: flash variant is fast enough for pipeline throughput

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/Somyasharmatech/Conversely_Somya_Sharma_assignment_2.git
cd Conversely_Somya_Sharma_assignment_2
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and fill in your Gemini API key:

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=<your key>
```

Get a free API key at: https://aistudio.google.com/

---

## How to Run

```bash
python main.py --file sample_inputs/sample.txt --urls sample_inputs/urls.txt --output output/
```

### Arguments

| Argument | Description |
|---|---|
| `--file` | Path to a `.txt` or `.pdf` file to process |
| `--urls` | Path to a text file containing one URL per line |
| `--output` | Directory to write output files (default: `output/`) |

### Example

```bash
python main.py --file sample_inputs/sample.txt --urls sample_inputs/urls.txt
```

Outputs written to `output/`:
- `results.json` — all extracted data
- `results.csv` — one row per chunk (Excel-compatible)
- `summary_report.txt` — aggregated findings

Logs written to `pipeline.log`.

---

## Project Structure

```
├── main.py                    # CLI entry point & pipeline orchestrator
├── ingestion/
│   ├── file_ingestor.py       # .txt and .pdf ingestion
│   └── url_ingestor.py        # URL fetch + HTML parsing
├── preprocessing/
│   └── cleaner.py             # Text cleaning + token-aware chunking
├── llm/
│   ├── client.py              # Gemini API client (reads key from env)
│   └── extractor.py           # Prompt, structured extraction, retry logic
├── storage/
│   ├── json_writer.py         # Writes results.json
│   ├── csv_writer.py          # Writes results.csv via pandas
│   └── report_writer.py       # Writes summary_report.txt
├── utils/
│   └── logger.py              # Logging configuration
├── sample_inputs/
│   ├── sample.txt             # Sample document
│   └── urls.txt               # Sample URL list
├── sample_outputs/            # Pre-generated sample outputs
│   ├── results.json
│   ├── results.csv
│   └── summary_report.txt
├── .env.example
└── requirements.txt
```

---

## Design Decisions

1. **No orchestration libraries**: All API calls use the `google-generativeai` SDK directly.
2. **Tenacity for retries**: Exponential backoff (1–60s) with 5 attempts. Retries on `Exception` subclasses that indicate transient failures (rate limits, timeouts, server errors).
3. **Malformed JSON handling**: First attempts `json.loads()`, then falls back to regex extraction of the JSON block, then logs and skips the chunk.
4. **Pipeline isolation**: Each chunk/URL is wrapped in its own try/except. A failure in one never crashes the whole pipeline.
5. **Token-aware chunking**: `tiktoken` (cl100k_base) counts tokens; chunks target ≤2000 tokens with 200-token overlap.
6. **Encoding repair**: `chardet` detects file encoding; HTML entities and non-ASCII are normalized before processing.

---

## Tested Inputs

- `sample_inputs/sample.txt` — multi-topic article about AI and technology
- `sample_inputs/urls.txt` containing:
  - `https://en.wikipedia.org/wiki/Artificial_intelligence`
  - `https://www.bbc.com/news/technology`
  - `https://broken-url-that-does-not-exist.xyz` (intentionally broken to test failure handling)

---

## Known Limitations

- PDF extraction may lose formatting from scanned/image-based PDFs (no OCR)
- Very large files are chunked but each chunk is processed independently (no cross-chunk context)
- Rate limits depend on your Gemini API tier; free tier allows ~15 req/min
- JavaScript-rendered web pages are not supported (uses `requests`, not a headless browser)
