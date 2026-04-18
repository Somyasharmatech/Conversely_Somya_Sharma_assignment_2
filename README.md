# LLM Integration & Data Pipeline

A modular Python CLI application that ingests unstructured text from `.txt`/`.pdf` files and URLs, processes it through the **Groq API** (llama-3.3-70b-versatile), and outputs structured JSON, CSV/Excel, and a plain-text summary report.

---

## Features

- **Multi-source ingestion**: `.txt` files, `.pdf` files, and web URLs in a single run
- **Smart preprocessing**: encoding repair, boilerplate removal, token-aware chunking (tiktoken)
- **Structured LLM extraction** via Groq API (no LangChain):
  - 2–3 sentence summary
  - Named entities (people, places, organizations)
  - Sentiment with confidence score (0.0–1.0)
  - 3 important questions the text raises
- **Robust error handling**: retry with exponential backoff (tenacity), malformed JSON regex repair, per-input failure isolation — pipeline never crashes on a single bad input
- **Three output formats**: `results.json` + `results.csv` + `results.xlsx` + `summary_report.txt`
- **Full logging** to both console and `pipeline.log`

---

## LLM Choice: Groq API (llama-3.3-70b-versatile)

- **Free tier**: 14,400 requests/day, 6,000 tokens/min — no billing required
- **JSON mode** (`response_format={"type": "json_object"}`) enforces structured output
- **Low latency**: Groq's LPU hardware processes tokens ~10× faster than typical LLM APIs
- **No regional restrictions**: works reliably worldwide

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/Somyasharmatech/Conversely_Somya_Sharma_assignment_2.git
cd Conversely_Somya_Sharma_assignment_2
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and fill in your Groq API key:

```bash
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...
```

Get a free key at: **https://console.groq.com/keys** (sign in with Google, no credit card)

---

## How to Run

```bash
python main.py --file sample_inputs/sample.txt --urls sample_inputs/urls.txt --output output/
```

### Arguments

| Argument | Description |
|---|---|
| `--file` | Path to a `.txt` or `.pdf` file |
| `--urls` | Path to a text file with one URL per line |
| `--output` | Output directory (default: `output/`) |
| `--max-chunks` | Max chunks per source (default: `3`; use `0` for unlimited) |

### Example — run on sample inputs

```bash
python main.py --file sample_inputs/sample.txt --urls sample_inputs/urls.txt
```

Outputs written to `output/`:
- `results.json` — all extracted data (one object per chunk)
- `results.csv` — flat table, one row per chunk (Excel-compatible UTF-8)
- `results.xlsx` — same data as Excel workbook
- `summary_report.txt` — aggregated findings across all inputs

Logs written to `pipeline.log`.

---

## Project Structure

```
├── main.py                    # CLI entry point & pipeline orchestrator
├── ingestion/
│   ├── file_ingestor.py       # .txt (chardet auto-encoding) + .pdf (pypdf)
│   └── url_ingestor.py        # URL fetch + HTML parsing (requests + BeautifulSoup)
├── preprocessing/
│   └── cleaner.py             # Unicode normalization, boilerplate removal, tiktoken chunking
├── llm/
│   ├── client.py              # Groq client setup (reads GROQ_API_KEY from env only)
│   └── extractor.py           # Prompt, JSON extraction, retry logic, JSON repair
├── storage/
│   ├── json_writer.py         # Writes results.json
│   ├── csv_writer.py          # Writes results.csv + results.xlsx (pandas)
│   └── report_writer.py       # Writes summary_report.txt
├── utils/
│   └── logger.py              # Dual logging (console + pipeline.log)
├── sample_inputs/
│   ├── sample.txt             # Sample AI/technology article
│   └── urls.txt               # 2 Wikipedia URLs + 1 broken URL (for failure demo)
├── sample_outputs/            # Real outputs from a pipeline run
│   ├── results.json
│   ├── results.csv
│   └── summary_report.txt
├── .env.example
└── requirements.txt
```

---

## Design Decisions

1. **No orchestration libraries**: All API calls use the `groq` SDK directly — no LangChain, LlamaIndex, etc.
2. **Tenacity for retries**: Exponential backoff (2→4→8→16→32s, capped at 60s) with 5 max attempts. Retries on all `Exception` subclasses covering rate limits, timeouts, and server errors.
3. **Malformed JSON handling**: First attempts `json.loads()`, then falls back to regex extraction of the JSON block (`{...}`), then logs and skips the chunk. Never crashes.
4. **Pipeline isolation**: Each chunk is wrapped in its own `try/except`. One bad input never stops the rest.
5. **Token-aware chunking**: `tiktoken` (cl100k_base) counts tokens; chunks target ≤ 2,000 tokens with 200-token overlap to maintain context.
6. **Encoding repair**: `chardet` detects file encoding; unicode normalised to NFC; control characters removed; smart quotes replaced.
7. **Rate-limit guard**: 5-second sleep between LLM requests to stay comfortably under the 30 req/min per-project limit.

---

## Tested Inputs

| Input | Type | Result |
|---|---|---|
| `sample_inputs/sample.txt` | TXT file | 1 chunk, extracted successfully |
| `https://en.wikipedia.org/wiki/Artificial_intelligence` | URL | 26 chunks detected, 3 processed |
| `https://en.wikipedia.org/wiki/Machine_learning` | URL | 15 chunks detected, 3 processed |
| `https://this-url-does-not-exist-xyz-12345.com/broken` | Broken URL | Logged as ERROR, skipped gracefully |

---

## Known Limitations

- PDF extraction may miss text in scanned/image-based PDFs (no OCR support)
- Very large files are chunked but each chunk is processed independently (no cross-chunk memory)
- JavaScript-rendered pages are not supported (uses `requests`, not a headless browser)
- Free tier: 30 req/min on Groq; the 5s inter-chunk delay keeps the pipeline within limits
