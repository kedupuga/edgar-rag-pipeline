# EDGAR RAG Pipeline

A Retrieval Augmented Generation (RAG) pipeline that extracts financial variables from SEC EDGAR filings using PySpark, OpenAI embeddings, and GPT-4o-mini. Extracted values are evaluated against a manually prepared ground truth.

## Pipeline

```
  Inputs
  ──────
  company + year  →  SEC EDGAR API (CIK lookup)
  ground_truth.csv  (provided manually)

  main.py
  ───────
  Step 1: Ingest    — download filing from HuggingFace EDGAR corpus, cache as parquet
       ↓
  Step 2: Chunk     — split sections into word-based chunks, load into PySpark DataFrame
       ↓
  Step 3: Embed     — embed all chunks via OpenAI (text-embedding-3-small, batches of 100)
       ↓
  Step 4: Retrieve + Extract
                    — cosine similarity to find top-k chunks per variable
                    — GPT-4o-mini extracts the value from retrieved chunks
       ↓
  Step 5: Evaluate  — compare extracted values against ground truth
                      (exact match after normalizing commas, currency symbols, and case)
       ↓
  data/results.csv  +  terminal summary
```

**Target variables**
- `Total Revenues` — numeric
- `Net Income` — numeric
- `Business Segment` — categorical

## Prerequisites

- Python 3.9+
- **Java 8** — required for PySpark 3.5.x. PySpark 4.x requires Java 17
- An OpenAI API key

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
```

Edit `.env` and set your `OPENAI_API_KEY`. All other values have sensible defaults.

## Usage

```bash
# Run with defaults from .env
python main.py

# Override company/year at the command line
python main.py --company MSFT --year 2018
python main.py --company AIG --year 2019
```

The ground truth CSV must exist at `GROUND_TRUTH_PATH` (default: `./data/ground_truth.csv`) before running. Expected columns: `company`, `year`, `variable`, `value`.

## Configuration

All settings are read from `.env`. CLI flags (`--company`, `--year`) override `.env` at runtime.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `COMPANY_NAME` | `AIG` | Ticker symbol of the company |
| `FILING_YEAR` | `2019` | Fiscal year of the filing |
| `FILING_TYPE` | `10-K` | Filing type — used in LLM prompts |
| `RAW_DATA_DIR` | `./data/raw` | Where to cache downloaded parquet files |
| `GROUND_TRUTH_PATH` | `./data/ground_truth.csv` | Manually prepared ground truth (pipeline input) |
| `RESULTS_PATH` | `./data/results.csv` | Evaluation results (pipeline output) |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap in words between consecutive chunks |
| `TOP_K_CHUNKS` | `3` | Chunks retrieved per variable |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EXTRACTION_MODEL` | `gpt-4o-mini` | OpenAI chat model for extraction |

## Project Structure

```
edgar-rag-pipeline/
├── main.py                     # Orchestrates the full pipeline
├── requirements.txt
├── .env.example
└── src/
    ├── ingest.py               # CIK lookup, HuggingFace download, Spark session
    ├── chunk.py                # Word-based chunking with overlap, PySpark DataFrame
    ├── retrieve.py             # OpenAI embeddings, cosine similarity retrieval
    ├── extract.py              # GPT-4o-mini extraction via structured JSON prompt
    └── evaluate.py             # Exact match scoring after value normalization
```

## Output

`data/results.csv` columns:

| Column | Description |
|---|---|
| `company` | Ticker symbol |
| `year` | Filing year |
| `variable` | Variable name |
| `ground_truth` | Value from ground truth CSV |
| `extracted_value` | Value extracted by the RAG pipeline |
| `match` | `True` if values match within tolerance |
| `context` | Quote from the filing text where the value was found |

The terminal summary shows PASS/FAIL per observation and an overall accuracy percentage.
