# EDGAR RAG Pipeline

A Retrieval Augmented Generation (RAG) pipeline that extracts financial variables from SEC EDGAR 10-K filings using PySpark, OpenAI embeddings, and GPT-4o-mini. Extracted values are evaluated against a manually prepared ground truth dataset.

## Pipeline

```
  Inputs
  ──────
  company + year     →  SEC EDGAR API (CIK lookup)
  ground_truth.csv      (manually prepared — see Ground Truth section)

  main.py
  ───────
  Step 1: Ingest    — download filing from HuggingFace EDGAR corpus, cache as parquet
       ↓
  Step 2: Chunk     — split sections into 500-word chunks with overlap, load into PySpark DataFrame
       ↓
  Step 3: Embed     — embed all chunks via OpenAI text-embedding-3-small (batches of 100)
       ↓
  Step 4: Retrieve + Extract
                    — per variable: section filter → keyword pre-filter → cosine similarity top-k
                    — GPT-4o-mini extracts ALL distinct values from retrieved chunks (JSON)
       ↓
  Step 5: Evaluate  — exact match after normalizing commas, currency symbols, and unit words
       ↓
  data/results.csv  +  terminal summary
```

## Variables

The pipeline extracts 2 numeric and 1 categorical variable per the assignment spec.

| Variable | Type | Description |
|---|---|---|
| `Total Revenues` | Numeric | Consolidated and entity-level total revenues from the filing |
| `Net Income` | Numeric | Net income figures across the consolidated statements and notes |
| `Business Segment` | Categorical | Operating sub-segments reported by the company |

## Ground Truth

Ground truth is prepared manually and stored in `data/ground_truth.csv`. The dataset has **15 observations** — 5 per variable — all sourced from a single filing (AIG 2019 10-K), consistent with the one-document constraint.


## Prerequisites

- Python 3.9+
- An OpenAI API key

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env          # Windows
```

Edit `.env` and set your `OPENAI_API_KEY`. All other values have sensible defaults.

## Usage

```bash
# Run with defaults from .env
python main.py

# Override company and year at runtime
python main.py --company AIG --year 2019
```

The ground truth CSV must exist at `GROUND_TRUTH_PATH` before running. Expected schema: `company`, `year`, `variable`, `value`.

## Configuration

All settings are read from `.env`. CLI flags (`--company`, `--year`) override `.env` at runtime.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `COMPANY_NAME` | `AIG` | Company ticker used for CIK lookup |
| `FILING_YEAR` | `2019` | Fiscal year of the 10-K filing |
| `FILING_TYPE` | `10-K` | Filing type — passed to LLM prompts |
| `RAW_DATA_DIR` | `./data/raw` | Where to cache downloaded parquet files |
| `GROUND_TRUTH_PATH` | `./data/ground_truth.csv` | Manually prepared ground truth (input) |
| `RESULTS_PATH` | `./data/results.csv` | Evaluation results (output) |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap in words between consecutive chunks |
| `TOP_K_CHUNKS` | `15` | Chunks retrieved per variable for extraction |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EXTRACTION_MODEL` | `gpt-4o-mini` | OpenAI chat model for extraction |

## Project Structure

```
edgar-rag-pipeline/
├── main.py                     # Orchestrates the full 5-step pipeline
├── requirements.txt
├── .env.example
├── data/
│   ├── ground_truth.csv        # Manually prepared (pipeline input)
│   ├── results.csv             # Extraction + evaluation output
│   └── raw/                    # Cached parquet files (auto-created)
└── src/
    ├── ingest.py               # CIK lookup, HuggingFace download, Spark session
    ├── chunk.py                # Word-based chunking with overlap, PySpark DataFrame
    ├── retrieve.py             # Embeddings, keyword pre-filter, cosine similarity
    ├── extract.py              # GPT-4o-mini extraction via structured JSON prompt
    └── evaluate.py             # Exact match scoring after value normalization
```

## Output

`data/results.csv` schema:

| Column | Description |
|---|---|
| `company` | Company ticker |
| `year` | Filing year |
| `variable` | Variable name (`Total Revenues`, `Net Income`, `Business Segment`) |
| `ground_truth` | Expected value from ground truth CSV |
| `extracted_values` | All values extracted by the pipeline (comma-separated) |
| `match` | `True` if any extracted value matches ground truth after normalization |

The terminal summary prints PASS/FAIL per row and an overall accuracy percentage.
