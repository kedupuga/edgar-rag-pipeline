import os
import logging
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.ingest import get_spark_session, load_and_cache, load_filing_spark, get_filing_as_dict
from src.chunk import extract_chunks_from_filing, chunks_to_spark_df
from src.retrieve import VARIABLE_QUERIES, embed_chunks, retrieve_top_k
from src.extract import extract_all_variables
from src.evaluate import build_results, print_summary

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RAG pipeline for SEC filing extraction")
    parser.add_argument("--company", default=os.getenv("COMPANY_NAME", "AIG"))
    parser.add_argument("--year", default=os.getenv("FILING_YEAR", "2019"))
    parser.add_argument("--gt-path", default=os.getenv("GROUND_TRUTH_PATH", "./data/ground_truth.csv"))
    parser.add_argument("--results-path", default=os.getenv("RESULTS_PATH", "./data/results.csv"))
    return parser.parse_args()


def load_ground_truth(gt_path, company, year):
    if not Path(gt_path).exists():
        raise FileNotFoundError(
            f"Ground truth not found at '{gt_path}'. "
            f"Run: python generate_ground_truth.py --company {company} --year {year}"
        )
    gt_df = pd.read_csv(gt_path)
    filtered = gt_df[
        (gt_df["company"] == company) & (gt_df["year"].astype(str) == str(year))
    ].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No ground truth entries for {company} {year} in {gt_path}.")
    logger.info(f"Loaded {len(filtered)} ground truth rows for {company} {year}")
    return filtered


def run_pipeline(company, year, gt_path, results_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment or .env file.")

    client = OpenAI(api_key=api_key)
    filing_type = os.getenv("FILING_TYPE", "10-K")
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    extract_model = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k = int(os.getenv("TOP_K_CHUNKS", "3"))
    raw_dir = os.getenv("RAW_DATA_DIR", "./data/raw")

    logger.info(f"=== Pipeline start: {company} {year} {filing_type} ===")

    logger.info("Step 1/5 — Ingesting filing")
    parquet_path = load_and_cache(company, year, raw_dir)
    spark = get_spark_session()
    spark_df = load_filing_spark(parquet_path, spark)
    spark_df.show(1, truncate=80)
    filing = get_filing_as_dict(parquet_path)

    logger.info("Step 2/5 — Chunking sections")
    chunks = extract_chunks_from_filing(filing, chunk_size=chunk_size, overlap=chunk_overlap)
    chunks_spark_df = chunks_to_spark_df(chunks, spark)
    logger.info(f"Chunks Spark DF: {chunks_spark_df.count()} rows")

    logger.info("Step 3/5 — Embedding chunks")
    chunks = embed_chunks(chunks, client, model=embed_model)

    logger.info("Step 4/5 — Retrieving and extracting variables")
    variables = list(VARIABLE_QUERIES.keys())
    retrieved = {
        var: retrieve_top_k(var, chunks, client, top_k=top_k, model=embed_model)
        for var in variables
    }
    extracted = extract_all_variables(variables, retrieved, client, model=extract_model, filing_type=filing_type)

    logger.info("Step 5/5 — Evaluating against ground truth")
    gt_df = load_ground_truth(gt_path, company, year)
    results_df = build_results(gt_df, extracted, company, year)

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved → {results_path}")

    print_summary(results_df)
    spark.stop()
    return results_df


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        company=args.company.upper(),
        year=args.year,
        gt_path=args.gt_path,
        results_path=args.results_path,
    )
