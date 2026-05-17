import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.ingest import get_spark_session, load_and_cache, get_filing_as_dict
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


def run_pipeline():
    company = os.getenv("COMPANY_NAME", "AIG").upper()
    year = os.getenv("FILING_YEAR", "2019")
    raw_dir = os.getenv("RAW_DATA_DIR", "./data/raw")
    gt_path = os.getenv("GROUND_TRUTH_PATH", "./data/ground_truth.csv")
    results_path = os.getenv("RESULTS_PATH", "./data/results.csv")
    filing_type = os.getenv("FILING_TYPE", "10-K")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    top_k = int(os.getenv("TOP_K_CHUNKS", "3"))
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    extract_model = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    logger.info(f"=== Pipeline start: {company} {year} {filing_type} ===")

    spark = get_spark_session()
    parquet_path = load_and_cache(company, year, raw_dir)
    filing = get_filing_as_dict(parquet_path)

    chunks = extract_chunks_from_filing(filing, chunk_size=chunk_size)
    chunks_df = chunks_to_spark_df(chunks, spark)
    logger.info(f"Chunks in Spark: {chunks_df.count()} rows, columns={chunks_df.columns}")

    chunks = embed_chunks(chunks, client, model=embed_model)

    variables = list(VARIABLE_QUERIES.keys())
    retrieved = {
        var: retrieve_top_k(var, chunks, client, top_k=top_k, model=embed_model)
        for var in variables
    }

    extracted = extract_all_variables(variables, retrieved, client, model=extract_model, filing_type=filing_type)

    gt_df = pd.read_csv(gt_path)
    results_df = build_results(gt_df, extracted, company, year)

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved → {results_path}")

    print_summary(results_df)

    spark.stop()
    return results_df


if __name__ == "__main__":
    run_pipeline()
