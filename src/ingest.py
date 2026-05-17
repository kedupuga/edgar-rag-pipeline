import os
import logging
from pathlib import Path

import requests
import pandas as pd
import datasets
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def get_spark_session(app_name="edgar_rag_pipeline"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def get_cik(ticker):
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": "edgar-rag-pipeline research@example.com"},
        timeout=10,
    )
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            logger.info(f"  {ticker} → CIK {cik}")
            return cik
    raise ValueError(
        f"'{ticker}' not found in SEC EDGAR. Pass the stock ticker (AIG, MSFT, etc.), not the full company name."
    )


def load_and_cache(company, year, raw_data_dir):
    out_path = Path(raw_data_dir) / f"{company.lower()}_{year}.parquet"

    if out_path.exists():
        logger.info(f"Using cached filing: {out_path}")
        return str(out_path)

    cik = get_cik(company)
    logger.info(f"Downloading EDGAR corpus year={year} from HuggingFace...")

    dataset = datasets.load_dataset("eloukas/edgar-corpus", f"year_{year}", split="train")

    df = dataset.to_pandas()
    filing_df = df[df["cik"].astype(str).str.strip() == cik]

    if filing_df.empty:
        filing_df = df[df["cik"].astype(str).str.strip() == str(int(cik))]

    if filing_df.empty:
        raise ValueError(
            f"No 10-K found for {company} (CIK {cik}) in {year}. "
            "Check the ticker or try a different year."
        )

    Path(raw_data_dir).mkdir(parents=True, exist_ok=True)
    filing_df.to_parquet(out_path, index=False)
    logger.info(f"Cached → {out_path}")

    return str(out_path)


def load_filing_spark(parquet_path, spark):
    df = spark.read.parquet(parquet_path)
    logger.info(f"Loaded into Spark: {df.count()} row(s), {len(df.columns)} columns")
    return df


def get_filing_as_dict(parquet_path):
    return pd.read_parquet(parquet_path).iloc[0].to_dict()
