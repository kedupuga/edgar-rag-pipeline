import re
import os
import logging
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField

logger = logging.getLogger(__name__)

# section_7 = MD&A, section_8 = financial statements, section_1 = business overview
TARGET_SECTIONS = ["section_1", "section_7", "section_7A", "section_8"]

_CHUNKS_TMP = os.path.abspath("./data/raw/chunks_tmp.parquet")

_CHUNKS_SCHEMA = StructType([
    StructField("section", StringType(), True),
    StructField("chunk_id", StringType(), True),
    StructField("text", StringType(), True),
])


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        # overlap ensures a value near a chunk boundary appears in both chunks
        start += chunk_size - overlap

    return chunks


def extract_chunks_from_filing(filing, chunk_size=500, overlap=50):
    all_chunks = []

    for section in TARGET_SECTIONS:
        text = clean_text(filing.get(section) or "")
        if len(text) < 200:
            continue

        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append({
                "section": section,
                "chunk_id": f"{section}_{i}",
                "text": chunk,
            })

    logger.info(f"Extracted {len(all_chunks)} chunks from {len(TARGET_SECTIONS)} sections")
    return all_chunks


def chunks_to_spark_df(chunks, spark: SparkSession):
    # PySpark 3.5 + Python 3.12+ have a cloudpickle incompatibility that breaks
    # createDataFrame when passing Python objects directly. Writing via pandas and
    # reading back with spark.read bypasses that serialization path entirely.
    Path(_CHUNKS_TMP).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(chunks).to_parquet(_CHUNKS_TMP, index=False)
    return spark.read.schema(_CHUNKS_SCHEMA).parquet(_CHUNKS_TMP)
