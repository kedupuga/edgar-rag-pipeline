import re
import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField

logger = logging.getLogger(__name__)

# section_7 = MD&A, section_8 = financial statements, section_1 = business overview
TARGET_SECTIONS = ["section_1", "section_7", "section_7A", "section_8"]


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
    schema = StructType([
        StructField("section", StringType(), True),
        StructField("chunk_id", StringType(), True),
        StructField("text", StringType(), True),
    ])
    return spark.createDataFrame(pd.DataFrame(chunks), schema=schema)
