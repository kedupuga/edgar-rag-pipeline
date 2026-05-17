# AIG RAG Pipeline — SEC 10-K Data Extraction

A Retrieval Augmented Generation (RAG) pipeline that extracts financial variables from SEC EDGAR 10-K filings using PySpark, OpenAI embeddings, and GPT-4o.

## Overview

The pipeline reads a single company's annual 10-K filing from the EDGAR corpus, chunks the relevant sections, retrieves the most semantically relevant passages for each target variable, and uses an LLM to extract the values. Results are compared against an auto-generated ground truth.

**Target variables:**
- `Total Revenues` — numeric
- `Net Income` — numeric  
- `Business Segment` — categorical

