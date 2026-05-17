import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# query + section scope + keywords per variable.
# keywords pre-filters to chunks that literally mention the variable before
# applying embedding similarity — critical for financial tables that have
# almost no natural language for embeddings to distinguish on.
VARIABLE_CONFIG = {
    "Total Revenues": {
        "query": "Total revenues condensed consolidating statements of income AIG consolidated parent subsidiary",
        "sections": ["section_7", "section_8"],
        "keywords": ["total revenues", "total revenue"],
    },
    "Net Income": {
        "query": "Net income attributable to AIG consolidated statements of income parent subsidiary Fortitude",
        "sections": ["section_7", "section_8"],
        "keywords": ["net income", "net loss"],
    },
    "Business Segment": {
        "query": "What are the operating sub-segments or divisions within each business segment, such as North America, International, Individual Retirement, Group Retirement, Life Insurance?",
        "sections": ["section_1", "section_7"],
        "keywords": [],
    },
}

VARIABLE_QUERIES = {k: v["query"] for k, v in VARIABLE_CONFIG.items()}


def _get_embeddings(texts, client, model):
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def embed_chunks(chunks, client, model="text-embedding-3-small"):
    batch_size = 100
    texts = [c["text"] for c in chunks]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(_get_embeddings(batch, client, model))
        logger.info(f"Embedded batch {i // batch_size + 1} ({len(batch)} chunks)")

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    return chunks


def retrieve_top_k(variable, chunks, client, top_k=3, model="text-embedding-3-small"):
    config = VARIABLE_CONFIG.get(variable, {})
    query = config.get("query", f"What is the {variable}?")
    sections = config.get("sections")
    keywords = config.get("keywords", [])

    search_pool = [c for c in chunks if c["section"] in sections] if sections else chunks
    if not search_pool:
        search_pool = chunks

    if keywords:
        keyword_pool = [
            c for c in search_pool
            if any(kw in c["text"].lower() for kw in keywords)
        ]
        if keyword_pool:
            search_pool = keyword_pool

    query_emb = np.array(_get_embeddings([query], client, model)[0]).reshape(1, -1)
    chunk_matrix = np.array([c["embedding"] for c in search_pool])

    scores = cosine_similarity(query_emb, chunk_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    logger.info(
        f"Top {top_k} chunks for '{variable}' "
        f"(sections={sections}, keyword_pool={len(search_pool)}) — "
        f"scores: {[round(float(s), 4) for s in scores[top_indices]]}"
    )

    return [search_pool[i] for i in top_indices]
