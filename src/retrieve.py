import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

VARIABLE_QUERIES = {
    "Total Revenues": "What is the total revenue or total net revenues for the company?",
    "Net Income": "What is the net income or net loss attributable to the company?",
    "Business Segment": "What are the operating sub-segments or divisions within each business segment, such as North America, International, Individual Retirement, Group Retirement, Life Insurance?",
}


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
    query = VARIABLE_QUERIES.get(variable, f"What is the {variable}?")
    query_emb = np.array(_get_embeddings([query], client, model)[0]).reshape(1, -1)
    chunk_matrix = np.array([c["embedding"] for c in chunks])

    scores = cosine_similarity(query_emb, chunk_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    logger.info(
        f"Top {top_k} chunks for '{variable}' — "
        f"scores: {[round(float(s), 4) for s in scores[top_indices]]}"
    )

    return [chunks[i] for i in top_indices]
