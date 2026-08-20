"""memory.py — HydraDB ingestion and recall."""
from __future__ import annotations

import json
import time
from typing import Optional

from hydra_db.core.api_error import ApiError
from src.config import hydra

# Confidence threshold below which we treat the answer as unknown.
ABSTAIN_THRESHOLD = 0.10

def setup_tenant(database_name: str) -> bool:
    """Create the database in HydraDB and wait for it to be ready.
    
    Returns:
        True if the database was newly created, False if it already existed.
    """
    is_new = True
    try:
        hydra.databases.create(database=database_name)
    except ApiError as e:
        if getattr(e, "status_code", None) == 409 or "409" in str(e):
            is_new = False
        else:
            raise

    while True:
        status_resp = hydra.databases.status(database=database_name)
        if status_resp and status_resp.data and status_resp.data.infra and status_resp.data.infra.ready_for_ingestion:
            break
        time.sleep(2)
        
    return is_new

def get_ingested_collections(database_name: str) -> list[str]:
    """Return a list of collections (chats) already ingested in this database."""
    try:
        resp = hydra.databases.collections(database=database_name)
        if resp and resp.data and resp.data.collections:
            return resp.data.collections
    except Exception:
        pass
    return []

def ingest_sessions(
    database_name: str,
    collection_name: str,
    sessions: list[list[dict]],
    timestamps: list[str],
) -> None:
    """Ingest a list of chat sessions into HydraDB for a given tenant.

    Args:
        database_name: Database name.
        collection_name: Unique identifier for this conversation.
        sessions: List of sessions; each session is a list of turn dicts
                  with "role" and "content" keys.
        timestamps: Parallel list of date strings, one per session.
    """
    source_ids = []
    for session_turns, ts in zip(sessions, timestamps):
        # Flatten a session into a readable transcript string.
        text = "\n".join(
            f"[{turn['role'].upper()}]: {turn['content']}"
            for turn in session_turns
        )
        resp = hydra.context.ingest(
            database=database_name,
            collection=collection_name,
            documents=("session.txt", text.encode("utf-8")),
            document_metadata=json.dumps([{"additional_metadata": {"timestamp": ts}}]),
            type="knowledge",
        )
        if resp and resp.data and resp.data.results:
            source_ids.append(resp.data.results[0].id)

    # Wait for processing to complete
    if source_ids:
        while True:
            status_resp = hydra.context.status(database=database_name, collection=collection_name, ids=source_ids)
            if status_resp and status_resp.data and status_resp.data.statuses:
                all_done = all(s.indexing_status in ["completed", "failed"] for s in status_resp.data.statuses)
                if all_done:
                    break
            time.sleep(2)



def _query_once(database_name: str, collection_name: str, query: str, alpha: float) -> list[tuple[str, float]]:
    """Run a single HydraDB query and return (text, score) pairs."""
    result = hydra.query(
        database=database_name,
        collection=collection_name,
        query=query,
        alpha=alpha,
        graph_context=True,
        type="knowledge",
        max_results=10,
        num_related_chunks=3,
    )
    data = getattr(result, "data", None)
    if not data or not getattr(data, "chunks", None):
        return []
    out = []
    for chunk in data.chunks:
        text = getattr(chunk, "chunk_content", "")
        score = float(getattr(chunk, "relevancy_score", 0.0))
        if text:
            out.append((text, score))
    return out


def recall(
    database_name: str,
    collection_name: str,
    query: str,
    alpha: float = 0.5,
    extra_queries: list[str] | None = None,
) -> tuple[str, float]:
    """Retrieve relevant context for a query from HydraDB.

    For complex multi-session questions, pass additional sub-queries via
    ``extra_queries`` to gather broader context across sessions.

    Returns:
        (context_text, confidence) — context_text is empty string when
        nothing useful was found, confidence is 0.0–1.0.
    """
    seen: set[str] = set()
    passages: list[str] = []
    scores: list[float] = []

    for q in [query] + (extra_queries or []):
        for text, score in _query_once(database_name, collection_name, q, alpha):
            if text not in seen:
                seen.add(text)
                passages.append(text)
                scores.append(score)

    if not passages:
        return "", 0.0

    context = "\n\n---\n\n".join(passages)
    confidence = sum(scores) / len(scores)
    return context, confidence


def should_abstain(confidence: float) -> bool:
    """Return True when retrieval confidence is too low to answer reliably."""
    return confidence < ABSTAIN_THRESHOLD

