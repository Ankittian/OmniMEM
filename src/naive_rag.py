"""naive_rag.py — Naive RAG baseline using pure TF-IDF + cosine similarity.

This intentionally replicates what a *basic* RAG system would do:
  1. Split chat sessions into fixed-size text chunks.
  2. At query time, embed everything with TF-IDF vectors (no graph, no
     hybrid index, no cross-session context graph).
  3. Return the top-k chunks ranked by cosine similarity.
  4. Feed them straight to Gemini with a single generic system prompt
     (NO category-aware prompting, NO multi-query recall, NO abstention logic).

This is the "naive baseline" that HydraGraph Memory is compared against.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

from google.genai import types
from google.genai.errors import ClientError
from src.config import gemini, GEMINI_MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80      # character overlap between consecutive chunks
TOP_K = 10              # number of chunks to retrieve

_NAIVE_SYSTEM_PROMPT = """You are a helpful assistant. Use the provided conversation excerpts to answer the question.
Answer based on the context given. If the answer cannot be found, say you don't know."""

_USER_TEMPLATE = """CONVERSATION HISTORY (excerpts):
{context}

QUESTION: {question}

Answer:"""

_DEFAULT_WAIT = 60


# ---------------------------------------------------------------------------
# Simple in-memory TF-IDF retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9']+", text.lower())


class NaiveRetriever:
    """A minimal TF-IDF retriever backed by an in-memory inverted index."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._idf: dict[str, float] = {}

    def add_texts(self, texts: list[str]) -> None:
        """Add a list of text chunks to the index."""
        for text in texts:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            for term in tf:
                self._df[term] += 1
            self._tf.append(tf)
            self._chunks.append(text)

        n = len(self._chunks)
        self._idf = {
            term: math.log((n + 1) / (df + 1)) + 1.0
            for term, df in self._df.items()
        }

    def _score(self, tf: Counter, query_tokens: list[str]) -> float:
        score = 0.0
        for token in query_tokens:
            idf = self._idf.get(token, 0.0)
            score += tf.get(token, 0) * idf
        return score

    def query(self, text: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        """Return top-k (chunk, score) tuples sorted by descending score."""
        if not self._chunks:
            return []
        q_tokens = _tokenize(text)
        scored = [
            (chunk, self._score(tf, q_tokens))
            for chunk, tf in zip(self._chunks, self._tf)
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def clear(self) -> None:
        self._chunks.clear()
        self._tf.clear()
        self._df.clear()
        self._idf.clear()


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


def build_index(sessions: list[list[dict]], timestamps: list[str]) -> "NaiveRetriever":
    """Convert raw chat sessions into a searchable TF-IDF index."""
    retriever = NaiveRetriever()
    all_chunks: list[str] = []

    for session_turns, ts in zip(sessions, timestamps):
        text = "\n".join(
            f"[{turn['role'].upper()}]: {turn['content']}"
            for turn in session_turns
        )
        if ts:
            text = f"[Date: {ts}]\n{text}"

        for chunk in _chunk_text(text):
            all_chunks.append(chunk)

    retriever.add_texts(all_chunks)
    return retriever


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

def recall(retriever: "NaiveRetriever", query: str, top_k: int = TOP_K) -> tuple[str, float]:
    """Retrieve context from the naive index.

    Returns:
        (context_text, confidence) — confidence is the normalised mean score.
    """
    results = retriever.query(query, top_k=top_k)
    if not results:
        return "", 0.0

    passages = [chunk for chunk, _ in results]
    scores = [score for _, score in results]

    max_score = max(scores) if scores else 1.0
    norm_scores = [s / max_score if max_score > 0 else 0.0 for s in scores]
    confidence = sum(norm_scores) / len(norm_scores)

    context = "\n\n---\n\n".join(passages)
    return context, confidence


# ---------------------------------------------------------------------------
# Answer generation — generic, no category awareness
# ---------------------------------------------------------------------------

def _parse_retry_after(message: str) -> int:
    m = re.search(r"retry in (\d+)", message, re.IGNORECASE)
    return int(m.group(1)) + 2 if m else _DEFAULT_WAIT


def generate_answer(context: str, question: str) -> str:
    """Generate an answer using a single generic system prompt (no category logic).

    The naive baseline — always generates, never abstains, no category routing.
    """
    if not context.strip():
        context = "[No relevant context found]"

    user_message = _USER_TEMPLATE.format(context=context, question=question)

    while True:
        try:
            response = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=_NAIVE_SYSTEM_PROMPT,
                    temperature=0,
                ),
            )
            return response.text.strip()
        except ClientError as e:
            if getattr(e, "status_code", None) == 429 or "429" in str(e):
                wait = _parse_retry_after(str(e))
                print(f"\n[rate limit] Gemini quota hit — waiting {wait}s …")
                time.sleep(wait)
            else:
                raise
