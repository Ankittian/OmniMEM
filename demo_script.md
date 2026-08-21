# HydraGraph Memory - 3 Minute Demo Script  
*(Updated: includes Naive RAG side-by-side comparison)*

---

## 0:00 - 0:30 | The Problem

**Visual:** Slide or whiteboard graphic showing a standard RAG pipeline failing.

**Speaker:**
"Hi everyone, today we're presenting HydraGraph Memory. When building AI agents, we ran into a massive problem: standard Retrieval-Augmented Generation — or RAG — fails at long-term memory. If a user tells an agent 'I'm using Flask' in Chat 1, and then 'I switched to FastAPI' in Chat 20, a standard TF-IDF or vector search just pulls whichever chunk looks most relevant. It fails at temporal reasoning, contradiction resolution, and multi-session aggregation. The agent either hallucinates or gives you the wrong, outdated fact."

---

## 0:30 - 1:00 | What We Built

**Visual:** Architecture diagram from the README (Ingestion → Recall Stage → Generate Answer Stage).

**Speaker:**
"To solve this, we built HydraGraph Memory on top of HydraDB. It has two core innovations over a naive RAG baseline.
First, **Multi-Query Recall**: instead of one lookup, our system fires multiple targeted sub-queries simultaneously and deduplicates results — pulling in diverse facts from different sessions.
Second, **Category-Aware Prompting**: we route each question to a specialized system prompt. A contradiction question gets a prompt that explicitly forces the LLM to surface the conflict. A temporal question gets one that does date arithmetic. An abstention question triggers a 'no information' response instead of hallucinating."

---

## 1:00 - 1:45 | The Side-by-Side Demo

**Visual:** Split terminal — left window is HydraGraph Memory, right is Naive RAG.

**Speaker:**
"Let's prove it with real data. I'm running `run_beam_compare.py` on the BEAM benchmark — the same set of 30 chat sessions with 10 reasoning categories.

Watch what happens on this **knowledge_update** question: 'What framework is the user currently using?'  
The naive RAG on the right retrieves the first semantically-close chunk — it's from Session 1, which says 'Flask'. It confidently returns 'Flask'.  
HydraGraph Memory on the left uses its graph-enhanced hybrid index to surface the *most recent* session, finds 'FastAPI', and returns the correct updated answer.

Now the **abstention** test: 'What is the user's favourite colour?'  
Naive RAG hallucinates a plausible answer because it has no abstention logic.  
HydraGraph correctly says: 'Based on the provided chat, there is no information related to the user's favourite colour.'

Finally, the **multi_session_reasoning** question: 'How many features were shipped across all sprints?'  
Naive RAG's top-k chunks happen to miss three sessions — it under-counts.  
Our multi-query recall fires five targeted sub-queries, de-duplicates across all 30 sessions, and counts correctly."

---

## 1:45 - 2:20 | The Numbers

**Visual:** Rich comparison table from `run_beam_compare.py` in the terminal.

**Speaker:**
"After the run completes, we get this comparison table. You can see HydraGraph Memory wins on every single reasoning category.  
The delta is largest on:
- **Knowledge Update** and **Temporal Reasoning** — where graph-ordered context is essential.
- **Abstention** — where naive RAG hallucinates instead of saying 'I don't know'.
- **Multi-Session Reasoning** — where a single-query approach simply misses facts.

Overall, HydraGraph Memory scores significantly higher than the naive RAG baseline — on the same LLM, same hardware, same data."

---

## 2:20 - 3:00 | Why HydraDB Matters

**Visual:** Code snippets of `src/memory.py` querying HydraDB with `graph_context=True`.

**Speaker:**
"None of this is possible with a plain vector store. HydraDB's hybrid Graph + Vector architecture lets us maintain *relational continuity* between facts over time — not just semantic proximity. The `graph_context=True` flag in our query activates the graph traversal layer, which is what enables cross-session reasoning and temporal ordering that a flat embedding index simply cannot do.

To summarise: naive RAG is a flat lookup. HydraGraph Memory is a time-aware, reasoning-aware, graph-enhanced memory system. Thank you — we're happy to take questions."

---

## Demo Commands (cheat sheet)

```bash
# 1. Run both systems on chats 0-2 for a quick demo
python run_beam_compare.py --size 100K --start 0 --end 3

# 2. Or run them separately
python run_beam.py              --size 100K --start 0 --end 3
python run_beam_naive_rag.py   --size 100K --start 0 --end 3

# 3. Evaluate separately with Gemini judge
python evaluate_beam_gemini.py --size 100K                               # HydraGraph
python evaluate_beam_gemini.py --size 100K --results-dir results/beam-naive/100K   # Naive RAG

# 4. Score report
python score_report.py --size 100K
```
