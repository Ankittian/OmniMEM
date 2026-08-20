# HydraGraph Memory - 3 Minute Demo Script

## 0:00 - 0:45 | The Problem
**Visual:** Slide or whiteboard graphic showing a standard RAG pipeline failing.
**Speaker:** 
"Hi everyone, today we're presenting HydraGraph Memory. When building AI agents, we ran into a massive problem: standard Retrieval-Augmented Generation, or RAG, fails at long-term memory. If a user tells an agent 'I'm using Flask' in Chat 1, and then 'I switched to FastAPI' in Chat 20, a standard semantic vector search just pulls whichever chunk is semantically closer. It fails at temporal reasoning, it fails at contradiction resolution, and it struggles to pull context spread across dozens of sessions. The agent just gets confused or hallucinates."

## 0:45 - 1:30 | What We Built
**Visual:** Show the architecture diagram from the README (Ingestion -> Recall Stage -> Generate Answer Stage).
**Speaker:** 
"To solve this, we built HydraGraph Memory. It's a memory layer designed to handle cross-session continuity across 30 to 40 sessions. We implemented two core innovations. First, **Multi-Query Recall**: instead of a single vector search, our system fires broad fallback queries to pull in diverse chunks simultaneously. Second, **Category-Aware Prompting**: we use Gemini Flash-Lite as a router that deploys specialized system prompts based on the question type. It explicitly forces the LLM to resolve contradictions, update old knowledge, follow preferences, or safely abstain if the answer isn't in the database."

## 1:30 - 2:15 | The Demo
**Visual:** Screen recording of the terminal. 
**Speaker:** 
"Let's see it in action against the BEAM benchmark. Here, I'm running our inference script, `run_beam.py`. You can see the agent ingesting raw chat sessions. Next, we throw a temporal contradiction at it. Watch as it successfully uses Multi-Query Recall to fetch the conflicting facts, realizes which one is more recent, and outputs the correct updated fact.
Because we didn't want to rely on costly closed-source judges, we also built a custom Gemini-powered evaluator. We run `evaluate_beam_gemini.py`, and finally, `score_report.py` generates a beautiful dashboard showing our high accuracy across all reasoning categories."

## 2:15 - 3:00 | Why HydraDB Matters
**Visual:** Show code snippets of `src/memory.py` interacting with HydraDB, or a visual representation of a Graph + Vector structure.
**Speaker:** 
"None of this would be possible without HydraDB. We used HydraDB as the backbone of our system because its hybrid Graph and Vector Store architecture is perfectly suited for complex, long-term memory. While a standard vector database just gives you isolated chunks based on semantic similarity, HydraDB allows us to maintain the relational continuity of facts over time. It gave us the ranking accuracy and contextual depth needed to build a production-ready agent memory. Thank you!"
