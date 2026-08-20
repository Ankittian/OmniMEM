"""answerer.py — LLM answer generation over retrieved context.

Each BEAM question category needs a different answering strategy:
  - abstention            → detect missing info and say so specifically
  - contradiction_resolution → detect and surface conflicting statements
  - event_ordering        → order events chronologically from context
  - information_extraction → extract specific facts
  - instruction_following → follow the formatting/style instruction stored in context
  - knowledge_update      → prefer the MOST RECENT value when values conflict
  - multi_session_reasoning → aggregate/count across all retrieved sessions
  - preference_following  → tailor answer to stored user preference
  - summarization         → synthesize a comprehensive narrative
  - temporal_reasoning    → compute durations/dates from facts in context
"""
from __future__ import annotations

import re
import time

from google.genai import types
from google.genai.errors import ClientError
from src.config import gemini, GEMINI_MODEL
from src.memory import should_abstain

# ---------------------------------------------------------------------------
# Per-category system prompts
# ---------------------------------------------------------------------------

_BASE = """You are a personal memory assistant with access to a user's past conversation history.
The CONTEXT below contains retrieved excerpts — treat them as ground truth.
The user is asking about their own history; answer in second person ("You worked on…", "Your sprint ends on…").
Do NOT hedge or say "based on our conversations" — give direct factual statements."""

_PROMPTS: dict[str, str] = {
    "abstention": _BASE + """

Your task: Determine whether the context actually answers the question.
- If the context genuinely contains the answer, state it directly.
- If the context mentions the topic but NOT the specific detail asked about, you MUST reply using EXACTLY this format (fill in the topic):
  "Based on the provided chat, there is no information related to [topic]."
- Do not fabricate details. Do not paraphrase the required format.""",

    "contradiction_resolution": _BASE + """

Your task: Look for CONTRADICTORY statements in the context about the same topic.
- If you find two conflicting claims, respond with:
  "I notice you've mentioned contradictory information about this. You said [statement A], but you also mentioned [statement B]. Could you clarify which is correct?"
- If there is no contradiction, answer normally.""",

    "event_ordering": _BASE + """

Your task: Order the events mentioned in the context chronologically.
- Use timestamps, session numbers, or logical sequence clues from the context.
- List each event in the exact order it occurred, numbered.""",

    "information_extraction": _BASE + """

Your task: Extract the specific fact(s) the question asks for.
- Be precise — dates, numbers, names, versions. Quote the context when helpful.
- If the context does not contain the answer, reply: "I don't have that information in our conversation history." """,

    "instruction_following": _BASE + """

Your task: Answer the question AND follow any formatting or style instruction mentioned in the context (e.g. always use syntax-highlighted code blocks, always include version numbers).
- Retrieve the relevant instruction from the context and apply it in your response.""",

    "knowledge_update": _BASE + """

Your task: The user's information may have been updated across sessions. 
- Always use the MOST RECENT value. If you see an original value and an updated value, report ONLY the updated one.
- State the final/latest value explicitly.""",

    "multi_session_reasoning": _BASE + """

Your task: Aggregate information spread across multiple sessions.
- Count, list, or combine facts from ALL retrieved excerpts.
- Be thorough — do not stop at the first match.""",

    "preference_following": _BASE + """

Your task: Answer the question while applying the user's stated preferences from the context.
- First identify the user's preference (e.g. "I prefer minimal dependencies").
- Then tailor your recommendation to match that preference explicitly.""",

    "summarization": _BASE + """

Your task: Write a comprehensive, chronological summary synthesizing ALL the retrieved excerpts.
- Cover: features implemented, timeline, security, and any other key themes present.
- Integrate details from every excerpt into a single coherent narrative.""",

    "temporal_reasoning": _BASE + """

Your task: Perform date/duration calculations using facts from the context.
- Extract the two time points mentioned, compute the duration, and state it clearly.
- Show your working (e.g. "March 29 → April 19 = 21 days").""",

    "default": _BASE + """

Answer DIRECTLY and CONCISELY.
If and ONLY IF the context contains absolutely no relevant information, reply: "I don't have that information in our conversation history." """,
}

_USER_TEMPLATE = """RETRIEVED CONVERSATION HISTORY (excerpts):
{context}

QUESTION: {question}

Answer directly from the context above:"""

_DEFAULT_WAIT = 60


def _parse_retry_after(message: str) -> int:
    m = re.search(r"retry in (\d+)", message, re.IGNORECASE)
    return int(m.group(1)) + 2 if m else _DEFAULT_WAIT


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """Call Gemini with automatic retry on 429 rate-limit errors."""
    while True:
        try:
            response = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                ),
            )
            return response.text.strip()
        except ClientError as e:
            if getattr(e, "status_code", None) == 429 or "429" in str(e):
                wait = _parse_retry_after(str(e))
                print(f"\n[rate limit] Gemini quota hit — waiting {wait}s before retry …")
                time.sleep(wait)
            else:
                raise


def _abstain_message(question: str) -> str:
    """Return the exact abstention phrasing the BEAM rubric expects."""
    # Rubric expects: "Based on the provided chat, there is no information related to [topic]."
    # Extract the core topic from the question heuristically (strip wh-words from the start).
    topic = question.rstrip("?").strip()
    # Lowercase wh-word prefixes to remove
    for prefix in ("how did", "how many", "how long", "how", "what is", "what are",
                   "what was", "what were", "what", "when did", "when was", "when",
                   "where did", "where", "who is", "who was", "who", "why did", "why",
                   "can you tell me about", "can you", "could you", "tell me about"):
        if topic.lower().startswith(prefix):
            topic = topic[len(prefix):].strip()
            break
    return f"Based on the provided chat, there is no information related to {topic}."


def generate_answer(
    context: str,
    question: str,
    confidence: float,
    category: str = "default",
) -> str:
    """Generate a category-aware answer grounded in retrieved context.

    Args:
        context:    Retrieved conversation excerpts (joined passages).
        question:   The probing question.
        confidence: Mean relevancy score from HydraDB.
        category:   BEAM question category (e.g. "abstention", "summarization").
    """
    if should_abstain(confidence) or not context.strip():
        return _abstain_message(question)

    system_prompt = _PROMPTS.get(category, _PROMPTS["default"])
    user_message = _USER_TEMPLATE.format(context=context, question=question)
    return _call_gemini(system_prompt, user_message)