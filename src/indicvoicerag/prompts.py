"""Prompt construction for the RAG generator.

The generator is instructed to answer ONLY from the delimited retrieved
context and to refuse rather than invent facts.
"""

from __future__ import annotations

from typing import Any

from .context import CONTEXT_BEGIN, CONTEXT_END


REFUSAL_MARKER = "REFUSED"
CONFIDENCE_MARKER = "CONFIDENCE:"

SYSTEM_PROMPT = """You are a grounded question-answering assistant.
You answer questions in the language of the user's question.

Hard rules:
1. Use ONLY the retrieved context delimited below as the source of facts.
2. Do NOT use prior knowledge. Do NOT invent facts, names, dates, numbers, or explanations.
3. If the retrieved context does not contain enough information to answer the
   question, reply with exactly: {marker}: <brief reason in the user's language>
4. Keep the answer concise and directly responsive to the question.
5. Quote or closely paraphrase the relevant context. Do not add unsupported detail.""".format(marker=REFUSAL_MARKER)

STRICT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You previously produced an answer that could not be grounded in the context.
Now re-answer obeying these stricter rules:
- Restate ONLY content that appears in the retrieved context, near-verbatim.
- If you cannot answer strictly from the context, reply with exactly:
  {marker}: <brief reason in the user's language>""".format(marker=REFUSAL_MARKER)


def build_rag_prompt(
    query: str,
    context_text: str,
    language: str | None = None,
    strict: bool = False,
) -> dict[str, str]:
    """Build the system/user message pair for a chat-style LLM.

    Returns {"system": ..., "user": ...}. The context is explicitly delimited
    so the model can distinguish evidence from instruction text.
    """
    system = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT
    if language:
        system += (
            f"\n\nThe user's question is in language code: {language}. "
            "Answer in that language."
        )
    user = (
        f"Question: {query}\n\n"
        f"{context_text}\n\n"
        f"Answer the question using only the text between {CONTEXT_BEGIN} and "
        f"{CONTEXT_END}. If it is insufficient, refuse."
    )
    return {"system": system, "user": user}


def build_rag_messages(
    query: str,
    context_text: str,
    language: str | None = None,
    strict: bool = False,
) -> list[dict[str, str]]:
    """Build the chat message list (system + user) for an LLMProvider."""
    prompt = build_rag_prompt(query, context_text, language=language, strict=strict)
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]


def parse_confidence(text: str) -> float | None:
    """Extract a trailing 'CONFIDENCE: <0-1>' token if the model emitted one."""
    if CONFIDENCE_MARKER not in text:
        return None
    tail = text.split(CONFIDENCE_MARKER, 1)[1].strip()
    try:
        value = float(tail.split()[0])
    except (ValueError, IndexError):
        return None
    return max(0.0, min(1.0, value))


def strip_confidence(text: str) -> str:
    """Remove the 'CONFIDENCE: ...' trailer from generated output."""
    if CONFIDENCE_MARKER in text:
        return text.split(CONFIDENCE_MARKER, 1)[0].rstrip()
    return text


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith(REFUSAL_MARKER)
