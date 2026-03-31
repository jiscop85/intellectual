from __future__ import annotations

from functools import lru_cache
from typing import List, Dict, Any
import os
import re


@lru_cache(maxsize=1)
def get_generator():
    try:
        from transformers import pipeline

        model_name = os.getenv("GEN_MODEL_NAME", "google/flan-t5-base")
        return pipeline("text2text-generation", model=model_name)
    except Exception:
        return None


def _simple_fallback_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "متنی برای پاسخ پیدا نشد."

    q_words = set(re.findall(r"\w+", question.lower()))
    ranked = []

    for item in contexts:
        text = item["text"]
        words = set(re.findall(r"\w+", text.lower()))
        score = len(q_words & words)
        ranked.append((score, text))

    ranked.sort(key=lambda x: x[0], reverse=True)
    best_text = ranked[0][1]

    return (
        "پاسخ بر اساس نزدیک‌ترین بخش‌های سند:\n\n"
        + best_text[:900]
        + ("\n\n..." if len(best_text) > 900 else "")
    )


def answer_question(question: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "هیچ زمینه‌ای برای پاسخ پیدا نشد."

    context_text = "\n\n".join(
        [f"[Source {i+1}] {c['text']}" for i, c in enumerate(contexts)]
    )

    prompt = f"""
You are a helpful AI assistant.
Answer the question using only the provided context.
If the answer is not in the context, say you do not know.

Question:
{question}

Context:
{context_text}

Answer:
""".strip()

    gen = get_generator()
    if gen is None:
        return _simple_fallback_answer(question, contexts)

    output = gen(
        prompt,
        max_new_tokens=220,
        do_sample=False,
    )[0]["generated_text"]

    return output.strip()