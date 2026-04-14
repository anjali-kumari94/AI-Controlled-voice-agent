"""
intent.py — Intent classification via local Ollama LLM.

Supported intents:
  - create_file       : User wants to create a blank file or folder
  - write_code        : User wants code generated and saved to a file
  - summarize         : User wants text summarised
  - general_chat      : Anything else / open-ended conversation
  - compound          : Multiple intents detected in one utterance
"""

from __future__ import annotations
import json
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

INTENT_LABELS = {
    "create_file": "📁 Create File / Folder",
    "write_code": "💻 Write Code",
    "summarize": "📝 Summarise Text",
    "general_chat": "💬 General Chat",
    "compound": "🔗 Compound Command",
}

SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled AI agent.
Analyse the user's message and return a JSON object with EXACTLY these keys:

{
  "intents": ["<one or more intent labels>"],
  "filename": "<suggested filename if applicable, else null>",
  "language": "<programming language if code is requested, else null>",
  "summary_target": "<text to summarise if given inline, else null>",
  "confidence": <float 0-1>
}

Valid intent labels (choose one or more):
- create_file   → user wants to create a file or directory
- write_code    → user wants code generated / written to a file
- summarize     → user wants text summarised
- general_chat  → open conversation, question-answer

Rules:
- If multiple intents apply, list all of them (compound command).
- For write_code, infer the programming language from context.
- For create_file or write_code, suggest a sensible filename.
- Return ONLY valid JSON. No markdown, no explanation.
"""


def classify(text: str) -> dict:
    """
    Classify the intent(s) of the transcribed text.

    Returns a dict:
    {
        "intents": list[str],
        "filename": str | None,
        "language": str | None,
        "summary_target": str | None,
        "confidence": float,
        "raw_text": str,
    }
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "format": "json",
    }

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        result = _safe_parse(raw)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Is it running? Run: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"Intent classification failed: {e}") from e

    result["raw_text"] = text
    return result


def _safe_parse(raw: str) -> dict:
    """Parse LLM output leniently, falling back to general_chat on failure."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
        # Normalise keys
        intents = data.get("intents", ["general_chat"])
        if isinstance(intents, str):
            intents = [intents]
        intents = [i for i in intents if i in INTENT_LABELS]
        if not intents:
            intents = ["general_chat"]
        if len(intents) > 1:
            intents = list(dict.fromkeys(["compound"] + intents))
        return {
            "intents": intents,
            "filename": data.get("filename"),
            "language": data.get("language"),
            "summary_target": data.get("summary_target"),
            "confidence": float(data.get("confidence", 0.8)),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "intents": ["general_chat"],
            "filename": None,
            "language": None,
            "summary_target": None,
            "confidence": 0.5,
        }


def label_for(intent: str) -> str:
    return INTENT_LABELS.get(intent, intent)
