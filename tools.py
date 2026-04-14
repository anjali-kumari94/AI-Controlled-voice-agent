"""
tools.py — Tool execution layer.

Each tool function accepts the parsed intent dict + the original transcribed
text and returns a ToolResult dataclass.
"""

from __future__ import annotations
import os
import re
import json
import requests
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")


# ── Data contract ────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    action: str          # human-readable description of what was done
    output: str          # main content (code, summary, chat reply, etc.)
    filepath: str | None = None   # if a file was created/written
    error: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _safe_filename(name: str | None, default: str, ext: str = "") -> str:
    if not name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{default}_{ts}"
    # Strip path traversal
    name = re.sub(r"[^\w\-. ]", "_", os.path.basename(name))
    if ext and not name.endswith(ext):
        name = name + ext
    return name


def _ollama_chat(system: str, user: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Run: ollama serve")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e


# ── Tool implementations ─────────────────────────────────────────────────────

def create_file(intent: dict, text: str) -> ToolResult:
    """Create an empty file (or directory) inside output/."""
    out = _ensure_output_dir()
    filename = _safe_filename(intent.get("filename"), "new_file", "")
    filepath = os.path.join(out, filename)

    try:
        if "." in filename:  # treat as file
            with open(filepath, "w") as f:
                f.write(f"# Created by Voice Agent on {datetime.now().isoformat()}\n")
            action = f"Created file `{filepath}`"
        else:  # treat as directory
            os.makedirs(filepath, exist_ok=True)
            action = f"Created directory `{filepath}`"

        return ToolResult(
            success=True,
            action=action,
            output=f"✅ {action}",
            filepath=filepath,
        )
    except Exception as e:
        return ToolResult(success=False, action="Create file", output="", error=str(e))


def write_code(intent: dict, text: str) -> ToolResult:
    """Generate code with Ollama and save to output/."""
    out = _ensure_output_dir()
    language = intent.get("language") or "python"

    # Infer extension
    ext_map = {
        "python": ".py", "javascript": ".js", "typescript": ".ts",
        "html": ".html", "css": ".css", "bash": ".sh", "shell": ".sh",
        "java": ".java", "c": ".c", "cpp": ".cpp", "go": ".go",
        "rust": ".rs", "sql": ".sql", "r": ".r", "ruby": ".rb",
    }
    ext = ext_map.get(language.lower(), ".txt")
    filename = _safe_filename(intent.get("filename"), "generated_code", ext)
    filepath = os.path.join(out, filename)

    system = (
        f"You are an expert {language} developer. "
        "Write clean, well-commented, production-ready code. "
        "Return ONLY the code — no explanation, no markdown fences."
    )

    try:
        code = _ollama_chat(system, text)
        # Strip accidental fences
        code = re.sub(r"^```[\w]*\n?", "", code).rstrip("`").strip()

        with open(filepath, "w") as f:
            f.write(code)

        action = f"Generated {language} code → `{filepath}`"
        return ToolResult(
            success=True,
            action=action,
            output=code,
            filepath=filepath,
        )
    except Exception as e:
        return ToolResult(success=False, action="Write code", output="", error=str(e))


def summarize(intent: dict, text: str) -> ToolResult:
    """Summarise text (inline or from the transcript itself)."""
    target = intent.get("summary_target") or text

    system = (
        "You are a concise summarisation assistant. "
        "Produce a clear, structured summary with bullet points for key facts. "
        "Be thorough but succinct."
    )

    try:
        summary = _ollama_chat(system, f"Summarise the following:\n\n{target}")

        # Optionally save to file
        out = _ensure_output_dir()
        filename = _safe_filename(intent.get("filename"), "summary", ".md")
        filepath = os.path.join(out, filename)

        # Only save if compound command explicitly mentioned saving
        intents = intent.get("intents", [])
        saved = False
        if "create_file" in intents or "write_code" in intents:
            with open(filepath, "w") as f:
                f.write(f"# Summary\n\n{summary}\n")
            saved = True

        return ToolResult(
            success=True,
            action="Summarised text" + (f" → saved `{filepath}`" if saved else ""),
            output=summary,
            filepath=filepath if saved else None,
        )
    except Exception as e:
        return ToolResult(success=False, action="Summarise", output="", error=str(e))


def general_chat(intent: dict, text: str, history: list[dict] | None = None) -> ToolResult:
    """Open-ended conversation with session memory."""
    system = (
        "You are a helpful, knowledgeable AI assistant. "
        "Respond clearly and concisely. "
        "If the user seems to want you to do something the system can handle "
        "(create files, write code, summarise), gently let them know they can ask directly."
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-10:])  # keep last 10 turns for context
    messages.append({"role": "user", "content": text})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        reply = resp.json()["message"]["content"].strip()
        return ToolResult(success=True, action="Responded to chat", output=reply)
    except Exception as e:
        return ToolResult(success=False, action="General chat", output="", error=str(e))


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch(intent: dict, text: str, history: list[dict] | None = None) -> list[ToolResult]:
    """
    Route to the correct tool(s) based on detected intents.
    Returns a list because compound commands trigger multiple tools.
    """
    intents = intent.get("intents", ["general_chat"])
    # Remove the meta-label "compound" — real intents follow it
    active = [i for i in intents if i != "compound"] or ["general_chat"]

    results: list[ToolResult] = []
    for i in active:
        if i == "create_file":
            results.append(create_file(intent, text))
        elif i == "write_code":
            results.append(write_code(intent, text))
        elif i == "summarize":
            results.append(summarize(intent, text))
        else:
            results.append(general_chat(intent, text, history))

    return results
