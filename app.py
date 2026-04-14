"""
app.py — Voice-Controlled Local AI Agent
Entry point: `python app.py`
"""

from __future__ import annotations
import os
import tempfile
import traceback

import gradio as gr
from dotenv import load_dotenv

from stt import transcribe
from intent import classify, label_for, INTENT_LABELS
from tools import dispatch, ToolResult
from memory import SessionMemory

load_dotenv()

# ── Global session memory (one per process) ──────────────────────────────────
memory = SessionMemory()

# ── Pending confirmation state ────────────────────────────────────────────────
_pending: dict = {}   # stores intent + text awaiting human confirmation


# ── CSS — dark glassmorphism theme ───────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg: #080c14;
    --surface: rgba(255,255,255,0.04);
    --surface-hover: rgba(255,255,255,0.07);
    --border: rgba(255,255,255,0.08);
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --danger: #f43f5e;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --radius: 14px;
    --font-display: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--font-display) !important;
    color: var(--text) !important;
    min-height: 100vh;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00e5ff 0%, #7c3aed 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-family: var(--font-mono);
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(0,229,255,0.2); }

.card-label {
    font-size: 0.7rem;
    font-family: var(--font-mono);
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Badge pill */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    font-weight: 500;
    letter-spacing: 0.04em;
}
.badge-cyan  { background: rgba(0,229,255,0.12); color: #00e5ff; border: 1px solid rgba(0,229,255,0.25); }
.badge-purple{ background: rgba(124,58,237,0.12); color: #a78bfa; border: 1px solid rgba(124,58,237,0.25); }
.badge-green { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }

/* Inputs */
input, textarea, .gr-textbox textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}

/* Buttons */
button.primary-btn {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    letter-spacing: 0.02em;
}
button.primary-btn:hover { opacity: 0.9; transform: translateY(-1px); }

.gr-button-primary {
    background: linear-gradient(135deg, #00e5ff, #7c3aed) !important;
    border: none !important;
    color: #000 !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    border-radius: var(--radius) !important;
}
.gr-button-secondary {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-display) !important;
    border-radius: var(--radius) !important;
}

/* Status ring animation */
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0 rgba(0,229,255,0.4); }
  70%  { box-shadow: 0 0 0 10px rgba(0,229,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,229,255,0); }
}
.processing { animation: pulse-ring 1.5s infinite; }

/* Output code block */
.output-code {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: #c9d1d9 !important;
    padding: 1rem !important;
    max-height: 420px;
    overflow-y: auto;
}

/* Confirmation box */
.confirm-box {
    background: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-radius: var(--radius);
    padding: 1.25rem;
}

/* History */
.history-entry {
    border-left: 2px solid var(--accent);
    padding-left: 0.75rem;
    margin-bottom: 0.75rem;
}

/* Tabs */
.gr-tab-item {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
}

/* Hide default Gradio footer */
footer { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
"""

HEADER_HTML = """
<div class="hero-header">
  <h1 class="hero-title">⬡ VoiceAgent</h1>
  <p class="hero-sub">// speak → transcribe → classify → execute → display</p>
</div>
"""

# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(audio_path: str, require_confirm: bool) -> tuple:
    """
    Core pipeline: STT → intent → (optional confirm) → tools.
    Returns (transcription, intent_html, action_html, output_text, confirm_visible, status)
    """
    # 1. STT
    try:
        transcription = transcribe(audio_path)
    except Exception as e:
        return ("", "", "", f"❌ STT Error: {e}", False, "error")

    if not transcription:
        return ("", "", "", "⚠️ Could not detect speech. Please try again.", False, "warning")

    # 2. Intent
    try:
        intent = classify(transcription)
    except Exception as e:
        return (transcription, "", "", f"❌ Intent Error: {e}", False, "error")

    intents = intent.get("intents", ["general_chat"])
    is_file_op = any(i in ("create_file", "write_code") for i in intents)

    intent_html = _build_intent_html(intent)

    # 3. Human-in-the-loop: pause before file ops if enabled
    if require_confirm and is_file_op:
        global _pending
        _pending = {"intent": intent, "text": transcription}
        confirm_msg = (
            f"⚠️ **Confirmation Required**\n\n"
            f"The agent wants to perform a file operation:\n\n"
            f"- Intents: {', '.join(label_for(i) for i in intents)}\n"
            f"- File: `{intent.get('filename') or 'auto-named'}`\n\n"
            f"Approve or reject below."
        )
        return (transcription, intent_html, confirm_msg, "", True, "confirm")

    # 4. Execute
    return _execute(intent, transcription, intent_html)


def _execute(intent: dict, transcription: str, intent_html: str) -> tuple:
    memory.add_user_turn(transcription)
    try:
        results: list[ToolResult] = dispatch(intent, transcription, memory.get_chat_history())
    except Exception as e:
        return (transcription, intent_html, "", f"❌ Tool Error: {e}", False, "error")

    actions = [r.action for r in results]
    files = [r.filepath for r in results if r.filepath]
    outputs = []
    for r in results:
        if r.error:
            outputs.append(f"❌ {r.error}")
        else:
            outputs.append(r.output)

    combined_output = "\n\n---\n\n".join(outputs)
    action_html = _build_action_html(actions, files)

    # Update memory
    if combined_output:
        memory.add_assistant_turn(combined_output[:500])
    memory.log_action(transcription, intent.get("intents", []), actions, files)

    return (transcription, intent_html, action_html, combined_output, False, "success")


def _build_intent_html(intent: dict) -> str:
    intents = intent.get("intents", [])
    conf = intent.get("confidence", 0.8)
    badges = " ".join(
        f'<span class="badge badge-cyan">{label_for(i)}</span>' for i in intents
    )
    lang = f'<span class="badge badge-purple">lang: {intent["language"]}</span>' if intent.get("language") else ""
    fname = f'<span class="badge badge-green">file: {intent["filename"]}</span>' if intent.get("filename") else ""
    conf_color = "#10b981" if conf >= 0.8 else "#f59e0b" if conf >= 0.5 else "#f43f5e"
    return (
        f'<div class="card" style="margin-top:0">'
        f'<div class="card-label">◈ detected intents</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.6rem">{badges} {lang} {fname}</div>'
        f'<div style="font-family:var(--font-mono);font-size:0.75rem;color:{conf_color};">'
        f'confidence: {conf:.0%}</div>'
        f'</div>'
    )


def _build_action_html(actions: list[str], files: list[str]) -> str:
    action_items = "".join(f"<li>✅ {a}</li>" for a in actions)
    file_items = "".join(f'<li><code>{f}</code></li>' for f in files) if files else "<li>—</li>"
    return (
        f'<div class="card" style="margin-top:0">'
        f'<div class="card-label">◈ actions taken</div>'
        f'<ul style="margin:0;padding-left:1.2rem;font-size:0.88rem">{action_items}</ul>'
        f'<div class="card-label" style="margin-top:0.75rem">◈ files created</div>'
        f'<ul style="margin:0;padding-left:1.2rem;font-family:var(--font-mono);font-size:0.8rem">{file_items}</ul>'
        f'</div>'
    )


# ── Gradio handlers ───────────────────────────────────────────────────────────

def handle_audio(audio, require_confirm):
    if audio is None:
        return "", "", "", "⚠️ No audio provided."
    
    # audio is already a filepath when type="filepath"
    return _run_pipeline(audio, require_confirm)[:4]


def handle_confirm(approved):
    global _pending
    if not _pending:
        return "", "", "⚠️ Nothing pending.", ""
    intent = _pending.pop("intent")
    text = _pending.pop("text", "")
    if not approved:
        return (
            text,
            _build_intent_html(intent),
            "🚫 Operation cancelled by user.",
            "",
        )
    transcription, intent_html, action_html, output, _, _ = _execute(intent, text, _build_intent_html(intent))
    return transcription, intent_html, action_html, output


def handle_clear():
    memory.clear()
    return "", "", "", "", "🗑️ Session cleared."


def get_history():
    return memory.to_history_markdown()


# ── UI Layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="VoiceAgent") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # ── Tab 1: Agent ─────────────────────────────────────────────────────
        with gr.Tab("🎙️ Agent"):
            with gr.Row():
                # LEFT — Input column
                with gr.Column(scale=1):
                    gr.HTML('<div class="card-label" style="padding:0.25rem 0">◈ audio input</div>')

                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="",
                        elem_classes=["card"],
                    )

                    require_confirm = gr.Checkbox(
                        label="🔐 Require confirmation before file operations",
                        value=True,
                        elem_classes=["card"],
                    )

                    with gr.Row():
                        run_btn = gr.Button("▶  Run Agent", variant="primary")
                        clear_btn = gr.Button("✕ Clear", variant="secondary")

                    # Confirmation panel (hidden by default)
                    with gr.Group(visible=False) as confirm_group:
                        gr.HTML('<div class="confirm-box">')
                        confirm_label = gr.Markdown("", elem_classes=["confirm-box"])
                        with gr.Row():
                            approve_btn = gr.Button("✅ Approve", variant="primary")
                            reject_btn = gr.Button("🚫 Reject", variant="secondary")
                        gr.HTML('</div>')

                    status_msg = gr.Markdown("")

                # RIGHT — Output column
                with gr.Column(scale=1):
                    gr.HTML('<div class="card-label" style="padding:0.25rem 0">◈ transcription</div>')
                    transcription_out = gr.Textbox(
                        label="",
                        lines=3,
                        interactive=False,
                        placeholder="Transcribed speech will appear here…",
                        elem_classes=["card"],
                    )

                    gr.HTML('<div class="card-label" style="padding:0.25rem 0">◈ intent analysis</div>')
                    intent_out = gr.HTML("")

                    gr.HTML('<div class="card-label" style="padding:0.25rem 0">◈ action log</div>')
                    action_out = gr.HTML("")

                    gr.HTML('<div class="card-label" style="padding:0.25rem 0">◈ output</div>')
                    output_out = gr.Textbox(
                        label="",
                        lines=12,
                        interactive=False,
                        placeholder="Agent output will appear here…",
                        elem_classes=["output-code"],
                    )

        # ── Tab 2: History ────────────────────────────────────────────────────
        with gr.Tab("📋 Session History"):
            refresh_btn = gr.Button("↻ Refresh History", variant="secondary")
            history_md = gr.Markdown("_No actions yet this session._")
            refresh_btn.click(get_history, outputs=history_md)

        # ── Tab 3: About ──────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## VoiceAgent — Architecture

```
Audio (mic / file)
       │
       ▼
  Groq Whisper          ← STT: fast, accurate, cloud-based
       │
       ▼
  Ollama (llama3)        ← Intent classifier — returns JSON
       │
       ├─ create_file   → Creates blank file in output/
       ├─ write_code    → Generates code with Ollama, saves to output/
       ├─ summarize     → Summarises text, optionally saves to output/
       └─ general_chat  → Conversational reply with session memory
       │
       ▼
  Gradio UI             ← Displays all pipeline stages
```

### Models
| Component | Model | Provider |
|-----------|-------|----------|
| STT | whisper-large-v3 | Groq API |
| LLM | llama3 (or mistral) | Ollama (local) |

### Bonus Features Implemented
- ✅ Compound commands (multiple intents per utterance)
- ✅ Human-in-the-loop confirmation before file ops
- ✅ Graceful error handling & fallbacks
- ✅ Session memory (chat context + action log)
""")

    # ── Wire events ───────────────────────────────────────────────────────────
    run_btn.click(
        fn=handle_audio,
        inputs=[audio_input, require_confirm],
        outputs=[transcription_out, intent_out, action_out, output_out],
    )

    # When confirm group shows, populate the confirmation label
    def update_confirm_label(a, r):
        if _pending.get("intent"):
            return gr.update(
                value=(
                    f"⚠️ **Confirm file operation?**\n\n"
                    f"Intents: {', '.join(label_for(i) for i in _pending.get('intent', {}).get('intents', []))}\n\n"
                    f"File: `{_pending.get('intent', {}).get('filename', 'auto-named')}`"
                )
            )
        return gr.update(value="")

    run_btn.click(
        fn=update_confirm_label,
        inputs=[audio_input, require_confirm],
        outputs=[confirm_label],
    )

    approve_btn.click(
        fn=lambda: handle_confirm(True),
        outputs=[transcription_out, intent_out, action_out, output_out],
    )

    reject_btn.click(
        fn=lambda: handle_confirm(False),
        outputs=[transcription_out, intent_out, action_out, output_out],
    )

    clear_btn.click(
        fn=handle_clear,
        outputs=[transcription_out, intent_out, action_out, output_out, status_msg],
    )


if __name__ == "__main__":
    os.makedirs(os.getenv("OUTPUT_DIR", "output"), exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )
