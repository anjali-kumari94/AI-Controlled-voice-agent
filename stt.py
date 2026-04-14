"""
stt.py — Speech-to-Text via Groq Whisper API
Converts audio (mic or uploaded file) to text.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example → .env and add your key from https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file using Groq's Whisper endpoint.

    Args:
        audio_path: Absolute or relative path to a .wav / .mp3 / .m4a file.

    Returns:
        Transcribed text string.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        EnvironmentError: If GROQ_API_KEY is missing.
        RuntimeError: For any upstream API error.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = _get_client()

    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f),
                model="whisper-large-v3",
                response_format="text",
                language="en",
            )
        # Groq returns a plain string when response_format="text"
        return str(transcription).strip()

    except Exception as e:
        raise RuntimeError(f"STT transcription failed: {e}") from e
