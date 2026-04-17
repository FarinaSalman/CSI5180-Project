from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
import winsound

_current_audio_path: str | None = None

import simpleaudio as sa

FFMPEG_EXE = os.getenv("FFMPEG")


def validate_audio_file(file_path: str | os.PathLike) -> Path:
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in [".mp3", ".wav"]:
        raise ValueError(f"Expected .mp3 or .wav file, got: {path.suffix}")

    return path

def play_audio(file_path: str | os.PathLike, wait: bool = True):
    global _current_audio_path

    path = validate_audio_file(file_path)
    _current_audio_path = str(path)

    if wait:
        winsound.PlaySound(_current_audio_path, winsound.SND_FILENAME)
    else:
        winsound.PlaySound(
            _current_audio_path,
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )

    return _current_audio_path

def stop_playback(play_result=None) -> None:
    global _current_audio_path
    winsound.PlaySound(None, 0)
    _current_audio_path = None

if __name__ == "__main__":
    sample_file = "assistant_response.wav"

    try:
        print("Testing file:", Path(sample_file).resolve())
        print("Exists:", Path(sample_file).exists())
        print("ffmpeg exists:", Path(FFMPEG_EXE).exists())
        play_audio(sample_file, wait=True)
        print("Playback finished.")
    except Exception as e:
        print(f"Could not play audio: {type(e).__name__}: {e}")