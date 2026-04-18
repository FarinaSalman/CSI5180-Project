from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

_current_process: subprocess.Popen | None = None


def validate_audio_file(file_path: str | os.PathLike) -> Path:
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in [".mp3", ".wav"]:
        raise ValueError(f"Expected .mp3 or .wav file, got: {path.suffix}")

    return path


def _play_with_subprocess(command: list[str], wait: bool) -> None:
    global _current_process

    if wait:
        subprocess.run(command, check=True)
        _current_process = None
    else:
        _current_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def play_audio(file_path: str | os.PathLike, wait: bool = True):
    path = validate_audio_file(file_path)
    system = platform.system()

    if system == "Windows":
        import winsound

        if wait:
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
        else:
            winsound.PlaySound(
                str(path),
                winsound.SND_FILENAME | winsound.SND_ASYNC,
            )
        return str(path)

    if system == "Darwin":
        # macOS built-in player
        _play_with_subprocess(["afplay", str(path)], wait)
        return str(path)

    # Linux / fallback
    if shutil.which("ffplay"):
        _play_with_subprocess(
            ["ffplay", "-nodisp", "-autoexit", str(path)],
            wait,
        )
        return str(path)

    raise RuntimeError(
        "No supported audio playback method found. "
        "On macOS, afplay should exist. On Linux, install ffmpeg/ffplay."
    )


def stop_playback(play_result=None) -> None:
    global _current_process
    system = platform.system()

    if system == "Windows":
        import winsound
        winsound.PlaySound(None, 0)
        return

    if _current_process is not None:
        try:
            _current_process.terminate()
        except Exception:
            pass
        _current_process = None


if __name__ == "__main__":
    sample_file = "assistant_response.wav"

    try:
        print("Testing file:", Path(sample_file).resolve())
        print("Exists:", Path(sample_file).exists())
        print("Platform:", platform.system())
        play_audio(sample_file, wait=True)
        print("Playback finished.")
    except Exception as e:
        print(f"Could not play audio: {type(e).__name__}: {e}")