from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

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


def convert_mp3_to_temp_wav(mp3_path: Path) -> str:
    ffmpeg = Path(FFMPEG_EXE)
    if not ffmpeg.exists():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_wav = tmp.name

    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(mp3_path),
        temp_wav,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")

    return temp_wav


def play_audio(file_path: str | os.PathLike, wait: bool = True):
    path = validate_audio_file(file_path)

    if path.suffix.lower() == ".wav":
        wave_obj = sa.WaveObject.from_wave_file(str(path))
        play_obj = wave_obj.play()
        if wait:
            play_obj.wait_done()
        return play_obj

    temp_wav = convert_mp3_to_temp_wav(path)

    try:
        wave_obj = sa.WaveObject.from_wave_file(temp_wav)
        play_obj = wave_obj.play()

        if wait:
            play_obj.wait_done()
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        return play_obj, temp_wav
    except Exception:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise


def stop_playback(play_result) -> None:
    if play_result is None:
        return

    if isinstance(play_result, tuple):
        play_obj, temp_wav = play_result
    else:
        play_obj, temp_wav = play_result, None

    play_obj.stop()

    if temp_wav and os.path.exists(temp_wav):
        os.remove(temp_wav)


if __name__ == "__main__":
    sample_file = "assistant_response.mp3"

    try:
        print("Testing file:", Path(sample_file).resolve())
        print("Exists:", Path(sample_file).exists())
        print("ffmpeg exists:", Path(FFMPEG_EXE).exists())
        play_audio(sample_file, wait=True)
        print("Playback finished.")
    except Exception as e:
        print(f"Could not play audio: {type(e).__name__}: {e}")