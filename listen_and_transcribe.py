import queue
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd  # pyright: ignore[reportMissingImports]
from scipy.io.wavfile import write


def list_input_devices():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(
                f"{i}: {dev['name']} "
                f"(default_samplerate={dev['default_samplerate']}, "
                f"max_input_channels={dev['max_input_channels']})"
            )


def get_input_device_info(device: int | None = None) -> dict:
    return sd.query_devices(device, "input")


def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> str:
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    write(path, sample_rate, audio)
    return path


def transcribe_with_openai_whisper(model, wav_path: str, language: str | None = None) -> str:
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(wav_path, **kwargs)
    return result["text"].strip()


def transcribe_with_faster_whisper(model, wav_path: str, language: str | None = None) -> str:
    kwargs = {}
    if language:
        kwargs["language"] = language
    segments, _info = model.transcribe(wav_path, **kwargs)
    return "".join(segment.text for segment in segments).strip()


def listen_until_silence(
    asr_model,
    model_type: str = "openai_whisper",
    device: int | None = None,
    sample_rate: int | None = None,
    channels: int = 1,
    block_duration: float = 0.25,
    silence_duration: float = 1.0,
    max_duration: float = 8.0,
    start_threshold: float = 250.0,
    continue_threshold: float = 120.0,
    language: str | None = "en",
    keep_file: bool = False,
    output_path: str | None = None,
) -> dict:
    if channels not in (1, 2):
        raise ValueError("channels should usually be 1 or 2")

    if sample_rate is None:
        device_info = get_input_device_info(device)
        sample_rate = int(device_info["default_samplerate"])

    q: queue.Queue[np.ndarray] = queue.Queue()
    blocksize = int(sample_rate * block_duration)

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata.copy())

    print(f"Listening... (device={device}, sample_rate={sample_rate})")

    heard_speech = False
    speech_chunks: list[np.ndarray] = []
    silence_blocks_needed = max(1, int(silence_duration / block_duration))
    silent_blocks = 0
    max_blocks = max(1, int(max_duration / block_duration))
    blocks_after_start = 0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=device,
        blocksize=blocksize,
        callback=callback,
    ):
        while True:
            chunk = q.get()

            mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
            level = float(np.mean(np.abs(mono)))

            if not heard_speech:
                if level >= start_threshold:
                    heard_speech = True
                    speech_chunks.append(chunk)
                    silent_blocks = 0
                    blocks_after_start = 1
                    print("Speech detected.")
            else:
                speech_chunks.append(chunk)
                blocks_after_start += 1

                if level < continue_threshold:
                    silent_blocks += 1
                else:
                    silent_blocks = 0

                if silent_blocks >= silence_blocks_needed:
                    print("Silence detected. Stopping capture.")
                    break

                if blocks_after_start >= max_blocks:
                    print("Max utterance duration reached.")
                    break

    if not heard_speech or not speech_chunks:
        return {
            "text": "",
            "wav_path": None,
            "sample_rate": sample_rate,
            "heard_speech": False,
        }

    audio = np.concatenate(speech_chunks, axis=0)

    temp_created = False
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()
        temp_created = True
    else:
        wav_path = output_path

    save_wav(audio, sample_rate, wav_path)

    try:
        if model_type == "openai_whisper":
            text = transcribe_with_openai_whisper(asr_model, wav_path, language=language)
        elif model_type == "faster_whisper":
            text = transcribe_with_faster_whisper(asr_model, wav_path, language=language)
        else:
            raise ValueError("model_type must be 'openai_whisper' or 'faster_whisper'")
    finally:
        if temp_created and not keep_file:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

    returned_path = wav_path if (keep_file or not temp_created) else None

    return {
        "text": text,
        "wav_path": returned_path,
        "sample_rate": sample_rate,
        "heard_speech": True,
    }


if __name__ == "__main__":
    print("Available input devices:")
    list_input_devices()