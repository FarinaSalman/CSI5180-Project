'''
CSI 5180 Final Project Group P-2

Group Members: 
Drishya Praveen Cherukunnummal Poothakandi,300516807
Farina Salman, 300129324
Ziyad Gaffar, 300148116


Organized by project section:
- Dataclass / shared state
- Section 1: User Verification
- Section 2: Wake Word Detection
- Section 3: Automatic Speech Recognition
- Section 4: Intent Detection
- Section 5: Fulfillment
- Section 6: Answer Generation
- Section 7: Text-to-Speech
- Full pipeline


To run the code, first you gotta install the requirements files using the following line in the terminal:
"pip install -r requirements.txt"

Prerequisite:

Install ffmpeg and add to PATH, and to do so you need to follow this video: https://www.youtube.com/watch?v=JR36oH35Fgg
- Install through the https://www.ffmpeg.org/download.html link and choose your OS
'''
from __future__ import annotations

# Backend imports
from dataclasses import dataclass, field
from typing import Any, Optional
import random
import torch
import edge_tts
import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
import tempfile
import warnings
import whisper
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import asyncio
import threading
from dotenv import load_dotenv
import os

load_dotenv()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
env_path = os.getenv("PATH")

if env_path:
    os.environ["PATH"] = env_path + ":" + os.environ["PATH"]

from pathlib import Path
from collections import defaultdict
from pydub import AudioSegment
from pydub.playback import play
from resemblyzer import VoiceEncoder, preprocess_wav
import tensorflow as tf
from tensorflow.keras import layers, models
import soundfile as sf
import requests
import random
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn as nn
import json
from tensorflow.keras.models import load_model
import shutil
import subprocess
from pydub.utils import which
from pathlib import Path
from kokoro_onnx import Kokoro
from audio_playback import play_audio, stop_playback

FFMPEG = os.getenv("FFMPEG")
FFPROBE = os.getenv("FFPROBE")


AudioSegment.converter = FFMPEG
AudioSegment.ffmpeg = FFMPEG
AudioSegment.ffprobe = FFPROBE

# Warnings to ignore
# Reduce TensorFlow C++ log noise before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Optional: disable oneDNN optimization message and behavior note
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work",
    category=RuntimeWarning,
)

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="You are sending unauthenticated requests to the HF Hub.*",
    category=UserWarning,
)


# Opening books from books.json
def reset_books_db():
    global BOOKS_DB
    BOOKS_DB = {}
    with open(BOOKS_PATH, "w", encoding="utf-8") as f:
        json.dump(BOOKS_DB, f, indent=2, ensure_ascii=False)

BOOKS_PATH = Path(__file__).with_name("books.json")

try:
    with open(BOOKS_PATH, "r", encoding="utf-8") as f:
        BOOKS_DB = json.load(f)
except Exception:
    BOOKS_DB = {}

reset_books_db()

# =========================
# Shared config and state
# =========================

TARGET_SR = 16000
N_MFCC = 13
WINDOW_SEC = 0.025
HOP_SEC = 0.025

EMBED_THRESHOLD = 0.75
MFCC_THRESHOLD = 0.95
FINAL_THRESHOLD = 0.75
EMBED_WEIGHT = 0.80
MFCC_WEIGHT = 0.20

BYPASS_PIN = "1234"
HEY_ATLAS_TEXT = "hey atlas"
WAKEWORD_BYPASS_TEXT = "hey atlas"
SUPPORTED_EXTENSIONS = (".m4a", ".wav")
live_voice_busy = False

PROJECT_DATASET_ROOT = "project_dataset_audio"
ENROLLED_VERIFICATION_DIR = os.path.join(PROJECT_DATASET_ROOT, "enrolled_verification")
TEST_AUDIO_DIR = os.path.join(PROJECT_DATASET_ROOT, "test_audio")
WAKEWORD_DATASET_DIR = os.path.join(PROJECT_DATASET_ROOT, "wakeword_dataset")

NAME_MAP = {
    "Salman": "Farina",
    "Gaffar": "Ziyad",
    "Barriere": "Caroline",
    "Cherukunnummal Poothakandi": "Drishya",
}
SAVED_INTENT_MODEL_DIR = "saved_intent_model"

voice_encoder = None
profiles = {}


pipeline_control_state = {
    "current_stage": "verification",
    "verification_passed": False,
    "wakeword_passed": False,
    "transcript": "",
    "intent_result": None,
    "fulfillment_result": None,
    "final_answer": None,
}

pending_book_selection = {
    "active": False,
    "intent": None,
    "query": "",
    "results": [],
    "page": 0,
    "page_size": 5,
}

# Core State Management
# Holds assistant state across pipeline
@dataclass
class AssistantState:
    lock_state: str = "Locked"
    wake_state: str = "Sleep"
    last_transcript: str = ""
    last_intent: Optional[str] = None
    last_slots: dict = field(default_factory=dict)

# =========================
# Section 1: User Verification
# =========================

# Helper Functions

# Extract speaker name from filename
def get_speaker_name(filename):
    return Path(filename).stem.split("-")[0]

# Normalize speaker names mapping
def normalize_speaker(name):
    return NAME_MAP.get(name, name)

# Audio Processing
# Convert audio to WAV format
def convert_to_wav(file_path, target_sr=TARGET_SR):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".wav":
        return file_path

    #debug print
    print("Using ffmpeg:", AudioSegment.converter)
    print("Using ffprobe:", getattr(AudioSegment, "ffprobe", None))

    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    audio.export(tmp_path, format="wav")
    del audio
    return tmp_path

# Helper for cloning wav
def clone_temp_wav(src_path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    shutil.copy(src_path, tmp_path)
    return tmp_path

def safe_remove_file(path):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Could not delete temp file {path}: {e}")


# Dataset Helpers

# Return project dataset folder paths
def get_project_audio_paths(project_root: str = PROJECT_DATASET_ROOT):
    return {
        "project_root": project_root,
        "enrolled_verification_dir": os.path.join(project_root, "enrolled_verification"),
        "test_audio_dir": os.path.join(project_root, "test_audio"),
        "WakeWord_Dataset_dir": os.path.join(project_root, "WakeWord_Dataset"),
    }

# Resolve relative audio file paths
def resolve_audio_path(file_name_or_path: str, base_dir: Optional[str] = None) -> str:
    if os.path.isabs(file_name_or_path) or os.path.exists(file_name_or_path):
        return file_name_or_path

    if base_dir is not None:
        candidate = os.path.join(base_dir, file_name_or_path)
        if os.path.exists(candidate):
            return candidate

    candidate_test = os.path.join(TEST_AUDIO_DIR, file_name_or_path)
    if os.path.exists(candidate_test):
        return candidate_test

    candidate_wake = os.path.join(WAKEWORD_DATASET_DIR, file_name_or_path)
    if os.path.exists(candidate_wake):
        return candidate_wake

    candidate_enrolled = os.path.join(ENROLLED_VERIFICATION_DIR, file_name_or_path)
    if os.path.exists(candidate_enrolled):
        return candidate_enrolled

    return file_name_or_path
    

def list_audio_files(folder_path: str):
    if not os.path.isdir(folder_path):
        return []

    return [
        os.path.join(folder_path, name)
        for name in sorted(os.listdir(folder_path))
        if name.lower().endswith(SUPPORTED_EXTENSIONS)
    ]


# Feature Extractions

# Extract MFCC features from audio
def extract_mfcc(file_path, sr=TARGET_SR, window_sec=WINDOW_SEC, hop_sec=HOP_SEC, n_mfcc=N_MFCC):
    wav_path = convert_to_wav(file_path, target_sr=sr)
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    n_fft = int(window_sec * sr)
    hop_length = int(hop_sec * sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc

# Compute MFCC statistical signature
def extract_mfcc_signature(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    signature = np.concatenate([mfcc_mean, mfcc_std], axis=0)
    return signature.astype(np.float32)

# Lazily loads voice encoder model
def get_voice_encoder():
    global voice_encoder
    if voice_encoder is None:
        voice_encoder = VoiceEncoder()
    return voice_encoder


# Extracts speaker embedding vector
def extract_embedding(file_path):
    wav_path = convert_to_wav(file_path, target_sr=TARGET_SR)
    wav = preprocess_wav(wav_path)
    embedding = get_voice_encoder().embed_utterance(wav)
    return np.array(embedding, dtype=np.float32)


# Computes cosine similarity score
def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


# Normalizes similarity score range
def normalize_cosine(score):
    return (score + 1.0) / 2.0

# Builds speaker profiles from audio
def build_speaker_profiles(enrolled_dir: str = ENROLLED_VERIFICATION_DIR):
    global profiles

    enrolled_dir = resolve_audio_path(enrolled_dir, base_dir=PROJECT_DATASET_ROOT)
    speaker_files = defaultdict(list)

    for filename in sorted(os.listdir(enrolled_dir)):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            raw_speaker = get_speaker_name(filename)
            speaker = normalize_speaker(raw_speaker)
            full_path = os.path.join(enrolled_dir, filename)
            speaker_files[speaker].append(full_path)

    built_profiles = {}

    for speaker, files in speaker_files.items():
        emb_list = []
        mfcc_list = []

        for file_path in files:
            try:
                emb_list.append(extract_embedding(file_path))
                mfcc_list.append(extract_mfcc_signature(file_path))
            except Exception as e:
                print("Skipped:", file_path, "|", e)

        if emb_list:
            built_profiles[speaker] = {
                "embedding_profile": np.mean(np.stack(emb_list), axis=0),
                "mfcc_profile": np.mean(np.stack(mfcc_list), axis=0),
                "num_files": len(emb_list),
            }

    profiles = built_profiles
    return built_profiles


# Verifies speaker identity from voice
def verify_any_user(test_file, pin=None):
    test_file = resolve_audio_path(test_file, base_dir=TEST_AUDIO_DIR)
    state_before = "Locked"

    if pin is not None and pin == BYPASS_PIN:
        return {
            "matched_user": None,
            "best_embedding_score": 0.0,
            "best_mfcc_score": 0.0,
            "best_final_score": 0.0,
            "accepted": True,
            "state_before": state_before,
            "state_after": "Unlocked",
            "bypass_used": True,
        }

    test_embedding = extract_embedding(test_file)
    test_mfcc = extract_mfcc_signature(test_file)

    best_user = None
    best_embedding_score = -1
    best_mfcc_score = -1
    best_final_score = -1

    for user, data in profiles.items():
        enrolled_embedding = data["embedding_profile"]
        enrolled_mfcc = data["mfcc_profile"]
        raw_embedding_score = cosine_similarity(enrolled_embedding, test_embedding)
        raw_mfcc_score = cosine_similarity(enrolled_mfcc, test_mfcc)
        embedding_score = normalize_cosine(raw_embedding_score)
        mfcc_score = normalize_cosine(raw_mfcc_score)
        final_score = EMBED_WEIGHT * embedding_score + MFCC_WEIGHT * mfcc_score

        if final_score > best_final_score:
            best_final_score = final_score
            best_embedding_score = embedding_score
            best_mfcc_score = mfcc_score
            best_user = user

    accepted = (
        best_embedding_score >= EMBED_THRESHOLD
        and best_mfcc_score >= MFCC_THRESHOLD
        and best_final_score >= FINAL_THRESHOLD
    )
    print(
    f"Best user={best_user} | "
    f"embed={best_embedding_score:.3f} (thr={EMBED_THRESHOLD}) | "
    f"mfcc={best_mfcc_score:.3f} (thr={MFCC_THRESHOLD}) | "
    f"final={best_final_score:.3f} (thr={FINAL_THRESHOLD}) | "
    f"accepted={accepted}"
)

    state_after = "Unlocked" if accepted else "Locked"

    return {
        "matched_user": best_user,
        "best_embedding_score": float(best_embedding_score),
        "best_mfcc_score": float(best_mfcc_score),
        "best_final_score": float(best_final_score),
        "accepted": bool(accepted),
        "state_before": state_before,
        "state_after": state_after,
        "bypass_used": False,
    }

# =========================
# Section 2: Wake Word Detection
# =========================

WAKE_WORD_TEXT = "hey atlas"
DURATION = 2.25
NUM_SAMPLES = int(TARGET_SR * DURATION)
WAKE_THRESHOLD = 0.55

def pad_or_truncate_audio(y, num_samples=NUM_SAMPLES):
    if len(y) > num_samples:
        y = y[:num_samples]
    elif len(y) < num_samples:
        y = np.pad(y, (0, num_samples - len(y)))
    return y

def preprocess_for_wakeword(file_path):
    file_path = resolve_audio_path(file_path, base_dir=WAKEWORD_DATASET_DIR)
    wav_path = convert_to_wav(file_path, target_sr=TARGET_SR)

    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    y = pad_or_truncate_audio(y, NUM_SAMPLES)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = tmp.name
    tmp.close()

    sf.write(temp_wav_path, y, sr)

    mfcc = extract_mfcc(temp_wav_path)   # reuse exact notebook MFCC pipeline
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc

# Builds CNN wakeword classification model
def build_wakeword_cnn(input_shape=(40, 100, 1)):
    # CNN Model used to detect wake word
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # --- CNN BLOCK 1 ---
        layers.Conv2D(16, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # --- CNN BLOCK 2 ---
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # --- CNN BLOCK 3 ---
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # --- PREPARE FOR RNN ---
        layers.Reshape((-1, 64)),  # treat time axis as sequence

        # --- RNN LAYER ---
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        # --- DENSE HEAD ---
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Detects wake word from audio
def detect_wakeword(test_file, typed_wakeword=None, wake_model=None):
    test_file = resolve_audio_path(test_file, base_dir=WAKEWORD_DATASET_DIR)
    state_before = "Sleep"

    if typed_wakeword is not None and typed_wakeword.strip().lower() == WAKE_WORD_TEXT:
        return {
            "wake_detected": True,
            "method": "typed_bypass",
            "score": 1.0,
            "state_before": state_before,
            "state_after": "Awake",
        }

    if wake_model is None:
        return {
            "wake_detected": False,
            "method": "model_missing",
            "score": 0.0,
            "state_before": state_before,
            "state_after": "Sleep",
        }

    x_input = preprocess_for_wakeword(test_file)
    score = float(wake_model.predict(x_input, verbose=0)[0][0])
    wake_detected = score >= WAKE_THRESHOLD

    return {
        "wake_detected": wake_detected,
        "method": "cnn_model",
        "score": score,
        "state_before": state_before,
        "state_after": "Awake" if wake_detected else "Sleep",
    }


# =========================
# Section 3: Automatic Speech Recognition
# =========================

# Normalizes audio length and format
def preprocess_audio_file(input_path: str, output_path: str, target_sr: int = 16000, duration: int = 5):
    y, _ = librosa.load(input_path, sr=target_sr, mono=True)
    num_samples = int(target_sr * duration)

    if len(y) < num_samples:
        y = np.pad(y, (0, num_samples - len(y)))
    else:
        y = y[:num_samples]

    sf.write(output_path, y, target_sr)
    return output_path


# Loads Whisper ASR model
def load_whisper_asr(model_size: str = "base"):
    return whisper.load_model(model_size)


# Transcribes speech to text
def transcribe_audio_file(audio_path: str, asr_model):
    audio_path = resolve_audio_path(audio_path, base_dir=TEST_AUDIO_DIR)
    result = asr_model.transcribe(audio_path)
    return result["text"].strip(), result


# =========================
# Section 4: Intent Detection
# =========================

# Builds joint intent and slot model
class JointIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.encoder.config.hidden_size
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    # Returns intent and slot logits
    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]

        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = None

        if intent_labels is not None and slot_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            slot_loss = slot_loss_fn(
                slot_logits.view(-1, slot_logits.shape[-1]),
                slot_labels.view(-1)
            )

            loss = intent_loss + slot_loss

        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }


# Loads intent and slot label maps
def load_label_maps(
    intent_map_path: str = os.path.join(SAVED_INTENT_MODEL_DIR, "intent2id.json"),
    slot_map_path: str = os.path.join(SAVED_INTENT_MODEL_DIR, "slot2id.json")
):
    with open(intent_map_path, "r", encoding="utf-8") as f:
        intent2id = json.load(f)

    with open(slot_map_path, "r", encoding="utf-8") as f:
        slot2id = json.load(f)

    id2intent = {int(v): k for k, v in intent2id.items()}
    id2slot = {int(v): k for k, v in slot2id.items()}

    return intent2id, id2intent, slot2id, id2slot


# Loads trained intent model and tokenizer
def load_intent_model(export_dir=SAVED_INTENT_MODEL_DIR, device="cpu"): 
    tokenizer = AutoTokenizer.from_pretrained(export_dir)

    with open(os.path.join(export_dir, "intent2id.json"), "r", encoding="utf-8") as f:
        intent2id = json.load(f)

    with open(os.path.join(export_dir, "slot2id.json"), "r", encoding="utf-8") as f:
        slot2id = json.load(f)

    id2intent = {int(v): k for k, v in intent2id.items()}
    id2slot = {int(v): k for k, v in slot2id.items()}

    model = JointIntentSlotModel(
        num_intents=len(intent2id),
        num_slots=len(slot2id)
    )

    model.load_state_dict(
        torch.load(os.path.join(export_dir, "intent_model.pt"), map_location=device)
    )

    model.to(device)
    model.eval()

    return model, tokenizer, id2intent, id2slot

# Predicts intent and slots from text
def predict_from_text(text, model, tokenizer, id2intent, id2slot, device):
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        intent_probs = torch.softmax(outputs["intent_logits"], dim=1)
        intent_pred_idx = torch.argmax(intent_probs, dim=1).item()
        pred_intent = id2intent[intent_pred_idx]
        intent_conf = intent_probs[0, intent_pred_idx].item()

        slot_preds_idx = torch.argmax(outputs["slot_logits"], dim=2)[0]
        pred_slots = [id2slot[int(p)] for p in slot_preds_idx]

    word_ids = encoding.word_ids(batch_index=0)
    extracted_slots = {}
    words = text.split()

    # for token_idx, word_idx in enumerate(word_ids):
    #     if word_idx is None:
    #         continue

    #     slot_label = pred_slots[token_idx]
    #     if slot_label == "O":
    #         continue

    #     if word_idx < len(words):
    #         word = words[word_idx].strip(" ?.,!")
    #         slot_type = slot_label.split("-")[-1].lower()

    #         if slot_type not in extracted_slots:
    #             extracted_slots[slot_type] = word
    #         else:
    #             extracted_slots[slot_type] += " " + word
    
    # Updated loop logic for pipeline.py


    
    processed_word_indices = set() # Track indices to avoid "gatsbytsby"

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        slot_label = pred_slots[token_idx]
        if slot_label == "O":
            continue

        # Skip if we've already added this word from the original text
        if word_idx in processed_word_indices:
            continue

        slot_type = slot_label.split("-")[-1].lower()
        word_to_add = words[word_idx].strip(" ?.,!")

        if slot_type not in extracted_slots:
            extracted_slots[slot_type] = word_to_add
        else:
            extracted_slots[slot_type] += " " + word_to_add
        
        # Mark this index as finished
        processed_word_indices.add(word_idx)

    print(f"Input text : {text}")
    print(f"Tokens     : {tokens}")
    print(f"Pred slots : {pred_slots}")
    print(f"Pred intent: {pred_intent} (conf={intent_conf:.3f})")
    print(f"Extracted slots: {extracted_slots}")

    return {
        "intent": pred_intent,
        "confidence": intent_conf,
        "slots": extracted_slots,
        "raw_slot_labels": pred_slots,
        "tokens": tokens,
    }


# =========================
# Section 5: Fulfillment
# =========================

ereader_state = {
    "current_book": None,
    "current_page": 0,
    "reading_session": False,
    "dark_mode": False,
}

#Weather code maps
weather_code_map = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    99: "thunderstorm with hail"
}

#Helper Functions:

def get_coordinates(city, user_lat=45.4215, user_lon=-75.6972):
  url = "https://geocoding-api.open-meteo.com/v1/search"
  params = {
      "name": city,
      "count": 10,
      "language": "en",
      "format": "json"
  }

  response = requests.get(url, params=params)
  if response.status_code != 200:
    return None

  data = response.json()
  if "results" not in data or len(data["results"]) == 0:
    return None

  # choose nearest result if multiple are returned
  def dist_sq(result):
    return (result["latitude"] - user_lat) ** 2 + (result["longitude"] - user_lon) ** 2

  best = min(data["results"], key=dist_sq)
  return best["latitude"], best["longitude"]


def call_weather_api(lat, lon):
  url = "https://api.open-meteo.com/v1/forecast"
  params = {
      "latitude": lat,
      "longitude": lon,
      "current": "temperature_2m,weather_code,wind_speed_10m",
      "daily": "weather_code,temperature_2m_max,temperature_2m_min",
      "timezone": "auto",
      "forecast_days": 3
  }

  response = requests.get(url, params=params)
  #print("STATUS:", response.status_code)  # DEBUG
  if response.status_code != 200:
    return None

  return response.json()

def extract_weather_info(weather_data):
  if weather_data is None:
    return None

  current = weather_data["current"]

  info = {
      "temperature": current["temperature_2m"],
      "windspeed": current["wind_speed_10m"],
      "weathercode": current["weather_code"],
      "daily_forecast": []
  }

  daily = weather_data.get("daily", {})
  dates = daily.get("time", [])
  highs = daily.get("temperature_2m_max", [])
  lows = daily.get("temperature_2m_min", [])
  codes = daily.get("weathercode", [])

  for i in range(min(len(dates), len(highs), len(lows), len(codes))):
    info["daily_forecast"].append({
        "date": dates[i],
        "high": highs[i],
        "low": lows[i],
        "weathercode": codes[i]
      })

  return info


def generate_weather_answer(city, info):
  description = weather_code_map.get(info["weathercode"], "unknown conditions")

  answer1 = (
      f"The current weather in {city} is {description}. "
      f"The temperature is {info['temperature']}°C with wind speed of {info['windspeed']} km/h."
    )
  answer2 = (
      f"In {city}, it is currently {description}. "
      f"The temperature is {info['temperature']}°C and the wind speed is {info['windspeed']} km/h."
    )
  answer3 = (
      f"{city} is currently experiencing {description}, "
      f"with a temperature of {info['temperature']}°C and wind speed of {info['windspeed']} km/h."
      )

  return random.choice([answer1, answer2, answer3])

## OPEN LIBRARY HELPERS

def call_open_library_search_api(title=None, author=None):
  url = "https://openlibrary.org/search.json"
  params = {}

  if title:
    params["title"] = title
  if author:
    params["author"] = author

  response = requests.get(url, params=params)
  if response.status_code != 200:
    return None

  return response.json()

def extract_book_info(data):
  if data is None or "docs" not in data or len(data["docs"]) == 0:
    return None

  first = data["docs"][0]

  return {
      "title": first.get("title", "Unknown title"),
      "author_name": first.get("author_name", ["Unknown author"]),
      "first_publish_year": first.get("first_publish_year", "Unknown year"),
      "docs": data["docs"]
  }

def get_top_book_candidates(data, limit=10):
    if data is None or "docs" not in data or len(data["docs"]) == 0:
        return []

    candidates = []
    for doc in data["docs"][:limit]:
        candidates.append({
            "title": doc.get("title", "Unknown title"),
            "author": ", ".join(doc.get("author_name", ["Unknown author"])),
            "year": doc.get("first_publish_year", "Unknown year"),
            "doc": doc,
        })
    return candidates

def save_books_db():
    global BOOKS_DB

    try:
        with open(BOOKS_PATH, "r", encoding="utf-8") as f:
            file_books = json.load(f)
    except Exception:
        file_books = {}

    file_books.update(BOOKS_DB)
    BOOKS_DB = file_books

    with open(BOOKS_PATH, "w", encoding="utf-8") as f:
        json.dump(BOOKS_DB, f, indent=2, ensure_ascii=False)


def make_lorem_pages(title: str, author: str | None = None, num_pages: int = 3):
    author_text = f" by {author}" if author else ""
    intro = f"{title}{author_text}\n\n"

    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    )

    pages = []
    for i in range(num_pages):
        if i == 0:
            page_text = intro + f"Page {i + 1}\n\n" + " ".join([filler] * 4)
        else:
            page_text = f"Page {i + 1}\n\n" + " ".join([filler] * 4)
        pages.append(page_text)

    return pages


def add_book_to_local_db(title: str, author: str | None = None, series: str = "OpenLibrary"):
    if not title:
        return None

    existing_title, existing_data = find_book_in_db(title)
    if existing_data:
        return existing_title

    BOOKS_DB[title] = {
        "series": series,
        "pages": make_lorem_pages(title, author=author, num_pages=3),
    }

    save_books_db()
    return title

def get_candidate_page():
    if not pending_book_selection["active"]:
        return []

    start = pending_book_selection["page"] * pending_book_selection["page_size"]
    end = start + pending_book_selection["page_size"]
    return pending_book_selection["results"][start:end]

def advance_candidate_page():
    total = len(pending_book_selection["results"])
    page_size = pending_book_selection["page_size"]
    next_start = (pending_book_selection["page"] + 1) * page_size

    if next_start >= total:
        return None

    pending_book_selection["page"] += 1
    return get_candidate_page()

def handle_book_candidate_selection(index: int):
    global pending_book_selection

    if not pending_book_selection["active"]:
        add_assistant_message("There is no active book selection.")
        return

    page_options = get_candidate_page()
    if index < 0 or index >= len(page_options):
        add_assistant_message("That book selection is invalid.")
        return

    selected = page_options[index]
    doc = selected["doc"]
    intent = pending_book_selection["intent"]
    query = pending_book_selection["query"]

    selected_title = doc.get("title", "Unknown title")
    selected_authors = doc.get("author_name", ["Unknown author"])
    selected_author = selected_authors[0] if selected_authors else "Unknown author"

    if intent == "GetBook":
        add_book_to_local_db(
            selected_title,
            author=selected_author,
            series="OpenLibrary",
        )

    info = {
        "title": selected_title,
        "author_name": selected_authors,
        "first_publish_year": doc.get("first_publish_year", "Unknown year"),
        "docs": [doc],
    }

    slots = {"book_title": query}
    response_text = generate_book_answer(intent, slots, info)

    pending_book_selection = {
        "active": False,
        "intent": None,
        "query": "",
        "results": [],
        "page": 0,
        "page_size": 5,
    }

    clear_book_candidates()
    deliver_assistant_response(response_text, {"intent": intent, "slots": slots})
    transition_after_response({"intent": intent, "slots": slots})

def handle_book_candidate_next_page():
    global pending_book_selection

    if not pending_book_selection["active"]:
        add_assistant_message("There is no active book selection.")
        return

    next_page = advance_candidate_page()

    if next_page is None:
        clear_book_candidates()
        add_assistant_message(
            "I couldn’t find the right match. Try a more specific title or include the author."
        )
        pending_book_selection = {
            "active": False,
            "intent": None,
            "query": "",
            "results": [],
            "page": 0,
            "page_size": 5,
        }
        return

    show_book_candidates(
        pending_book_selection["query"],
        next_page,
        pending_book_selection["page"],
        len(pending_book_selection["results"]),
    )
    deliver_assistant_response(
        "Here are more matches. Please choose one.",
        {"intent": pending_book_selection["intent"], "slots": {}},
    )

def handle_book_candidate_cancel():
    global pending_book_selection

    pending_book_selection = {
        "active": False,
        "intent": None,
        "query": "",
        "results": [],
        "page": 0,
        "page_size": 5,
    }

    clear_book_candidates()
    add_assistant_message("Book selection canceled.")

# Handles e-reader control actions
def fulfill_ereader_control(intent, slots):
    if intent == "OpenBook":
        requested_title = (
            slots.get("title")
            or slots.get("BNAME")
            or slots.get("book_title")
            or slots.get("bname")
            or ""
        ).strip()

        if not requested_title:
            return "Please tell me which book you want to open."

        matched_title, book_data = find_book_in_db(requested_title)
        if not book_data:
            return f"Sorry, I could not find {requested_title} in the available books."

        ereader_state["current_book"] = matched_title
        ereader_state["current_page"] = 0
        ereader_state["reading_session"] = True

        return f"Opened {matched_title}. You are now on page 1."

    elif intent == "NextPage":
        if not ereader_state["reading_session"] or not ereader_state["current_book"]:
            return "You need to open a book first."

        title = ereader_state["current_book"]
        pages = BOOKS_DB[title]["pages"]

        if ereader_state["current_page"] < len(pages) - 1:
            ereader_state["current_page"] += 1
            return f"Turned to the next page. You are now on page {ereader_state['current_page'] + 1}."

        return "You are already on the last page."
    
    elif intent == "PreviousPage":
        if not ereader_state["reading_session"] or not ereader_state["current_book"]:
            return "You need to open a book first."

        if ereader_state["current_page"] > 0:
            ereader_state["current_page"] -= 1
            return f"Turned to the previous page. You are now on page {ereader_state['current_page'] + 1}."

        return "You are already on the first page."

    elif intent == "IncreaseFont":
        return "Increased the font size."

    elif intent == "DecreaseFont":
        return "Decreased the font size."

    elif intent == "IncreaseBrightness":
        return "Increased the brightness."

    elif intent == "DecreaseBrightness":
        return "Decreased the brightness."

    elif intent == "EnableDarkMode":
        ereader_state["dark_mode"] = True
        return "Dark mode has been enabled."

    return "Unknown e-reader control request."

# Executes intent using APIs/system
def fulfill_intent(request):
    intent = request["intent"]
    slots = request.get("slots", {})

    if intent in ["Greeting", "Goodbye"]:
        return generate_basic_answer(intent, slots)
    
    elif intent in ["SetTimer", "SetAlarm"]:
        duration_text = slots.get("duration")
        seconds = duration_to_seconds(duration_text)

        if seconds is None:
            return "I understood that you want a timer, but I couldn’t tell the duration."

        return generate_basic_answer(intent, {"duration": duration_text})

    elif intent == "AskForWeather":
        city = (
            slots.get("location")
            or slots.get("city")
            or slots.get("LOCATION")
            or slots.get("CITY")
        )

        if city:
            city = city.strip(" ?.,!")

        if not city:
            city = "Ottawa"

        coords = get_coordinates(city)
        if coords is None:
            return "Sorry, I could not find that city"

        lat, lon = coords
        weather_data = call_weather_api(lat, lon)
        if weather_data is None:
            return "Sorry, I could not retrieve the weather right now"

        info = extract_weather_info(weather_data)
        return generate_weather_answer(city, info)

    elif intent in ["GetBook", "GetAuthor", "GetPublishingYear"]:
        book_title = (
            slots.get("book_title")
            or slots.get("title")
            or slots.get("book_name")
            or slots.get("BNAME")
            or slots.get("bname")
            or slots.get("B-NAME")
        )

        if not book_title:
            return "Sorry, I need a book title for that request"

        data = call_open_library_search_api(title=book_title)
        candidates = get_top_book_candidates(data, limit=10)

        if not candidates:
            return "Sorry, I could not find that book."

        global pending_book_selection
        pending_book_selection = {
            "active": True,
            "intent": intent,
            "query": book_title,
            "results": candidates,
            "page": 0,
            "page_size": 5,
        }

        first_page = candidates[:5]
        show_book_candidates(book_title, first_page, page=0, total=len(candidates))
        return "__AWAITING_BOOK_SELECTION__"

    elif intent == "GetBooksByAuthor":
        author_name = (
            slots.get("author_name")
            or slots.get("name")
            or slots.get("NAME")
        )

        if not author_name:
            return "Sorry, I need an author name for that request"

        data = call_open_library_search_api(author=author_name)
        info = extract_book_info(data)

        if info is None:
            return "Sorry, I could not find any books by that author."

        return generate_book_answer(intent, slots, info)

    elif intent in [
        "OpenBook",
        "NextPage",
        "PreviousPage",
        "EnableDarkMode",
        "IncreaseFont",
        "DecreaseFont",
        "IncreaseBrightness",
        "DecreaseBrightness",
    ]:
        return fulfill_ereader_control(intent, slots)

    elif intent == "OOS":
        return "Sorry, that request is outside the scope of this assistant."

    return "Sorry, I do not know how to fulfill that intent."



# Helper that syncs fulfillment to UI
import re

def duration_to_seconds(duration_value):
    if not duration_value:
        return None

    text = str(duration_value).strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()

    word_to_num = {
        "a": 1,
        "an": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
    }

    if text in {"half an hour", "half hour"}:
        return 30 * 60

    # normalize simple word numbers
    for word, num in sorted(word_to_num.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{re.escape(word)}\b", str(num), text)

    # handle "1 minute 30 seconds"
    matches = re.findall(r"(\d+)\s*(hour|hours|hr|hrs|minute|minutes|min|mins|second|seconds|sec|secs)", text)

    if matches:
        total = 0
        for value, unit in matches:
            value = int(value)
            if unit.startswith(("hour", "hr")):
                total += value * 3600
            elif unit.startswith(("minute", "min")):
                total += value * 60
            elif unit.startswith(("second", "sec")):
                total += value
        return total if total > 0 else None

    # fallback: bare number defaults to minutes
    bare = re.fullmatch(r"(\d+)", text)
    if bare:
        return int(bare.group(1)) * 60

    # fallback: phrases like "for 5"
    partial = re.search(r"\bfor\s+(\d+)\b", text)
    if partial:
        return int(partial.group(1)) * 60

    return None

def update_ui_from_intent(intent_result, fulfillment_text):
    if not intent_result:
        return

    intent = intent_result.get("intent")
    slots = intent_result.get("slots", {})

    if intent in ["SetTimer", "SetAlarm"]:
        duration_text = slots.get("duration")
        seconds = duration_to_seconds(duration_text)

        if seconds is not None:
            start_timer(seconds)
        else:
            return

    elif intent == "OpenBook":
        matched_title = ereader_state.get("current_book")
        if matched_title and matched_title in BOOKS_DB:
            open_book(matched_title, BOOKS_DB[matched_title].get("pages", []))

    elif intent == "NextPage":
        next_page()

    elif intent == "PreviousPage":
        prev_page()

    elif intent == "EnableDarkMode":
        toggle_reader_theme()

    elif intent == "IncreaseFont":
        increase_font_size()

    elif intent == "DecreaseFont":
        decrease_font_size()

    elif intent == "IncreaseBrightness":
        increase_brightness()

    elif intent == "DecreaseBrightness":
        decrease_brightness()


# =========================
# Section 6: Answer Generation
# =========================

def generate_book_answer(intent, slots, info):
    if intent == "GetBook":
        title = info["title"]
        author = info["author_name"][0]

        answer1 = f"I found {title} by {author} and added it to your available books."
        answer2 = f"The book I found is {title}, written by {author}. I also added it to your available books."
        answer3 = f"I found a result for that request: {title} by {author}, and it is now in your available books."
        return random.choice([answer1, answer2, answer3])

    elif intent == "GetAuthor":
        requested_title = slots.get("book_title", info["title"])
        author = info["author_name"][0]
        print(f"DEBUG: requested_title='{requested_title}', author='{author}'")  # DEBUG

        answer1 = f"The author of {requested_title} is {author}."
        answer2 = f"{requested_title} was written by {author}."
        answer3 = f"The listed author for {requested_title} is {author}."
        try:
            answer = generate_qwen_answer(intent, slots , info)
            print(f"DEBUG LLM RAW ANSWER: {repr(answer)}")
            print(f"DEBUG LLM ANSWER LEN: {len(answer) if answer is not None else 'None'}")
            if answer and len(answer) <= 300:
                return answer
        except Exception:
            pass # Fall back to templates if LLM fails
            
        return random.choice([answer1, answer2, answer3])

    elif intent == "GetPublishingYear":
        requested_title = slots.get("book_title", info["title"])
        year = info["first_publish_year"]

        answer1 = f"{requested_title} was first published in {year}."
        answer2 = f"The first publish year listed for {requested_title} is {year}."
        answer3 = f"{requested_title} appears to have first been published in {year}."
        try:
            answer = generate_qwen_answer(intent, slots , info)
            print(f"DEBUG LLM RAW ANSWER: {repr(answer)}")
            print(f"DEBUG LLM ANSWER LEN: {len(answer) if answer is not None else 'None'}")
            if answer and len(answer) <= 300:
                return answer
        except Exception:
            pass # Fall back to templates if LLM fails
            
        return random.choice([answer1, answer2, answer3])

    elif intent == "GetBooksByAuthor":
        author = slots.get("author_name", "that author")
        titles = []

        for doc in info["docs"][:5]:
            if "title" in doc:
                titles.append(doc["title"])

        if len(titles) == 0:
            return f"I could not find any books by {author}."

        titles_text = ", ".join(titles)

        answer1 = f"Some books by {author} include {titles_text}."
        answer2 = f"I found these books by {author}: {titles_text}."
        answer3 = f"{author} has books such as {titles_text}."
        try:
            answer = generate_qwen_answer(intent, slots , info)
            print(f"DEBUG LLM RAW ANSWER: {repr(answer)}")
            print(f"DEBUG LLM ANSWER LEN: {len(answer) if answer is not None else 'None'}")
            if answer and len(answer) <= 300:
                return answer
        except Exception as e:
            print(f"QWEN ERROR: {e}")
            pass # Fall back to templates if LLM fails
            
        return random.choice([answer1, answer2, answer3])

    return "Sorry, I could not generate a book answer."


def generate_basic_answer(intent, slots=None):
    slots = slots or {}

    if intent == "Greeting":
        return random.choice([
            "Hello! How can I help you today?",
            "Hi there! What would you like me to do?",
            "Hey! I'm ready to help.",
        ])

    elif intent == "Goodbye":
        return random.choice([
            "Goodbye!",
            "See you later!",
            "Bye! Have a great day!",
        ])

    elif intent in ["SetTimer", "SetAlarm"]:
        duration = slots.get("duration", "the requested amount of time")
        return random.choice([
            f"Timer set for {duration}.",
            f"Alright, I started a timer for {duration}.",
            f"Your timer for {duration} is now running.",
        ])

    return "Sorry, I could not generate a response."


# LLM part for answer generation
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)


def build_qwen_prompt(intent,slots,info = None):
    # intent = request["intent"]
    # slots = request.get("slots", {})

    if intent == "Greeting":
        task = "Generate a short greeting response."

    elif intent == "Goodbye":
        task = "Generate a short goodbye response."

    elif intent in ["SetTimer", "SetAlarm"]:
        duration = slots.get("duration", "the requested time")
        task = f"Generate a short response confirming that a timer was set for {duration}."

    elif intent == "GetAuthor":
        book_title = slots.get("book_title") or  slots.get("bname") or slots.get("BNAME") 
        task = (
            f"Generate a short response giving the author of the book titled '{book_title}'. "
            f"Do not ask follow-up questions."
        )

    elif intent == "GetBooksByAuthor":
        author_name = slots.get("author_name") or slots.get("name") or slots.get("NAME")
<<<<<<< Updated upstream
        if info and info.get("docs"):
            titles = [d["title"] for d in info["docs"][:5] if "title" in d]
            titles_text = ", ".join(titles)
            task = (
                f"The following books were found for {author_name}: {titles_text}. "
                f"State this as one short natural sentence. "
                f"Use only these titles — do not add any others."
                    )
        else:
            task = f"Generate a short response listing books by {author_name}."
=======
        task = (
            f"Generate a short response giving the titles of books by {author_name}. "
            f"Do not ask follow-up questions."
        )

>>>>>>> Stashed changes
    elif intent == "GetPublishingYear":
        book_title = slots.get("book_title") or  slots.get("bname") or slots.get("BNAME")
        task = (
            f"Generate a short response giving the first publishing year of the book titled '{book_title}'. "
            f"Do not ask follow-up questions."
        )

    elif intent == "AskForWeather":
        city = slots.get("location") or slots.get("city") or "Ottawa"
        task = (
            f"Generate a short weather response for {city}. "
            f"Do not ask follow-up questions."
        )

    elif intent == "OpenBook":
        title = slots.get("title") or slots.get("book_title") or "Unknown Book"
        task = f"Generate a short response confirming that the book '{title}' was opened on page 1."

    elif intent == "NextPage":
        task = "Generate a short response confirming that the page was advanced."

    elif intent == "EnableDarkMode":
        task = "Generate a short response confirming that dark mode was enabled."

    else:
        task = "Generate a short helpful assistant response."

    return (
        "You are a voice assistant.\n"
        "Respond in one short sentence.\n"
        "Do not ask follow-up questions.\n"
        "Do not mention intents or slots.\n"
        f"{task}"
    )


def generate_qwen_answer(intent, slots, info = None, max_new_tokens=256):
    prompt = build_qwen_prompt(intent, slots, info = info)
    print(f"DEBUG LLM PROMPT:\n{prompt}\n")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = qwen_tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,       # slight variety vs greedy
            temperature=0.7,
            top_p=0.9
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = qwen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer



# =========================
# Section 7: Text-to-Speech (Kokoro)
# =========================


# Load once at module level
_kokoro_model = None

def get_kokoro():
    global _kokoro_model
    if _kokoro_model is None:
        _kokoro_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    return _kokoro_model


def choose_tts_strategy(request):
    intent = request["intent"]
    if intent in ["Greeting", "Goodbye"]:
        return "reinforce_positive"
    elif intent in ["SetTimer", "SetAlarm", "OpenBook", "NextPage", "EnableDarkMode"]:
        return "reassuring"
    elif intent in ["AskForWeather", "GetAuthor", "GetPublishingYear", "GetBook", "GetBooksByAuthor"]:
        return "informative"
    elif intent == "OOS":
        return "neutralize"
    return "neutral_response"


# Maps strategy to Kokoro speed
STRATEGY_TO_SPEED = {
    "reinforce_positive": 1.2,
    "supportive":         0.85,
    "calm_down":          0.8,
    "reassuring":         0.92,
    "neutralize":         0.95,
    "informative":        1.0,
    "neutral_response":   1.0,
    "clarify_intent":     0.95,
    "gentle_probe":       0.88,
}


def speak_text_response(response_text: str, intent_result: dict | None = None):
    if not response_text or not response_text.strip():
        return None

    request = intent_result or {"intent": "OOS", "slots": {}}
    strategy = choose_tts_strategy(request)
    speed = STRATEGY_TO_SPEED.get(strategy, 1.0)
    # output_file = "assistant_response.wav"

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_file = tmp.name
    tmp.close()

    print(f"DEBUG: TTS generating for strategy={strategy}, speed={speed}")

    try:
        kokoro = get_kokoro()
        samples, sample_rate = kokoro.create(
            response_text,
            voice="af_bella",   # change voice here if desired
            speed=speed,
            lang="en-us",
        )

        sf.write(output_file, samples, sample_rate)
        print("DEBUG: TTS generation complete.")

    except Exception as e:
        import traceback
        print(f"DEBUG: TTS generation error: {e}")
        traceback.print_exc()
        return None

    print("DEBUG: Starting audio playback...")
    try:
        stop_playback()
        play_audio(output_file, wait=False)
        print("DEBUG: Audio playback complete.")
    except Exception as e:
        import traceback
        print(f"DEBUG: Audio playback error: {e}")
        traceback.print_exc()
    
    return output_file


def deliver_assistant_response(response_text: str, intent_result: dict | None = None):
    print("DEBUG: before add_assistant_message inside deliver_assistant_response")
    add_assistant_message(response_text)
    print("DEBUG: after add_assistant_message inside deliver_assistant_response")

    def _tts_worker():
        try:
            print("DEBUG: before speak_text_response inside tts worker")
            speak_text_response(response_text, intent_result)
            print("DEBUG: after speak_text_response inside tts worker")
        except Exception as e:
            import traceback
            print(f"DEBUG: TTS worker error: {e}")
            traceback.print_exc()

    threading.Thread(target=_tts_worker, daemon=True).start()

    
# =========================
# UI Sync up
# =========================

#UI Imports

from ui_bridge import (
    set_locked,
    set_awake,
    set_listening,
    add_user_message,
    add_assistant_message,
    clear_history,
    start_timer,
    pause_timer,
    stop_timer,
    reset_timer,
    open_book,
    close_book,
    next_page,
    prev_page,
    increase_font_size,
    decrease_font_size,
    increase_brightness,
    decrease_brightness,
    toggle_reader_theme,
    set_input_text,
    show_book_candidates,
    clear_book_candidates,
    select_book_candidate,
    next_book_candidate_page,
)

def transition_after_response(intent_result: dict | None):
    global pipeline_control_state

    intent = (intent_result or {}).get("intent")

    if intent == "Goodbye":
        reset_pipeline_state()
        return

    pipeline_control_state["current_stage"] = "asr"
    pipeline_control_state["transcript"] = ""        # clear old transcript
    pipeline_control_state["intent_result"] = None   # clear old intent
    pipeline_control_state["fulfillment_result"] = None  # clear old result

    set_awake(True)
    set_listening(False)   # make sure listening indicator resets


from listen_and_transcribe import listen_until_silence

# Ensures that Input is being done in order
def handle_text_bypass_input(text: str):
    global pipeline_control_state

    if text.startswith("__select_candidate__:"):
        try:
            idx = int(text.split(":")[-1])
            handle_book_candidate_selection(idx)
        except ValueError:
            add_assistant_message("Invalid book selection.")
        return

    if text == "__next_candidate_page__":
        handle_book_candidate_next_page()
        return

    if text == "__cancel_candidate_selection__":
        handle_book_candidate_cancel()
        return

    text = (text or "").strip()
    if not text:
        return

    add_user_message(text)

    stage = pipeline_control_state["current_stage"]

    if stage == "verification":
        handle_verification_bypass(text)
    elif stage == "wakeword":
        handle_wakeword_bypass(text)
    elif stage in ["asr", "awaiting_asr_confirmation"]:
        handle_asr_bypass(text)
    elif stage == "intent":
        handle_intent_bypass(text)
    elif stage == "fulfillment":
        handle_fulfillment_bypass(text)
    elif stage == "answer":
        handle_answer_bypass(text)
    else:
        add_assistant_message("Invalid pipeline stage.")


def handle_verification_bypass(text: str):
    global pipeline_control_state

    if text == BYPASS_PIN:
        pipeline_control_state["verification_passed"] = True
        pipeline_control_state["current_stage"] = "wakeword"
        set_locked(False)
        add_assistant_message("Verification bypass accepted. You can now enter the wake word bypass.")
    else:
        set_locked(True)
        add_assistant_message("Verification bypass failed. Enter the correct PIN first.")

def handle_wakeword_bypass(text: str):
    global pipeline_control_state

    if not pipeline_control_state["verification_passed"]:
        add_assistant_message("You must complete user verification first.")
        return

    if text.lower() == WAKEWORD_BYPASS_TEXT:
        pipeline_control_state["wakeword_passed"] = True
        pipeline_control_state["current_stage"] = "asr"
        set_awake(True)
        add_assistant_message("Wake word bypass accepted. You can now speak your command or enter an ASR correction later.")
    else:
        set_awake(False)
        add_assistant_message("Wake word bypass failed. Enter the correct wake word.")


# Handle how to bypass ASR to write the texts
def handle_asr_bypass(text: str):
    global pipeline_control_state

    cleaned = (text or "").strip()
    if not cleaned:
        add_assistant_message("I didn’t receive any text.")
        return

    pipeline_control_state["transcript"] = cleaned

    intent_result = predict_from_text(
        cleaned,
        intent_model,
        intent_tokenizer,
        id2intent,
        id2slot,
        device,
    )
    
    def fallback_book_intent(transcript: str):
        t = (transcript or "").strip().lower()

        for title in BOOKS_DB.keys():
            if title.lower() in t:
                return {
                    "intent": "OpenBook",
                    "confidence": 0.99,
                    "slots": {"title": title},
                }

        return None

    if not intent_result or intent_result.get("intent") == "OOS":
        fallback_result = fallback_book_intent(cleaned)
        if fallback_result:
            intent_result = fallback_result

    pipeline_control_state["intent_result"] = intent_result

    if not intent_result or not intent_result.get("intent") or intent_result.get("intent") == "OOS":
        pipeline_control_state["current_stage"] = "asr"
        pipeline_control_state["wakeword_passed"] = True
        set_awake(True)
        add_assistant_message("I didn’t catch what you meant. Please type that again.")
        return

    fulfillment_result = fulfill_intent(intent_result)
    pipeline_control_state["fulfillment_result"] = fulfillment_result
    if fulfillment_result == "__AWAITING_BOOK_SELECTION__":
        deliver_assistant_response(
            f"I found multiple matches for {pending_book_selection['query']}. Please select the correct option.",
            intent_result,
        )
        return
    update_ui_from_intent(intent_result, fulfillment_result)

    # special case: goodbye should reset immediately after responding
    if intent_result.get("intent") == "Goodbye":
        deliver_assistant_response(fulfillment_result, intent_result)
        reset_pipeline_state()
        return
    
    deliver_assistant_response(fulfillment_result, intent_result)
    # set_input_text("")
    def _delayed_transition():
        import time
        time.sleep(0.15)
        transition_after_response(intent_result)

    threading.Thread(target=_delayed_transition, daemon=True).start()
    


def handle_intent_bypass(text: str):
    global pipeline_control_state

    try:
        data = json.loads(text)
        intent = data.get("intent")
        slots = data.get("slots", {})

        if not intent:
            add_assistant_message("Manual intent bypass failed. JSON must include an intent.")
            return

        pipeline_control_state["intent_result"] = {
            "intent": intent,
            "slots": slots,
            "confidence": 1.0,
        }
        pipeline_control_state["current_stage"] = "fulfillment"
        add_assistant_message(f"Manual intent accepted: {intent}")
    except Exception:
        add_assistant_message(
            'Enter manual intent/slots as JSON, for example: {"intent":"SetTimer","slots":{"duration":"10 minutes"}}'
        )

def handle_fulfillment_bypass(text: str):
    global pipeline_control_state

    pipeline_control_state["fulfillment_result"] = text
    pipeline_control_state["current_stage"] = "answer"
    add_assistant_message(text)
    speak_text_response(text, pipeline_control_state.get("intent_result"))
    transition_after_response(pipeline_control_state.get("intent_result"))
    

def handle_answer_bypass(text: str):
    global pipeline_control_state

    pipeline_control_state["final_answer"] = text
    add_assistant_message(text)
    transition_after_response(pipeline_control_state.get("intent_result"))

# Thats for typed dashboard input should call
def handle_text_command(text: str):
    set_listening(True)

    try:
        cleaned = (text or "").strip()
        if not cleaned:
            add_assistant_message("I didn’t receive any input.")
            return None

        add_user_message(cleaned)

        # Since typed input bypasses speaker verification and wake word,
        # we treat it like an already-unlocked, awake command path.
        intent_result = predict_from_text(
            cleaned,
            intent_model,
            intent_tokenizer,
            id2intent,
            id2slot,
            device,
        )

        fulfillment_result = fulfill_intent(intent_result)

        if fulfillment_result == "__AWAITING_BOOK_SELECTION__":
            deliver_assistant_response(
                f"I found multiple matches for {pending_book_selection['query']}. Please select the correct option.",
                intent_result,
            )
            return {
                "transcript": cleaned,
                "intent_result": intent_result,
                "fulfillment_result": fulfillment_result,
            }

        update_ui_from_intent(intent_result, fulfillment_result)
        deliver_assistant_response(fulfillment_result, intent_result)

        return {
            "transcript": cleaned,
            "intent_result": intent_result,
            "fulfillment_result": fulfillment_result,
        }

    except Exception as e:
        add_assistant_message(f"An error occurred: {e}")
        return None

    finally:
        set_locked(False)
        set_listening(False)

def handle_live_voice_pipeline():
    global pipeline_control_state, live_voice_busy

    if live_voice_busy:
        print("handle_live_voice_pipeline ignored: already running")
        return None

    live_voice_busy = True
    set_listening(True)

    temp_wav_path = None
    verification_wav = None
    wakeword_wav = None
    asr_wav = None

    try:
        stage = pipeline_control_state["current_stage"]
        print("CURRENT STAGE:", stage)

        # -------------------------
        # Capture microphone audio once
        # -------------------------
        capture = listen_until_silence(
            asr_model=asr_model,
            model_type="openai_whisper",
            language="en",
            #device=10,
            keep_file=True,   # IMPORTANT: keep the file so verification/wakeword can use it
        )

        temp_wav_path = capture.get("wav_path")
        verification_wav = None
        wakeword_wav = None
        asr_wav = None

        if not temp_wav_path:
            add_assistant_message("No audio was recorded.")
            return None

        # -------------------------
        # STAGE 1: verification
        # Use your real voice verification function
        # -------------------------
        if stage == "verification":
            verification_wav = clone_temp_wav(temp_wav_path)
            verification_result = verify_any_user(verification_wav)

            if not verification_result["accepted"]:
                set_locked(True)
                set_awake(False)
                add_assistant_message(
                    f"Verification failed. Best match: {verification_result['matched_user']} "
                    f"(score={verification_result['best_final_score']:.3f})"
                )
                return {
                    "stage": "verification",
                    "verification_result": verification_result,
                }

            pipeline_control_state["verification_passed"] = True
            pipeline_control_state["current_stage"] = "wakeword"
            set_locked(False)
            set_awake(False)

            add_assistant_message(
                f"Verification accepted. Identified as {verification_result['matched_user']}. "
                f"Now say the wake word."
            )

            return {
                "stage": "verification",
                "verification_result": verification_result,
            }

        # -------------------------
        # STAGE 2: wakeword
        # Use your real wakeword detector
        # -------------------------
        elif stage == "wakeword":
            wakeword_wav = clone_temp_wav(temp_wav_path)
            wakeword_result = detect_wakeword(
                test_file=wakeword_wav,
                typed_wakeword=None,
                wake_model=wake_model,
            )

            if not wakeword_result["wake_detected"]:
                set_awake(False)
                add_assistant_message(
                    f"Wake word not detected. Score={wakeword_result['score']:.3f}"
                )
                return {
                    "stage": "wakeword",
                    "wakeword_result": wakeword_result,
                }

            pipeline_control_state["wakeword_passed"] = True
            pipeline_control_state["current_stage"] = "asr"
            set_awake(True)

            add_assistant_message("Wake word detected. Now say your command.")

            return {
                "stage": "wakeword",
                "wakeword_result": wakeword_result,
            }

        # -------------------------
        # STAGE 3: command speech
        # Intent, fulfillment, answer all happen automatically here
        # -------------------------

        elif stage == "asr":
            asr_wav = clone_temp_wav(temp_wav_path)
            transcript, _ = transcribe_audio_file(asr_wav, asr_model)

            if not transcript.strip():
                add_assistant_message("I didn’t catch that.")
                pipeline_control_state["current_stage"] = "asr"
                pipeline_control_state["wakeword_passed"] = True
                set_awake(True)
                return None

            transcript = transcript.strip()
            pipeline_control_state["transcript"] = transcript

            # put transcript into the dashboard text area for review
            set_input_text(transcript)

            # move to a separate waiting stage
            pipeline_control_state["current_stage"] = "awaiting_asr_confirmation"
            pipeline_control_state["wakeword_passed"] = True
            set_awake(True)

            add_assistant_message(
                "I transcribed your speech. Please review or edit it in the text box, then press Submit."
            )

            return {
                "stage": "asr",
                "transcript": transcript,
            }

        else:
            # any unexpected stage: reset to wakeword instead of blocking
            add_assistant_message(
                f"Unexpected stage '{stage}'. Resetting to wakeword."
            )
            pipeline_control_state["current_stage"] = "wakeword"
            pipeline_control_state["wakeword_passed"] = False
            set_awake(False)
            return None

    except Exception as e:
        add_assistant_message(f"An error occurred: {e}")

        # stay awake unless the intent was Goodbye
        if pipeline_control_state.get("wakeword_passed"):
            pipeline_control_state["current_stage"] = "asr"
            set_awake(True)
        else:
            set_awake(False)

        return None

    finally:
        safe_remove_file(verification_wav)
        safe_remove_file(wakeword_wav)
        safe_remove_file(asr_wav)
        safe_remove_file(temp_wav_path)

        set_listening(False)
        live_voice_busy = False


def reset_pipeline_state():
    global pipeline_control_state, ereader_state, pending_book_selection

    pipeline_control_state = {
        "current_stage": "verification",
        "verification_passed": False,
        "wakeword_passed": False,
        "transcript": "",
        "intent_result": None,
        "fulfillment_result": None,
        "final_answer": None,
    }

    ereader_state = {
        "current_book": None,
        "current_page": 0,
        "reading_session": False,
        "dark_mode": False,
    }

    pending_book_selection = {
        "active": False,
        "intent": None,
        "query": "",
        "results": [],
        "page": 0,
        "page_size": 5,
    }

    clear_history()
    set_locked(True)
    set_awake(False)
    set_listening(False)
    reset_timer()
    close_book()
    add_assistant_message("System reset. Please verify your identity.")


# =========================
# Full Assistant Pipeline
# =========================

# Runs all modules sequentially
def run_full_pipeline(
    verification_file: str,
    wakeword_file: Optional[str] = None,
    typed_wakeword: Optional[str] = None,
    verification_pin: Optional[str] = None,
    wake_model=None,
    asr_model=None,
    intent_model=None,
    tokenizer=None,
    id2intent=None,
    id2slot=None,
    device=None,
    manual_transcript: Optional[str] = None,
    manual_intent: Optional[str] = None,
    manual_slots: Optional[dict] = None,
):
    state = AssistantState()

    verification_file = resolve_audio_path(verification_file, base_dir=TEST_AUDIO_DIR)
    if wakeword_file is not None:
        wakeword_file = resolve_audio_path(wakeword_file, base_dir=WAKEWORD_DATASET_DIR)

    verification_result = verify_any_user(verification_file, pin=verification_pin)
    state.lock_state = verification_result["state_after"]
    if not verification_result["accepted"]:
        return {
            "state": state,
            "verification_result": verification_result,
            "wakeword_result": None,
            "transcript": None,
            "intent_result": None,
            "fulfillment_result": None,
        }

    wakeword_result = detect_wakeword(
        test_file=wakeword_file or verification_file,
        typed_wakeword=typed_wakeword,
        wake_model=wake_model,
    )
    state.wake_state = wakeword_result["state_after"]
    if not wakeword_result["wake_detected"]:
        return {
            "state": state,
            "verification_result": verification_result,
            "wakeword_result": wakeword_result,
            "transcript": None,
            "intent_result": None,
            "fulfillment_result": None,
        }

    if manual_transcript is not None:
        transcript = manual_transcript
    elif asr_model is not None and wakeword_file is not None:
        transcript, _ = transcribe_audio_file(wakeword_file, asr_model)
    else:
        transcript = ""

    state.last_transcript = transcript

    if manual_intent is not None:
        intent_result = {"intent": manual_intent, "slots": manual_slots or {}}
    elif transcript and all(x is not None for x in [intent_model, tokenizer, id2intent, id2slot, device]):
        intent_result = predict_from_text(transcript, intent_model, tokenizer, id2intent, id2slot, device)
    else:
        intent_result = None

    if intent_result:
        state.last_intent = intent_result["intent"]
        state.last_slots = intent_result["slots"]
        fulfillment_result = fulfill_intent(intent_result)
    else:
        fulfillment_result = None

    return {
        "state": state,
        "verification_result": verification_result,
        "wakeword_result": wakeword_result,
        "transcript": transcript,
        "intent_result": intent_result,
        "fulfillment_result": fulfillment_result,
    }


# =========================
# Runtime Initialization
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load speaker profiles
build_speaker_profiles()

# Load ASR
asr_model = load_whisper_asr("base")

# Load intent model + tokenizer + label maps
intent_model, intent_tokenizer, id2intent, id2slot = load_intent_model(
    export_dir="saved_intent_model",
    device=device
)

wake_model = build_wakeword_cnn((13, 91, 1))
wake_model.load_weights("saved_intent_model/wake_model.weights.h5")




# # =========================
# # Books
# # =========================

def find_book_in_db(requested_title: str):
    global BOOKS_DB

    if not requested_title:
        return None, None

    try:
        with open(BOOKS_PATH, "r", encoding="utf-8") as f:
            BOOKS_DB = json.load(f)
    except Exception:
        BOOKS_DB = {}

    requested_title = requested_title.strip().lower()

    # exact match first
    for title, data in BOOKS_DB.items():
        if title.lower() == requested_title:
            return title, data

    # partial match second
    for title, data in BOOKS_DB.items():
        if requested_title in title.lower():
            return title, data

    return None, None