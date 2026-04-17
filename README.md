# Atlas Voice Assistant & E-Reader Dashboard

An end-to-end **voice-controlled assistant with an interactive e-reader UI**, built for CSI 5180. This system integrates speech processing, NLP, APIs, and a real-time dashboard to create a multimodal assistant experience.

---

## Team

* Farina Salman (300129324)
* Drishya Praveen Cherukunnummal Poothakandi (300516807)
* Ziyad Gaffar (300148116)

---

## Overview

Atlas is a **pipeline-based voice assistant** that supports:

* **User Verification (voice biometrics)**
* **Wake Word Detection**
* **Speech-to-Text (ASR)**
* **Intent Detection + Slot Filling (BERT-based)**
* **Intent Fulfillment (APIs + system actions)**
* **Response Generation (templates + LLM)**
* **Text-to-Speech**
* **Interactive Dashboard UI (Dash)**

The system can handle tasks like:

* Asking for weather
* Book queries (Open Library API)
* Reading books via an e-reader
* Setting timers
* Voice-driven interaction

---

## Installation

### 1. Clone the repo

<pre class="overflow-visible! px-0!" data-start="1529" data-end="1581"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span class="ͼs">git</span><span> clone <your-repo-url></span><br/><span class="ͼs">cd</span><span> <repo-name></span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### 2. Install dependencies

<pre class="overflow-visible! px-0!" data-start="1611" data-end="1654"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>pip install </span><span class="ͼu">-r</span><span> requirements.txt</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### 3. Install FFmpeg (REQUIRED)

This project depends on FFmpeg for audio processing.

* Download: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
* Add it to your system PATH

---

## Running the Project

### Run the Dashboard (main entry point)

<pre class="overflow-visible! px-0!" data-start="1979" data-end="2010"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python dashboard.py</span></div></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This will:

* Start the Dash UI
* Initialize the pipeline
* Enable text + voice interaction

---

## System Architecture

The system follows a  **modular pipeline design** :

### 1. User Verification

* Uses **voice embeddings + MFCC features**
* Combines similarity scores for authentication
* Supports PIN bypass "1234"

### 2. Wake Word Detection

* CNN + LSTM model
* Detects "hey atlas"

### 3. Automatic Speech Recognition

* Uses **Whisper**
* Supports both file-based and live input

### 4. Intent Detection

* Custom **DistilBERT-based joint model**
* Predicts:
* Intent (classification)
* Slots (token labeling)

### 5. Fulfillment

* Weather → Open-Meteo API
* Books → Open Library API
* E-reader → Local JSON database
* Timer + system actions

### 6. Response Generation

* Template-based responses

### 7. Text-to-Speech

* Uses `edge_tts`
* Playback handled via custom audio module

---

## Dashboard Features

Built using  **Dash + Bootstrap** :

* Input + output history
* Real-time system status:
* Locked / Unlocked
* Listening / Not Listening
* Awake / Asleep
* E-reader interface:
  * Page navigation
  * Font size + brightness controls
  * Dark mode
* Timer display

UI updates are handled through an event queue system.

---

## Voice Input System

The system supports:

* Live microphone input
* Silence-based stopping
* Automatic transcription

Handled in:

* `listen_and_transcribe.py`

---

## E-Reader System

* Uses a local JSON database of books
* Supports:
* Open book
* Next/previous page
* UI rendering in dashboard

---

## APIs Used

* **Open-Meteo API** → Weather data
* **Open Library API** → Book metadata

---

## Key Design Decisions

* **Pipeline modularity** → each stage is independent
* **Shared state object** → tracks system status across stages
* **UI bridge (queue-based)** → decouples backend from frontend

---

## Known Limitations

* Intent detection may struggle with ambiguous phrasing
* Wake word model requires more training data
* API latency can delay responses
* Limited local book database due to copyright issues

---

## Future Improvements

* Improve intent detection robustness
* Expand book database coverage
* Optimize latency for API calls
* Add more voice commands
* Enhance UI responsiveness and feedback
