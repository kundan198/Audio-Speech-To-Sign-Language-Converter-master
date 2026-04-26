# Audio Speech To Sign Language Converter

A Web Application which takes in live audio speech recording as input, converts it into text and displays the relevant Indian Sign Language animations.

> - Front-end using HTML,CSS,JavaScript.
> - Speech recognition using JavaScript Web speech API.
> - Text Preprocessing using Natural Language Toolkit(NLTK).
> - A 3D animation of a character created
>   using Blender 3D tool.

Project Demo Video: https://youtu.be/YiHhD0QGrno

## Prerequisites

> - Python >= 3.7
> - Browser supports Web Speech API
> - Download all required packages for python script A2SL/views.py

## Two-Way Translation (NEW)

This fork adds a **reverse direction**: webcam Sign Language → English text → spoken audio.

- **Speech → Sign** (`/animation/`) — original direction. Speech/text becomes ISL animation videos.
- **Sign → Speech** (`/sign-to-text/`) — new direction. Webcam captures a short ASL motion window, sends frames to a local **WLASL I3D** recognizer first, falls back to **Gemini** when needed, and speaks the recognized text with ElevenLabs or browser TTS.

### Setup
1. `pip install -r requirements.txt` (installs `google-generativeai`, `python-dotenv`, `requests`).
2. Get a **Gemini** API key: https://aistudio.google.com/apikey
3. Get an **ElevenLabs** API key: https://elevenlabs.io/app/settings/api-keys
4. Create a `.env` file in the project root (already gitignored):
   ```
   GEMINI_API_KEY=AIza...
   ELEVENLABS_API_KEY=sk_...
   ELEVENLABS_DEFAULT_VOICE=21m00Tcm4TlvDq8ikWAM
   ```
5. `python manage.py migrate` then `python manage.py runserver`.
6. Optional local WLASL recognizer setup:
   ```
   /Users/rs/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 -m venv .wlasl-venv
   .wlasl-venv/bin/python -m pip install -r requirements-wlasl.txt
   .wlasl-venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='raghuhasan/asl2000-i3d', filename='pytorch_model.bin', local_dir='models/wlasl_i3d')"
   .wlasl-venv/bin/python ml/wlasl_i3d/server.py
   ```
7. Log in, then visit `/conversation/` for the flagship demo.

The local recognizer listens on `http://127.0.0.1:8766` and is used automatically by `/api/recognize-sign/`. Set `SIGN_RECOGNITION_ENGINE=gemini` to skip WLASL and use Gemini only. Set `WLASL_MIN_CONFIDENCE` to tune when the local model returns `[unclear]`.

The MediaPipe scripts load from a CDN, so the browser must have internet access on first load. ElevenLabs voices are fetched dynamically — whatever voices are on your ElevenLabs account will appear in the dropdown.

## Installation Guide:

These instructions will get you download the project and running on your local machine for development and testing purposes.

### Instructions

1. Open the Downloads folder and then open the terminal.
2. From the terminal, run the python file using the command "python manage.py runserver ####" (#### optional port number).
3. From the terminal, it shows localhost address (looks like this "server at http://127.0.0.1:8000/") run on browser.
4. Sign up and start exploring.
5. Click on mic button to record speech.
6. Speech is going to processed and respective animated outputs are shown accordingly and it also support entered text manually.
