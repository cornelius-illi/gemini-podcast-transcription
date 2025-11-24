# Google Gemini 2.5 Pro for Audio Understanding & Transcription

This workspace contains a Jupyter notebook example and a small CLI tool to download audio from a link (for example, YouTube) and transcribe it using Google Gemini (via the `google-genai` Python client).

Files added:
- `transcribe_from_link.py`: CLI script to download audio, upload to the Files API, request transcription, and post-process the output.
- `requirements.txt`: Python dependencies.
- `setup.sh`: Simple script to create a virtualenv and install dependencies.
- `.env.example`: Example env file for `GOOGLE_API_KEY`.

Quick start

1. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`.

2. Run setup to create a venv and install dependencies:

```bash
./setup.sh
source .venv/bin/activate
```

3. Run the CLI to transcribe a URL:

```bash
python transcribe_from_link.py "https://www.youtube.com/watch?v=..." --speakers "Host,Guest" --output episode.txt
```

Notes & system requirements

- `ffmpeg` must be installed on your system for `yt-dlp` to extract and convert audio. On macOS you can install via Homebrew: `brew install ffmpeg`.

Prompt optimization recommendations

The notebook's prompt is already thorough. To improve reliability and consistency when calling the Gemini transcription model, consider the following optimisations:

- Be explicit about output format: ask for one caption per line in `HH:MM:SS Speaker: text` format. This simplifies downstream parsing.
- Provide a maximum caption duration and a preferred caption length (e.g. "Each caption should cover at most 20 seconds and be one or two short sentences").
- Ask the model to normalize timestamps to `HH:MM:SS` even for inputs like `MM:SS` or `H:MM:SS`.
- If speaker labels are uncertain, ask for both a label (A/B) and a confidence estimate, e.g. "Speaker A (likely host, 70% confidence):" — this can help automated post-processing.
- If you require timestamps synchronized to the audio (not just approximated from speech), request machine-readable JSON in addition to the human transcript. Example: `Return also a JSON array of {"start": "HH:MM:SS","end": "HH:MM:SS","speaker":"...","text":"..."}`.
- Limit verbosity and instruct the model to avoid extra commentary or disclaimers so the output is cleaner for parsing.

Example improved prompt snippet

```
Produce a plain-text transcript with one caption per line. Each line must be exactly:
[HH:MM:SS] SpeakerName: text
Captions should be short (one or two short sentences) and cover at most 20 seconds of audio. If you are unsure of a speaker's real name, label them with a single letter (A, B...). Do not include any markdown or extra commentary. End with [END].
```

If you want, I can integrate an option to request JSON-aligned timestamps as well.
