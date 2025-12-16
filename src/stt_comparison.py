"""
STT Comparison:
- Vosk
- Whisper Tiny (CPU only)
- Whisper Base (CPU only)

Uses push-to-talk and compares all 3 results.

Requirements:
    pip install vosk sounddevice numpy scipy
    brew install whisper-cpp

Models needed in folder:
    ggml-tiny.en.bin
    ggml-base.en.bin
"""

import json
import os
import sounddevice as sd
import time
import wave
import numpy as np
from scipy.signal import resample
from vosk import Model, KaldiRecognizer

# -------- CONFIG --------

VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"

WHISPER_TINY = "../models/ggml-tiny.en.bin"
WHISPER_BASE = "../models/ggml-base.en.bin"

TEMP_WAV_PATH = "recorded_audio.wav"
TARGET_RATE = 16000  # Whisper & Vosk expected sample rate

# ------------------------


def record_audio_push_to_talk() -> None:
    """
    Press ENTER to start and stop recording.
    Writes normalized 16kHz WAV file for Whisper & Vosk.
    """
    input("\nPress ENTER to START speaking...")
    print("Recording... Press ENTER to STOP.")

    recorded_chunks = []

    device_rate = int(sd.query_devices(kind="input")["default_samplerate"])

    def callback(indata, frames, time_info, status):
        recorded_chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=device_rate,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        input()

    print("Recording stopped.")

    audio = np.concatenate(recorded_chunks, axis=0)

    # Resample to 16kHz
    num_samples = int(len(audio) * TARGET_RATE / device_rate)
    resampled = resample(audio, num_samples).astype(np.int16)

    with wave.open(TEMP_WAV_PATH, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(resampled.tobytes())

    print(f"Saved recording to {TEMP_WAV_PATH}")


def run_vosk_stt(model: Model) -> str:
    with wave.open(TEMP_WAV_PATH, "rb") as wf:
        recognizer = KaldiRecognizer(model, wf.getframerate())
        audio_bytes = wf.readframes(wf.getnframes())
        recognizer.AcceptWaveform(audio_bytes)
        text = json.loads(recognizer.Result()).get("text", "")
    return text.strip()


def run_whisper_model(model_path: str) -> str:
    txt_path = TEMP_WAV_PATH + ".txt"

    # Remove old output so results don't get reused
    if os.path.exists(txt_path):
        os.remove(txt_path)

    # CPU-only mode: --no-gpu
    # Disable flash attention so the speed matches Raspberry Pi
    cmd = (
        f"whisper-cli "
        f"-m {model_path} "
        f"-f {TEMP_WAV_PATH} "
        f"--no-gpu "
        f"--no-flash-attn "
        f"-otxt"
    )

    print(f"Running Whisper model: {model_path}")
    os.system(cmd)

    if not os.path.exists(txt_path):
        return ""

    with open(txt_path, "r") as f:
        return f.read().strip()



def main():
    print("Loading Vosk model...")
    vosk_model = Model(VOSK_MODEL_PATH)
    print("Vosk loaded.")

    while True:
        print("\n--- New STT Comparison ---")

        record_audio_push_to_talk()

        # --- Vosk ---
        print("\nRunning Vosk...")
        t1 = time.time()
        vosk_text = run_vosk_stt(vosk_model)
        vosk_time = time.time() - t1

        # --- Whisper Tiny ---
        print("Running Whisper Tiny (CPU)...")
        t2 = time.time()
        tiny_text = run_whisper_model(WHISPER_TINY)
        tiny_time = time.time() - t2

        # --- Whisper Base ---
        print("Running Whisper Base (CPU)...")
        t3 = time.time()
        base_text = run_whisper_model(WHISPER_BASE)
        base_time = time.time() - t3

        # --- Results ---
        print("\n================ RESULTS ================")
        print(f"Vosk         ({vosk_time:.2f}s): '{vosk_text}'")
        print(f"Whisper Tiny ({tiny_time:.2f}s): '{tiny_text}'")
        print(f"Whisper Base ({base_time:.2f}s): '{base_text}'")
        print("=========================================\n")

        again = input("Run again? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
