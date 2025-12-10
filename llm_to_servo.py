"""
Voice-controlled servo simulator using Whisper.cpp for speech-to-text
and a local LLM (Ollama) for command interpretation.

Requirements:
    pip install sounddevice numpy ollama
    brew install whisper-cpp
    ollama pull gemma3
    ollama serve

Download Whisper tiny model:
    ggml-tiny.en.bin
"""

import time
import json
import re
import os
import wave
import sounddevice as sd
import numpy as np
from scipy.signal import resample
from typing import Dict, Optional

from ollama import chat

# ---------------- CONFIG -------------------

MODEL_NAME = "gemma3:1b"

WHISPER_MODEL_PATH = "ggml-tiny.en.bin"
TEMP_WAV_PATH = "speech.wav"
SAMPLE_RATE = 16000

MIN_ANGLE = 0
MAX_ANGLE = 180

START_PAN = 90
START_TILT = 90

SIMULATION_SECONDS = 5

# ---------------- HELPERS -------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    return json.loads(match.group(0))


def record_audio() -> str:
    """Push-to-talk recording and saving to WAV at 16 kHz."""
    input("\nPress ENTER to START recording...")
    print("Recording... Press ENTER to STOP.")

    chunks = []
    device_rate = int(sd.query_devices(kind="input")["default_samplerate"])

    def callback(indata, frames, t, status):
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=device_rate,
        channels=1,
        dtype="int16",
        callback=callback
    ):
        input()

    print("Recording stopped.")

    audio = np.concatenate(chunks, axis=0)
    target_len = int(len(audio) * SAMPLE_RATE / device_rate)
    audio_resampled = resample(audio, target_len).astype(np.int16)

    with wave.open(TEMP_WAV_PATH, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_resampled.tobytes())

    return TEMP_WAV_PATH


def run_whisper(audio_path: str) -> str:
    """Runs Whisper.cpp tiny model in CPU-only mode."""
    txt_path = audio_path + ".txt"

    # Remove old file if exists
    if os.path.exists(txt_path):
        os.remove(txt_path)

    cmd = (
        f"whisper-cli "
        f"-m {WHISPER_MODEL_PATH} "
        f"-f {audio_path} "
        f"-ng "
        f"-otxt"
    )

    os.system(cmd)

    # Whisper writes to audio.wav.txt
    if not os.path.exists(txt_path):
        print("Whisper output file missing:", txt_path)
        return ""

    with open(txt_path, "r") as f:
        text = f.read().strip()

    return text


# ---------------- LLM COMMAND PARSER -------------------

def get_servo_plan_from_llm(text: str, pan: float, tilt: float) -> Optional[Dict]:
    """
    Ask the LLM to turn a natural language command + current servo state
    into a JSON movement plan.
    """

    system_prompt = """
You control a 2-axis servo turret: pan and tilt.

Your job:
- Read the user's command.
- Look at the current pan and tilt angles.
- Decide how the turret should move.
- Output a JSON object ONLY.

JSON format:
{
  "mode": "delta" or "absolute",
  "pan": <number>,
  "tilt": <number>
}

Meanings:
- mode = "delta": move by that amount relative to current angles.
- mode = "absolute": go directly to those angles.
- pan: positive = right, negative = left.
- tilt: positive = up, negative = down.

Physical limits:
- pan angle range: 0 to 180 degrees.
- tilt angle range: 0 to 180 degrees.
- You must choose values that respect these limits.
- Reasonable delta moves are usually much smaller than the full range:
  do not use huge deltas unless the user clearly asks for "all the way" or "as far as possible".

High-level behavior (not rigid rules):
- "a little", "slightly", "a bit" → small movement.
- "a lot", "far", "strong" → large movement.
- The closer you are to the limit, the smaller the movement should be in that direction.
- If the user only talks about left/right, you should normally keep tilt unchanged (tilt = 0 in delta mode).
- If the user only talks about up/down, you should normally keep pan unchanged (pan = 0 in delta mode).

You must:
- Use the current pan and tilt values that are provided to you.
- Choose numeric values that are reasonable and proportional to the wording and current state.
- Never move both axes if the user clearly referred to only one axis.
- Never output angles or deltas that would send the turret outside [0, 180].

Output:
- Valid JSON only.
- No backticks.
- No explanations.
"""

    # Pack current state + command as a simple text block
    user_content = (
        f"Current state:\n"
        f"- pan: {pan} degrees\n"
        f"- tilt: {tilt} degrees\n\n"
        f"User command:\n"
        f"{text}\n"
    )

    print("Sending text to LLM...")

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.message.content
    print("LLM raw output:", raw)

    try:
        plan = extract_json(raw)
        return plan
    except Exception as e:
        print("LLM output parse error:", e)
        return None


# ---------------- SERVO SIMULATION -------------------

def simulate_motion(pan, tilt, pan_target, tilt_target):
    steps = max(1, SIMULATION_SECONDS)
    d_pan = (pan_target - pan) / steps
    d_tilt = (tilt_target - tilt) / steps

    print(f"Start pan={pan}, tilt={tilt}")
    print("Simulating movement...")

    for i in range(steps):
        pan += d_pan
        tilt += d_tilt
        print(f"Pan={pan:.1f}, Tilt={tilt:.1f}")
        time.sleep(1)

    return pan, tilt


# ---------------- MAIN -------------------

def main():
    pan = START_PAN
    tilt = START_TILT

    print("=== Voice-controlled LLM Servo System ===")

    while True:
        wav = record_audio()
        text = run_whisper(wav)

        print("\nYou said:", text)

        if not text:
            print("No speech recognized.")
            continue

        plan = get_servo_plan_from_llm(text, pan, tilt)
        if not plan:
            print("Invalid command.")
            continue

        mode = plan["mode"]
        val_pan = float(plan["pan"])
        val_tilt = float(plan["tilt"])

        if mode == "delta":
            pan_target = clamp(pan + val_pan, MIN_ANGLE, MAX_ANGLE)
            tilt_target = clamp(tilt + val_tilt, MIN_ANGLE, MAX_ANGLE)

        elif mode == "absolute":
            pan_target = clamp(val_pan, MIN_ANGLE, MAX_ANGLE)
            tilt_target = clamp(val_tilt, MIN_ANGLE, MAX_ANGLE)

        else:
            print("Unknown mode:", mode)
            continue

        pan, tilt = simulate_motion(pan, tilt, pan_target, tilt_target)

        print("\nNew position:", pan, tilt)
        print("\nSay another command...\n")


if __name__ == "__main__":
    main()
