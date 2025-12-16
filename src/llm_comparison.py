"""
Voice-controlled servo simulator using Whisper.cpp for speech-to-text
and a local LLM (Ollama) for command interpretation.

Requirements:
    pip install sounddevice numpy ollama scipy
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

# MODEL_NAME = "gemma3:1b"
# MODEL_NAME = "qwen2.5:1.5b-instruct-q4_K_M"
# MODEL_NAME = "phi3:mini"
MODEL_LIST = ["phi3:mini", "gemma3:1b", "qwen2.5:1.5b-instruct-q4_K_M", "phi:2.7b", "qwen3-coder:30b", "deepseek-r1:8b"]


WHISPER_MODEL_PATH = "models/ggml-tiny.en.bin"
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
    """
    Extract first {...} block and parse as JSON.
    """
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def record_audio() -> str:
    """
    Push-to-talk recording and saving to WAV at 16 kHz.
    Press ENTER to start, speak, ENTER to stop.
    """
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
    """
    Runs Whisper.cpp tiny model in CPU-only mode on the given WAV file.
    Returns the transcribed text.
    """
    txt_path = audio_path + ".txt"

    # Clean previous result
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

    if not os.path.exists(txt_path):
        print("Whisper output file missing:", txt_path)
        return ""

    with open(txt_path, "r") as f:
        return f.read().strip()


# ---------------- LLM COMMAND PARSER -------------------

def get_servo_plan_from_llm(text: str, pan: float, tilt: float, model_name: str) -> Optional[Dict]:
    """
    LLM converts natural language into ABSOLUTE target angles.

    Expected output:
    {
        "pan": <absolute angle 0..180>,
        "tilt": <absolute angle 0..180>
    }
    """

    system_prompt = """
You control a 2-axis servo turret with pan and tilt angles.

Your task:
- Read the user's natural language command.
- Consider the CURRENT pan and tilt angles.
- Decide the NEXT ABSOLUTE positions for pan and tilt.

You MUST output ONLY a JSON object:

{
  "pan": <number>,
  "tilt": <number>
}

No other fields. No explanations. No quotes. No backticks.

Constraints:
- 0 <= pan <= 180
- 0 <= tilt <= 180
- Do NOT output delta values or "mode". Only absolute final angles.

Behavior guidelines (not rigid rules, but IMPORTANT):
- "a little", "slightly" -> small change from current angle.
- "a lot", "far", "strong" -> large change from current angle.
- If the user mentions only left/right -> change pan and keep tilt the same.
- If the user mentions only up/down -> change tilt and keep pan the same.
- If the user says "center" -> pan = 90, tilt = 90.
- Do not move in a direction that would exceed 0..180.
- Use the current angles as context for how much to move.

Examples:

Current: pan=90, tilt=90
User: "go a little bit to the right"
Reasonable output (example):
{"pan": 100, "tilt": 90}

Current: pan=45, tilt=100
User: "a lot up"
Reasonable output:
{"pan": 45, "tilt": 140}

Current: pan=120, tilt=70
User: "go a little left and down"
Reasonable output:
{"pan": 110, "tilt": 60}

Remember:
- Use the given current angles.
- Adjust only the axes mentioned by the user.
- Always return ONLY the final JSON object with "pan" and "tilt" with no extra explanation.
"""

    # Give the model a very clear, simple description of the situation
    user_content = (
        f"Current pan angle: {pan}\n"
        f"Current tilt angle: {tilt}\n\n"
        f"User command: {text}\n\n"
        "Return ONLY the JSON with the new absolute pan and tilt values."
    )

    start_llm = time.time()

    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    print(f"{model_name} time is: ", time.time() - start_llm)
    raw = extract_llm_content(response)
    print(f"{model_name} raw output:", raw)

    try:
        plan = extract_json(raw)
        # Minimal sanity check
        if "pan" not in plan or "tilt" not in plan:
            print("LLM JSON missing keys:", plan)
            return None
        return plan
    except Exception as e:
        print("LLM output parse error:", e)
        return None

def extract_llm_content(response):
    """
    Handles all Ollama response formats:
    - response.message.content
    - response["message"]["content"]
    - response["messages"][0]["content"]
    - response["content"]
    """
    if isinstance(response, dict):
        if "message" in response and isinstance(response["message"], dict):
            if "content" in response["message"]:
                return response["message"]["content"]

        if "messages" in response and isinstance(response["messages"], list):
            if len(response["messages"]) > 0:
                return response["messages"][0].get("content", "")

        if "content" in response:
            return response["content"]

    # Try legacy object API
    try:
        return response.message.content
    except:
        return ""


# ---------------- SERVO SIMULATION -------------------

def simulate_motion(pan, tilt, pan_target, tilt_target):
    steps = max(1, SIMULATION_SECONDS)
    d_pan = (pan_target - pan) / steps
    d_tilt = (tilt_target - tilt) / steps

    print(f"Start pan={pan}, tilt={tilt}")
    print("Simulating movement...")

    for _ in range(steps):
        pan += d_pan
        tilt += d_tilt
        print(f"Pan={pan:.1f}, Tilt={tilt:.1f}")
        time.sleep(1)

    return pan, tilt


# ---------------- MAIN -------------------

def main():
    pan = START_PAN
    tilt = START_TILT

    print("=== Voice-controlled LLM Servo System (ABSOLUTE MODE) ===")

    while True:
        wav = record_audio()
        text = run_whisper(wav)

        print("\nYou said:", text)

        if not text:
            print("No speech recognized.")
            continue

        for model in MODEL_LIST:
            plan = get_servo_plan_from_llm(text, pan, tilt, model)
            if not plan:
                print("Invalid or unusable LLM command.")
                continue

        pan_target = clamp(float(plan["pan"]), MIN_ANGLE, MAX_ANGLE)
        tilt_target = clamp(float(plan["tilt"]), MIN_ANGLE, MAX_ANGLE)

        pan, tilt = simulate_motion(pan, tilt, pan_target, tilt_target)

        print("\nNew position:", pan, tilt)
        print("\nSay another command...\n")


if __name__ == "__main__":
    main()
