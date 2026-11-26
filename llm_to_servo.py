"""
Simple demo: control 2 simulated servos(pan,tilt) with a local LLM.

REQUIREMENT:
    pip install -r requirements.txt
    download vosk-model-small-en-us-0.15 into models/ (see README)
    ollama pull gemma3:1b
    ollama serve
"""
import time

import json
import re
import time
from typing import Dict, Optional

from ollama import chat

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ------ CONFIG -----

MODEL_NAME = "gemma3:1b"

MIN_ANGLE = 0.0
MAX_ANGLE = 180.0

START_PAN = 90.0
START_TILT = 90.0

SIMULATION_SECONDS = 5

# Speech-to-text config
STT_SAMPLE_RATE = 16000
STT_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # change if you put it somewhere else

# HELPERS

def clamp(value: float, low: float, high: float) -> float:
    return max(low, (min(value, high)))


def extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the LLM reponse")

    json_str = match.group(0)
    return json.loads(json_str)


def validate_servo_plan(data:Dict) -> Dict:
    # Ensure the JSON has the fields we expect and the correct types.
    fields_set = {"pan_delta_deg", "tilt_delta_deg"}
    if not fields_set.issubset(data):
        raise ValueError("Missing one of the JSON fields")

    pan_delta = float(data["pan_delta_deg"])
    tilt_delta = float(data["tilt_delta_deg"])

    max_delta = 45

    pan_delta = clamp(pan_delta, -max_delta, max_delta)
    tilt_delta = clamp(tilt_delta, -max_delta, max_delta)

    return{
        "pan_delta_deg": pan_delta,
        "tilt_delta_deg": tilt_delta
    }


def get_servo_plan_from_llm(user_instructions: str) -> Optional[Dict]:
    """Send the text from the user to the LLM and get a movement plan
    The system prompt forces the model to answer only with a JSON"""

    system_prompt = (
        "You control a 2 axis turret with 'pan' and 'tilt' angles.\n"
        "The user gives natural commands about how to move it.\n\n"
        "You must respond with only a JSON object and nothing else.\n"
        "Format exactly:\n"
        "{\n"
        '   "pan_delta_deg": <number>,\n'
        '   "tilt_delta_deg": <number>\n'
        "}\n\n"
        "Meaning:\n"
        " - pan_delta_deg: positive = move right, negative = move left.\n"
        " - tilt_delta_deg: positive = move up, negative = move down.\n"
        " - Each Value is in degree, keep value between 45 and -45."
    )

    print("Sending Prompt to LLM")

    response = chat(
        model=MODEL_NAME,
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instructions},
    ],
    )

    print("Response Received from LLM")

    raw_text = response.message.content

    print("LLM raw output", raw_text)

    try:
        data = extract_json(raw_text)
        plan = validate_servo_plan(data)
        return plan
    except Exception as exc:
        print(f"[ERROR] Could not parse LLM output as valid plan: {exc}")
        return None


def simulate_servo_motion(
        current_pan: float,
        current_tilt: float,
        target_pan: float,
        target_tilt: float,
        seconds: int = SIMULATION_SECONDS,
) -> (float, float):

    steps = max(1, seconds)
    step_pan = (target_pan - current_pan) / steps
    step_tilt = (target_tilt - current_tilt) / steps

    print(f"Pan angle: {current_pan}")
    print(f"Tilt angle: {current_tilt}")
    time.sleep(1)

    print("Simulating movement")
    for i in range(seconds):
        current_pan += step_pan
        current_tilt += step_tilt
        print(f"Pan angle: {current_pan}")
        print(f"Tilt angle: {current_tilt}")

        time.sleep(1)

    return current_pan, current_tilt


def load_stt_model() -> Model:
    """Load the offline Vosk speech model once."""
    print("Loading STT model...")
    model = Model(STT_MODEL_PATH)
    print("STT model loaded.")
    return model


def recognize_speech_ptt(model: Model) -> str:
    """
    Press ENTER to start recording.
    Speak freely.
    Press ENTER again to stop and process speech.
    """
    input("\nPress ENTER to START speaking...")
    print("Recording... Press ENTER to STOP.")

    recorded_chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        recorded_chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=STT_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        input()  # Wait for ENTER to stop

    print("Recording stopped. Processing...")

    audio_data = b"".join(chunk.tobytes() for chunk in recorded_chunks)

    recognizer = KaldiRecognizer(model, STT_SAMPLE_RATE)
    recognizer.AcceptWaveform(audio_data)
    result_json = recognizer.Result()
    result_data = json.loads(result_json)

    text = result_data.get("text", "").strip()

    if text:
        print(f"Recognized: '{text}'")
    else:
        print("Did not recognize any speech.")

    return text


def main() -> None:
    pan = START_PAN
    tilt = START_TILT

    stt_model = load_stt_model()

    user_input= recognize_speech_ptt(stt_model)

    if not user_input:
        print("No user input processed")
        return

    llm_start = time.time()
    plan = get_servo_plan_from_llm(user_input)
    print("LLM time: ",time.time() - llm_start)

    if plan is None:
        print("No valid plan from the LLM, not moving")
        return

    pan_target = clamp(pan + plan["pan_delta_deg"], MIN_ANGLE, MAX_ANGLE)
    tilt_target = clamp(tilt + plan["tilt_delta_deg"], MIN_ANGLE, MAX_ANGLE)

    simulate_servo_motion(
        pan,
        tilt,
        pan_target,
        tilt_target
    )

if __name__ == "__main__":
    main()



