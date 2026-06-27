
import os
import sys
import re
import argparse
import threading
import time
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from groq import Groq
from dotenv import load_dotenv

from rag import retrieve, format_context_for_llm, _is_vehicle_query, _expand_query

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

WHISPER_MODEL = "base"
GROQ_MODEL = "llama-3.3-70b-versatile"
SAMPLE_RATE = 16000
CHANNELS = 1
RAG_TOP_K = 12
SILENCE_THRESHOLD = 0.015   
SILENCE_DURATION = 1.8      
MAX_RECORD_SECS = 30

FAREWELL_TRIGGERS = {
    "goodbye", "bye", "exit", "quit", "that's all",
    "thats all", "no thanks", "thank you bye", "see you"
}

SYSTEM_PROMPT = """You are Lara, a warm and enthusiastic sales advisor for Genesis Certified Pre-Owned vehicles at a premium dealership in Saudi Arabia. You genuinely love these cars and it shows.

Your personality:
- Confident, charming, and naturally persuasive — like a trusted friend who happens to know everything about luxury cars
- You oversell in the best way: highlight the most impressive features, compare trims, mention what upgrading gets them
- You speak conversationally, like a phone call — no lists, no bullet points, just natural flowing sentences
- Light enthusiasm where it fits: "honestly this one is special", "you'll absolutely love it", "this is one of our best"
- Prices are in Saudi Riyals (SAR)

Your strict rules:
- ONLY reference specific vehicles and specs that appear in the RETRIEVED INVENTORY section of the current turn, or vehicles that were already explicitly discussed in the immediate conversation history. Never invent specs, prices, or availability.
- If the customer asks for a specific trim, year, or variant that is not in our inventory (such as a G80 2.5T Premium), explicitly state that we do not have that exact trim/year, but immediately suggest the closest matching variant that IS available in the RETRIEVED INVENTORY section (e.g. the G80 2.5 Royal or 2.5T Platinum).
- Always weave in 2-3 standout features naturally when describing a car.
- Always end with a warm follow-up question to keep the conversation going.
- If the customer says "show me it", "I want to proceed with this specific car", asks for the link, or wants to view/buy a specific vehicle, you MUST explicitly output the full URL of that specific vehicle from the context (e.g., "Please visit our website at [URL] to proceed with this vehicle."). Always output the full URL exactly as it appears in the RETRIEVED INVENTORY context (do not invent, shorten, or change the URL).
- Keep responses concise: 2-4 sentences for simple questions, up to 6 for complex comparisons.

Important context:
- Every car is Genesis Certified Pre-Owned — multi-point inspection, warranty included, peace of mind guaranteed.
- Genesis is Korea's ultra-luxury marque, rivaling Mercedes-Benz, BMW, and Lexus at a better value.
- The customer called in — this is a phone conversation, make them feel like a VIP.
"""

# TTS: three-tier fallback
def _play_wav(path: str):
    """Play a WAV file through sounddevice."""
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate=samplerate)
    sd.wait()


def speak_gtts(text: str) -> bool:
    """Google TTS (free, requires internet)."""
    try:
        from gtts import gTTS
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_mp3 = f.name
        gTTS(text=text, lang="en", slow=False).save(tmp_mp3)
        # Convert MP3 -> WAV via soundfile (needs ffmpeg) or use pygame
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_mp3)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.quit()
        except ImportError:
            subprocess.run(["cmd", "/c", "start", "/wait", "", tmp_mp3],
                           capture_output=True)
        os.unlink(tmp_mp3)
        return True
    except Exception as e:
        print(f"[TTS] gTTS failed: {e}", flush=True)
        return False


def speak_pyttsx3(text: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        for voice in engine.getProperty("voices"):
            if "zira" in voice.name.lower() or "female" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"[TTS] pyttsx3 failed: {e}", flush=True)
        return False


def speak_elevenlabs(text: str) -> bool:
    # ElevenLabs TTS (requires paid plan for pre-made voices).
    el_key = os.getenv("ELEVENLABS_API_KEY")
    if not el_key:
        return False
    try:
        from elevenlabs import ElevenLabs, VoiceSettings
        el_client = ElevenLabs(api_key=el_key)
        voices_resp = el_client.voices.get_all()
        voice_id = None
        for v in (voices_resp.voices or []):
            if v.category == "cloned" or v.category == "generated":
                voice_id = v.voice_id
                break
        if not voice_id:
            return False  # no custom voices, skip
        audio_bytes = b"".join(
            el_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5",
                voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.8),
                output_format="pcm_16000",
            )
        )
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(arr, samplerate=16000)
        sd.wait()
        return True
    except Exception as e:
        print(f"[TTS] ElevenLabs failed: {e}", flush=True)
        return False


def speak(text: str) -> None:
    print(f"\n[LARA] {text}\n", flush=True)
    speech_text = re.sub(r'https?://[^\s]+', 'our website', text)
    # Try in order: ElevenLabs -> gTTS -> pyttsx3 -> silent
    if speak_elevenlabs(speech_text):
        return
    if speak_gtts(speech_text):
        return
    if speak_pyttsx3(speech_text):
        return
    print("[TTS] All TTS engines failed -- text only mode.", flush=True)


def list_devices():
    print("\n=== AUDIO DEVICES ===")
    for i, d in enumerate(sd.query_devices()):
        marker = ""
        if i == sd.default.device[0]: marker += "  <-- default input"
        if i == sd.default.device[1]: marker += "  <-- default output"
        print(f"  [{i:2d}] {d['name'][:58]:<58}  in={d['max_input_channels']}  out={d['max_output_channels']}{marker}")
    print()


def find_mic_device() -> int | None:
   
    devices = sd.query_devices()
    AVOID = {
        "stereo mix", "loopback", "what u hear", "wave out mix",
        "sound mapper", "primary sound capture", 
    }

    def is_bad(name: str) -> bool:
        nl = name.lower()
        return any(bad in nl for bad in AVOID)

    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and not is_bad(d["name"]):
            if "microphone" in d["name"].lower():
                return i

    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and not is_bad(d["name"]):
            return i

    return None  


def record_push_to_talk(device: int | None) -> np.ndarray | None:
    print("\n[PTT] Press ENTER to start speaking, ENTER again when done...")
    input()
    chunks, going = [], True

    def cb(indata, frames, time_info, status):
        if going:
            chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", device=device, callback=cb):
        print("[REC] Recording... (press ENTER to stop)")
        input()
        going = False

    if not chunks:
        return None
    audio = np.concatenate(chunks, axis=0).flatten()
    return audio if len(audio) > SAMPLE_RATE * 0.3 else None


def record_auto_stop(device: int | None) -> np.ndarray | None:
 
    print("[MIC] Listening... (speak now)", flush=True)

    chunks         = []
    speech_seen    = [False]
    silence_start  = [None]
    done_event     = threading.Event()
    rms_ref        = [0.0]

    def cb(indata, frames, time_info, status):
        chunk = indata.copy()
        chunks.append(chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_ref[0] = rms

        if rms >= SILENCE_THRESHOLD:
            speech_seen[0]   = True
            silence_start[0] = None
        elif speech_seen[0]:
            # Counting silence only after we heard speech
            if silence_start[0] is None:
                silence_start[0] = time.time()
            elif time.time() - silence_start[0] >= SILENCE_DURATION:
                done_event.set()

    def meter():
        while not done_event.is_set():
            rms = rms_ref[0]
            bar = "#" * min(int(rms * 800), 30)
            print(f"\r  vol: [{bar:<30}] {'SPEECH' if speech_seen[0] else '      '}",
                  end="", flush=True)
            time.sleep(0.05)
        print("\r" + " " * 55 + "\r", end="", flush=True)

    threading.Thread(target=meter, daemon=True).start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", device=device, callback=cb):
        done_event.wait(timeout=MAX_RECORD_SECS)

    if not speech_seen[0]:
        return None  # never heard anything

    audio = np.concatenate(chunks, axis=0).flatten()
    if len(audio) < SAMPLE_RATE * 0.3 or np.abs(audio).max() < 0.001:
        return None
    return audio


# STT
def transcribe(audio: np.ndarray, model) -> str:
    print("[STT] Transcribing...", flush=True)
    result = model.transcribe(audio, fp16=False, language="en")
    text   = result["text"].strip()
    if text:
        print(f"[YOU] {text}", flush=True)
    return text


# LLM
def generate_response(groq_client, history: list[dict], user_text: str, context: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\n{context}"}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_text})
    print("[LLM] Thinking...", flush=True)
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.75, max_tokens=350
        )
    except Exception as e:
        print(f"[LLM] Groq call failed on {GROQ_MODEL}: {e}. Falling back to llama-3.1-8b-instant...", flush=True)
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=messages, temperature=0.75, max_tokens=350
        )
    return resp.choices[0].message.content.strip()


def is_farewell(text: str) -> bool:
    lower = text.lower().strip(".,!?")
    return any(t in lower for t in FAREWELL_TRIGGERS)


# Main
def main():
    parser = argparse.ArgumentParser(description="Genesis CPO Voice Agent (CLI)")
    parser.add_argument("--device",       type=int,            help="Mic device index")
    parser.add_argument("--push-to-talk", action="store_true", help="PTT mode")
    parser.add_argument("--list-devices", action="store_true", help="Print devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices(); return

    print("=" * 60)
    print("  Genesis CPO Voice AI Agent (CLI)")
    print("=" * 60)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[ERR] GROQ_API_KEY not set in .env"); sys.exit(1)

    if args.device is not None:
        mic_device = args.device
    else:
        mic_device = find_mic_device()

    if mic_device is not None:
        print(f"[MIC] Device {mic_device}: {sd.query_devices(mic_device)['name']}")
    else:
        print("[MIC] Using system default")

    print("\n[LOAD] Loading Whisper model...")
    model = whisper.load_model(WHISPER_MODEL)
    print("[OK] Whisper ready.")
    groq_client = Groq(api_key=groq_key)
    print("[OK] All systems ready.\n")

    greeting = (
        "Hello, and welcome to Genesis Certified Pre-Owned! "
        "I'm Lara, your personal vehicle advisor. "
        "We have a stunning selection of G80 sedans, the flagship G90, "
        "and the incredible GV80 SUV -- every single one Genesis Certified. "
        "What are you looking for today?"
    )
    speak(greeting)

    history: list[dict] = []
    ptt = args.push_to_talk
    print(f"[TIP] Mode: {'Push-to-talk' if ptt else 'Auto-stop on silence'}")
    print("[TIP] Say 'goodbye' to end. Ctrl+C to force quit.\n")

    try:
        while True:
            audio = record_push_to_talk(mic_device) if ptt else record_auto_stop(mic_device)

            if audio is None:
                print("[!] No speech detected. Try --device <id> (run --list-devices to see options)\n")
                continue

            user_text = transcribe(audio, model)
            if not user_text or len(user_text.strip()) < 3:
                print("[!] Could not understand. Please try again.\n")
                continue

            if is_farewell(user_text):
                speak("It was a pleasure speaking with you! Don't hesitate to call back whenever you're ready. Have a wonderful day!")
                break

            print("[RAG] Searching inventory...", flush=True)
            search_query = _expand_query(user_text, history)
            cars    = retrieve(search_query, n_results=RAG_TOP_K)
            context = format_context_for_llm(cars)
            print(f"      {len(cars)} vehicle(s) retrieved.", flush=True)

            response = generate_response(groq_client, history, user_text, context)
            history.extend([{"role": "user", "content": user_text},
                             {"role": "assistant", "content": response}])
            speak(response)

    except KeyboardInterrupt:
        print("\n[EXIT] Session ended by user.")
    print("Session complete. Goodbye!")


if __name__ == "__main__":
    main()
