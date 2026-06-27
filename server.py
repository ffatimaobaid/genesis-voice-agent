"""
server.py -- FastAPI backend for the Genesis CPO Voice Agent
=============================================================
Serves the web frontend and handles all AI processing server-side.

Run:
    python server.py
    # or
    uvicorn server:app --reload --port 8000
"""

import os
import sys
import json
import tempfile
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Lazy-load heavy models
_groq_client   = None

def get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

from rag import retrieve, format_context_for_llm, _is_vehicle_query, _expand_query

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Genesis CPO Agent API")

@app.on_event("startup")
def startup_event():
    print("=" * 60, flush=True)
    print("[STARTUP] Automatically pre-loading heavy machine learning models...", flush=True)
    print("=" * 60, flush=True)
    # Warm up Groq Client
    try:
        get_groq()
        print("[STARTUP] Groq client initialized.", flush=True)
    except Exception as e:
        print(f"[STARTUP] Error initializing Groq client: {e}", flush=True)
    
    # Warm up ChromaDB and SentenceTransformer Embedding model
    try:
        from rag import _get_collection, retrieve
        print("[STARTUP] Initializing SentenceTransformer model and loading ChromaDB collection...", flush=True)
        col = _get_collection()
        print(f"[STARTUP] ChromaDB collection '{col.name}' loaded successfully. Count: {col.count()}", flush=True)
        # Perform a quick dummy query to trigger model load
        _ = retrieve("warmup", n_results=1)
        print("[STARTUP] Vector search engine warmed up successfully.", flush=True)
    except Exception as e:
        print(f"[STARTUP] Error warming up RAG/database: {e}", flush=True)
    print("=" * 60, flush=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared config ─────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
RAG_TOP_K  = 12

# Minimum cosine similarity for a car to be surfaced as a UI recommendation card
RELEVANCE_THRESHOLD = 0.20


def _filter_cars_for_display(cars: list[dict], query: str) -> list[dict]:
    """
    Return only the cars that should be shown as UI cards.
    """
    return cars

SYSTEM_PROMPT = """You are Lara, an enthusiastic and knowledgeable sales agent for Genesis Certified Pre-Owned (CPO) vehicles at a premium dealership in Saudi Arabia.

Your personality:
- Warm, confident, and genuinely excited about Genesis cars.
- Make it sound like a natural, flowing back-and-forth conversation (acknowledge the customer's previous point, validate their interest, use natural transitions).
- You proactively upsell and highlight the most impressive, standout features.
- You suggest alternatives, compare trims, and mention what upgrading gets the customer.
- You speak naturally -- no bullet points, no markdown, just flowing conversational sentences.
- Keep responses SHORT: 1-2 sentences for simple questions, 3 sentences maximum for complex ones.
- Never list multiple features at once; pick the single most exciting detail and weave it naturally into conversation.
- Prices are in Saudi Riyals (SAR).
- Occasionally use light enthusiasm: "absolutely", "you'll love this", "honestly this one is special".

Your CPO Inventory Context:
- There are exactly three luxury models of cars available in our Certified Pre-Owned inventory: the G80 executive sedan, the GV80 flagship SUV, and the G90 flagship sedan. There are several variants of each of these three models available.
- ONLY reference specific vehicles and specs that appear in the RETRIEVED INVENTORY section of the current turn, or vehicles that were already explicitly discussed in the immediate conversation history. Never invent specs, prices, or availability.
- If the customer asks for a specific trim, year, or variant that is not in our inventory (such as a G80 2.5T Premium), explicitly state that we do not have that exact trim/year, but immediately offer the closest matching variant that IS available in the RETRIEVED INVENTORY section (e.g. the G80 2.5 Royal or 2.5T Platinum).
- When describing a car, naturally weave in 2-3 of its most impressive features.
- Always end with a warm, open-ended follow-up question to keep the conversation going.
- If the customer asks to "show me it", says they "want to proceed with this specific car", asks for the link, or wants to view/buy a specific vehicle, you MUST explicitly output the full URL of that specific vehicle from the context (e.g., "Please visit our website at [URL] to proceed with this vehicle."). Always output the full URL exactly as it appears in the RETRIEVED INVENTORY context (do not invent, shorten, or change the URL).

General context:
- All cars are Genesis Certified Pre-Owned (CPO) with rigorous multi-point inspection and warranty.
- Genesis is South Korea's ultra-premium luxury marque, competing with Mercedes, BMW, and Lexus.
- You are speaking with a customer in a voice conversation -- keep it warm, dialogue-based, and highly conversational.
"""

# ── Models ────────────────────────────────────────────────────────────────────
class TextChatRequest(BaseModel):
    text: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    response: str
    you_said: str
    retrieved_cars: list[dict]
    context_used: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "groq_key": bool(os.getenv("GROQ_API_KEY"))}


@app.get("/api/inventory")
async def get_inventory():
    """Return the full inventory list."""
    try:
        with open("inventory.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"count": len(data), "inventory": data}
    except FileNotFoundError:
        raise HTTPException(404, "inventory.json not found")


@app.post("/api/chat/text", response_model=ChatResponse)
async def chat_text(req: TextChatRequest):
    """Accept plain text input, return agent response + retrieved cars."""
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(500, "GROQ_API_KEY not configured")

    user_text = req.text.strip()
    if not user_text:
        raise HTTPException(400, "text is empty")

    # RAG - expand referential queries with context from conversation history
    search_query = _expand_query(user_text, req.history)
    cars    = retrieve(search_query, n_results=RAG_TOP_K)
    context = format_context_for_llm(cars)

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\n{context}"}]
    messages.extend(req.history[-10:])
    messages.append({"role": "user", "content": user_text})

    # LLM
    groq = get_groq()
    try:
        completion = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
    except Exception as e:
        print(f"[LLM] Groq call failed on {GROQ_MODEL}: {e}. Falling back to llama-3.1-8b-instant...", flush=True)
        completion = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
    response_text = completion.choices[0].message.content.strip()

    # Clean car metadata for JSON response — only surface cards relevant to this query
    display_cars = _filter_cars_for_display(cars, user_text)
    clean_cars = []
    for c in display_cars:
        car = {k: v for k, v in c.items() if not k.startswith("_")}
        car["features"] = [f.strip() for f in car.get("features", "").split("|") if f.strip()]
        clean_cars.append(car)

    return ChatResponse(
        response=response_text,
        you_said=user_text,
        retrieved_cars=clean_cars,
        context_used=context,
    )


@app.post("/api/chat/audio", response_model=ChatResponse)
async def chat_audio(
    audio: UploadFile = File(...),
    history: str = Form("[]"),
):
    """
    Accept an audio file (webm/ogg/wav from browser MediaRecorder),
    transcribe with Groq Whisper API (fast cloud inference), then run the
    same RAG+LLM pipeline.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(500, "GROQ_API_KEY not configured")

    # Read the uploaded audio bytes
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.webm"
    suffix = "." + filename.split(".")[-1]

    # Groq Whisper API requires an actual file-like object with a name
    # Use a NamedTemporaryFile so the MIME type is inferred correctly
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        groq = get_groq()
        print(f"[STT] Sending {len(audio_bytes)} bytes to Groq Whisper API...", flush=True)
        with open(tmp_path, "rb") as f:
            transcription = groq.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=(filename, f, "audio/webm"),
                response_format="text",
                language="en",
            )
        user_text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        print(f"[STT] Transcribed: {user_text!r}", flush=True)

        if not user_text or len(user_text) < 2:
            raise HTTPException(422, "Could not transcribe audio — please speak clearly and try again.")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[STT] Groq Whisper error: {e}", flush=True)
        raise HTTPException(422, f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Parse conversation history from JSON string
    try:
        history_list = json.loads(history)
    except Exception:
        history_list = []

    # Reuse text chat logic
    req = TextChatRequest(text=user_text, history=history_list)
    return await chat_text(req)


@app.get("/api/inventory/summary")
async def inventory_summary():
    """Quick summary stats."""
    try:
        with open("inventory.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        by_model = {}
        price_range = {}
        for car in data:
            m = car.get("model", "Unknown")
            by_model[m] = by_model.get(m, 0) + 1
            try:
                p = int(car.get("price_sar", 0))
                if m not in price_range:
                    price_range[m] = {"min": p, "max": p}
                else:
                    price_range[m]["min"] = min(price_range[m]["min"], p)
                    price_range[m]["max"] = max(price_range[m]["max"], p)
            except Exception:
                pass
        return {"total": len(data), "by_model": by_model, "price_range": price_range}
    except FileNotFoundError:
        raise HTTPException(404, "inventory.json not found")


# ── TTS Endpoint (Dynamic: Edge-TTS & ElevenLabs with Fallback) ───────────────

class TTSRequest(BaseModel):
    text: str
    provider: str = "edge"
    voice: str = "en-US-JennyNeural"

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """
    Convert text to natural-sounding speech.
    Supports ElevenLabs (Premium) and Microsoft Edge TTS (Free, natural neural voices).
    If ElevenLabs is selected but fails or exceeds quota, it falls back automatically to Edge-TTS Jenny.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")

    provider = req.provider.strip().lower()
    voice = req.voice.strip()

    # Default voice settings
    default_edge_voice = "en-US-JennyNeural"
    default_eleven_voice = "21m00Tcm4TlvDq8ikWAM" # Rachel

    if provider == "eleven":
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print("[TTS] ElevenLabs selected but API key missing. Falling back to Edge-TTS.", flush=True)
            provider = "edge"
        else:
            import httpx
            # If the user specified an Edge-TTS voice string, use Rachel for ElevenLabs
            voice_id = voice if ("Neural" not in voice) else default_eleven_voice
            if not voice_id:
                voice_id = default_eleven_voice
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            }
            body = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                }
            }
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=body, headers=headers, timeout=20.0)
                    if resp.status_code == 200:
                        audio_data = resp.content
                        return StreamingResponse(
                            io.BytesIO(audio_data),
                            media_type="audio/mpeg",
                            headers={"Content-Length": str(len(audio_data))},
                        )
                    else:
                        print(f"[TTS] ElevenLabs failed with status {resp.status_code}: {resp.text}. Falling back to Edge-TTS.", flush=True)
                        provider = "edge"
            except Exception as e:
                print(f"[TTS] ElevenLabs exception: {e}. Falling back to Edge-TTS.", flush=True)
                provider = "edge"

    if provider == "edge":
        import edge_tts
        # If the user specified an ElevenLabs voice ID, use Jenny for Edge-TTS
        selected_voice = voice if ("Neural" in voice) else default_edge_voice
        if not selected_voice:
            selected_voice = default_edge_voice

        try:
            # Use rate="+0%" for normal speed (less robotic, more natural breathing)
            communicate = edge_tts.Communicate(text, selected_voice, rate="+0%")
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])

            if not audio_data:
                raise HTTPException(500, "Edge-TTS produced no audio")

            return StreamingResponse(
                io.BytesIO(bytes(audio_data)),
                media_type="audio/mpeg",
                headers={"Content-Length": str(len(audio_data))},
            )
        except Exception as e:
            print(f"[TTS] Edge-TTS error: {e}. Falling back to gTTS.", flush=True)
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang="en")
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                audio_bytes = fp.getvalue()
                return StreamingResponse(
                    io.BytesIO(audio_bytes),
                    media_type="audio/mpeg",
                    headers={"Content-Length": str(len(audio_bytes))},
                )
            except Exception as gtts_err:
                print(f"[TTS] gTTS fallback failed: {gtts_err}", flush=True)
                raise HTTPException(500, f"TTS failed: {e}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Genesis CPO Agent API on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
