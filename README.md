# Genesis CPO Voice AI Agent

A RAG-augmented voice AI agent that acts as customer support for the [Genesis Certified Pre-Owned](https://genesis-cpo.netlify.app/) car listings website. Customers can call in (or speak to their mic) and ask about available cars — models, variants, pricing, specs, availability, and the agent answers accurately based on live inventory.

![Genesis CPO Voice Agent](screenshots/image.jpg)

---

## Architecture

```
Microphone --> [Whisper STT]  ──── transcribes speech locally (no API key)

--> [ChromaDB RAG] ──── semantic search over inventory (sentence-transformers) top-5 relevant car listings

--> [Groq LLM]     ──── generates natural, grounded response (llama-3.3-70b)

--> [TTS] ── ElevenLabs (if API key + paid plan) → gTTS → pyttsx3 fallback

--> Speaker
```

| Component | Technology | Reason |
|-----------|-----------|--------|
| Scraping | Playwright | Handles JS-rendered pages |
| Vector Store | ChromaDB (local, persistent) | No external service; fast |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Free, fast, great accuracy |
| STT | OpenAI Whisper `base` | Fully local, no API key |
| LLM | Groq `llama-3.3-70b-versatile` | Sub-second inference |
| TTS |ElevenLabs Turbo v2.5 / gTTS / pyttsx3 | Fallback chain — ElevenLabs for best quality, gTTS and pyttsx3 as offline fallbacks  |
| Audio I/O | sounddevice + soundfile | Cross-platform |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd genesis-agent
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# Install Playwright browsers (needed for scraper only)
playwright install chromium
```

### 3. Set API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
- **GROQ_API_KEY** — free at [console.groq.com](https://console.groq.com/)
- **ELEVENLABS_API_KEY** — free tier at [elevenlabs.io](https://elevenlabs.io/) (10k chars/month)

### 4. (Optional) Re-scrape the inventory

The repo includes a pre-scraped `inventory.json`. To refresh it:

```bash
python scraper.py
```

---

## Running

### Step 1 — Build the vector index (run once)

```bash
python ingest.py
```

This reads `inventory.json`, builds natural-language text chunks for each car, embeds them with `sentence-transformers`, and persists the ChromaDB collection to `./chroma_db/`. Takes ~30 seconds on first run (model download + embedding).

You'll see a sanity-check query at the end confirming retrieval works.

### Step 2 — Start the voice agent

You can run the agent in either **Web App Mode** (web interface) or **CLI Mode** (command-line terminal).

#### Option A: Web App Mode (FastAPI Backend + React Frontend)

1. **Start the backend server:**
   ```bash
   python server.py
   ```
   The backend will start running at `http://localhost:8000`.

2. **Start the frontend application:**
   ```bash
   cd frontend
   npm run dev
   ```
   Open your browser and navigate to `http://localhost:3000` (or the port shown in your terminal) to interact with Lara.

#### Option B: CLI Mode (Terminal Only)

```bash
python agent.py
```
The agent greets you, then enters a listening loop:
* **Auto-stop mode** (default): Starts listening immediately. Stops after ~1.5s of silence.
* Say your question naturally — e.g. *"Do you have any blue G80s under 250,000 riyals?"*
* To switch to push-to-talk mode, run with the `--push-to-talk` flag: `python agent.py --push-to-talk`.
* Say **"goodbye"** or **"bye"** to end the session.

### Step 3 — Test RAG retrieval standalone

```bash
python rag.py "cheapest GV80 with panoramic sunroof"
```

---

## Project Structure

```
genesis-agent/
├── scraper.py        # Playwright scraper → inventory.json
├── inventory.json    # 56 scraped Genesis CPO listings
├── ingest.py         # Embeds & indexes inventory into ChromaDB
├── rag.py            # Semantic retrieval module
├── agent.py          # Voice agent main loop
├── server.py         # FastAPI Backend server
├── frontend/         # Next.js Frontend application
├── chroma_db/        # Persistent vector store (created by ingest.py)
├── requirements.txt
├── .env.example      # API key template
└── README.md
```
