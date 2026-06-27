"""
rag.py — Retrieval module for Genesis CPO inventory

Usage:
    from rag import retrieve

    cars = retrieve("cheapest GV80 in white")
    for car in cars:
        print(car["year"], car["model"], car["price_sar"])
"""

import sys
import chromadb
from chromadb.utils import embedding_functions

# Ensure UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHROMA_PATH = "./chroma_db"
COLLECTION  = "genesis_inventory"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Keywords that indicate the user is actively searching/browsing/asking about specific vehicles
VEHICLE_INTENT_KEYWORDS = {
    # Model names & variants
    "g80", "gv80", "g90", "ev", "electric", "hybrid", "petrol", "gasoline", "diesel",
    # Shopping intent
    "show", "recommend", "suggest", "find", "looking for", "want", "need", "search",
    "budget", "price", "cost", "how much", "cheap", "affordable", "cheapest", "expensive",
    "available", "in stock", "buy", "purchase", "certified", "warranty", "inspect", "inspection",
    # Vehicle attributes & features
    "color", "trim", "mileage", "km", "kilometers", "awd", "rwd", "4wd", "fwd",
    "sedan", "suv", "mpv", "v6", "v8", "turbo", "sport", "prestige", "luxury",
    "sunroof", "moonroof", "leather", "seats", "navigation", "audio", "sound", "camera", "sensor",
    # Comparison & details
    "compare", "difference", "better", "vs", "versus", "tell me about", "more details", "spec", "specs", "specification", "specifications",
    # Explicit vehicle request
    "car", "vehicle", "model", "which one", "them", "those",
}

def _is_vehicle_query(query: str) -> bool:
    """Return True when the user query is clearly about finding or comparing vehicles."""
    q = query.lower()
    return any(kw in q for kw in VEHICLE_INTENT_KEYWORDS)

def _expand_query(query: str, history: list[dict]) -> str:
    """If the query is a follow-up referential query, append key nouns/models from history."""
    q = query.lower()
    referential_keywords = {"it", "this", "that", "the car", "the vehicle", "the suv", "the sedan", "more details", "proceed", "go ahead", "link", "show me"}
    if any(kw in q for kw in referential_keywords) and history:
        # Find the last assistant/agent message and extract any model names & colors
        for msg in reversed(history):
            role = msg.get("role", "")
            if role in ("assistant", "agent"):
                content = msg.get("content", "")
                models = []
                for m in ["G80", "GV80", "G90"]:
                    if m.lower() in content.lower():
                        models.append(m)
                colors = []
                for color in ["blue", "white", "black", "red", "green", "grey", "silver", "gold"]:
                    if color in content.lower():
                        colors.append(color)
                
                expanded = query
                if models:
                    expanded += " " + " ".join(models)
                if colors:
                    expanded += " " + " ".join(colors)
                return expanded
    return query

# Module-level singletons — initialised once, reused across calls
_client     = None
_collection = None
_ef         = None


def _get_collection():
    global _client, _collection, _ef
    if _collection is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_collection(
            name=COLLECTION,
            embedding_function=_ef,
        )
    return _collection


def retrieve(query: str, n_results: int = 5) -> list[dict]:
    """
    Semantic search over the Genesis CPO inventory.

    Returns a list of car dicts (metadata) ranked by relevance.
    Each dict has keys: year, model, trim, drivetrain, exterior_color,
    interior_color, fuel_type, engine, transmission, body_type,
    mileage_km, availability, price_sar, genesis_certified, url, features.
    """
    col = _get_collection()

    results = col.query(
        query_texts=[query],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    cars = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        car = dict(meta)
        car["_document"] = doc          # full text chunk
        car["_score"]    = round(1 - dist, 4)   # cosine similarity [0..1]
        cars.append(car)

    return cars


def format_context_for_llm(cars: list[dict]) -> str:
    """
    Format retrieved car dicts into a structured context block
    to insert into the LLM prompt.
    """
    if not cars:
        return "No matching vehicles found in the current inventory."

    lines = ["=== RETRIEVED INVENTORY (most relevant first) ===\n"]
    for i, car in enumerate(cars, 1):
        features_list = [f.strip() for f in car.get("features", "").split("|") if f.strip()]
        features_str  = ", ".join(features_list) if features_list else "standard features"

        price = car.get("price_sar", "Unknown")
        try:
            price_fmt = f"SAR {int(price):,}"
        except (ValueError, TypeError):
            price_fmt = "price not listed"

        mileage = car.get("mileage_km", "Unknown")
        try:
            mileage_fmt = f"{int(mileage):,} km"
        except (ValueError, TypeError):
            mileage_fmt = "mileage not listed"

        certified = "✓ Genesis Certified Pre-Owned" if car.get("genesis_certified") == "Yes" else "Not certified"
        avail     = car.get("availability", "Unknown")

        lines.append(
            f"[Vehicle {i}] {car.get('year')} Genesis {car.get('model')} {car.get('trim')} "
            f"({car.get('drivetrain')})\n"
            f"  Price       : {price_fmt}\n"
            f"  Body        : {car.get('body_type')}\n"
            f"  Fuel        : {car.get('fuel_type')}\n"
            f"  Exterior    : {car.get('exterior_color')}\n"
            f"  Interior    : {car.get('interior_color')}\n"
            f"  Mileage     : {mileage_fmt}\n"
            f"  Availability: {avail}\n"
            f"  Certification: {certified}\n"
            f"  Features    : {features_str}\n"
            f"  URL         : {car.get('url')}\n"
        )

    return "\n".join(lines)


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "affordable G80 sedan"
    print(f"\n[QUERY] {query!r}\n")
    cars = retrieve(query, n_results=3)
    print(format_context_for_llm(cars))
