
import sys
import json
import re
import chromadb
from chromadb.utils import embedding_functions

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ──────────────────────────────────────────────────────────────────
INVENTORY_FILE = "inventory.json"
CHROMA_PATH    = "./chroma_db"
COLLECTION     = "genesis_inventory"
EMBED_MODEL    = "all-MiniLM-L6-v2"  
FEATURE_JUNK = {
    "The BrandThe Brand OverviewBrand NewsOne of One",
    "MagmaProgram OverviewGenesis Magma RacingFIA World Endurance Championship",
    "Facebook",
    "Find a Service Network",
    "Digital Services",
    "Browse Inventory",
    "Book a Test Drive",
    "Find a Dealer",
    "Contact Us",
    "Download a Catalog",
    "Register Your Interest",
}

# Helpers
def clean_features(features: list[str]) -> list[str]:
    """Remove nav-bar junk and duplicates, keep real feature strings."""
    seen = set()
    cleaned = []
    for f in features:
        f = f.strip()
        if f in FEATURE_JUNK:
            continue
        if len(f) < 5 or len(f) > 100:
            continue
        if f in seen:
            continue
        seen.add(f)
        cleaned.append(f)
    return cleaned


def format_price(price_str: str) -> str:
    if price_str == "Unknown":
        return "price not listed"
    try:
        return f"SAR {int(price_str):,}"
    except ValueError:
        return f"SAR {price_str}"


def build_document(car: dict) -> str:

    features = clean_features(car.get("features", []))
    features_str = ", ".join(features) if features else "standard features"

    mileage = car.get("mileage_km", "Unknown")
    mileage_str = f"{int(mileage):,} km" if mileage != "Unknown" else "mileage not listed"

    certified = car.get("genesis_certified", "No")
    certified_str = "Genesis Certified Pre-Owned" if certified == "Yes" else "not Genesis Certified"

    availability = car.get("availability", "Unknown")

    doc = (
        f"{car['year']} Genesis {car['model']} {car['trim']} ({car['drivetrain']}) — "
        f"{format_price(car['price_sar'])}. "
        f"Body: {car.get('body_type', 'Unknown')}. "
        f"Fuel: {car.get('fuel_type', 'Unknown')}. "
        f"Exterior: {car.get('exterior_color', 'Unknown')}. "
        f"Interior: {car.get('interior_color', 'Unknown')}. "
        f"Mileage: {mileage_str}. "
        f"Transmission: {car.get('transmission', 'Automatic')}. "
        f"Availability: {availability}. "
        f"Certification: {certified_str}. "
        f"Key features: {features_str}. "
        f"Listing URL: {car.get('url', '')}."
    )
    return doc


def build_metadata(car: dict) -> dict:
    features = clean_features(car.get("features", []))
    return {
        "year":           car.get("year", "Unknown"),
        "model":          car.get("model", "Unknown"),
        "trim":           car.get("trim", "Unknown"),
        "drivetrain":     car.get("drivetrain", "Unknown"),
        "exterior_color": car.get("exterior_color", "Unknown"),
        "interior_color": car.get("interior_color", "Unknown"),
        "fuel_type":      car.get("fuel_type", "Unknown"),
        "engine":         car.get("engine", "Unknown"),
        "transmission":   car.get("transmission", "Unknown"),
        "body_type":      car.get("body_type", "Unknown"),
        "mileage_km":     car.get("mileage_km", "Unknown"),
        "availability":   car.get("availability", "Unknown"),
        "price_sar":      car.get("price_sar", "Unknown"),
        "genesis_certified": car.get("genesis_certified", "No"),
        "url":            car.get("url", ""),
        "features":       " | ".join(features),   # pipe-delimited for storage
    }


# Main
def ingest():
    print("[*] Loading inventory...")
    with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
        inventory = json.load(f)
    print(f"    {len(inventory)} listings loaded.")

    print(f"\n[*] Connecting to ChromaDB at {CHROMA_PATH!r}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    try:
        client.delete_collection(COLLECTION)
        print(f"    Deleted existing collection '{COLLECTION}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"    Collection '{COLLECTION}' created.")

    print("\n[*] Building documents & upserting...")
    documents, metadatas, ids = [], [], []

    for i, car in enumerate(inventory):
        doc = build_document(car)
        meta = build_metadata(car)
        uid = f"car_{i:04d}"

        documents.append(doc)
        metadatas.append(meta)
        ids.append(uid)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"\n[DONE] {len(ids)} documents indexed into ChromaDB.")
    print(f"       Collection '{COLLECTION}' has {collection.count()} entries.\n")

    print("[TEST] Sanity check — querying 'blue G80 under 250000':")
    results = collection.query(
        query_texts=["blue G80 under 250000"],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"   [{dist:.3f}] {meta['year']} {meta['model']} {meta['trim']} "
              f"-- {meta['exterior_color']} -- SAR {meta['price_sar']}")


if __name__ == "__main__":
    ingest()
