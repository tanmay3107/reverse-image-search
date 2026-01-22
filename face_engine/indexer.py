import os
import json
import requests
import numpy as np
import faiss
from deepface import DeepFace

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data/embeddings"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "face_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# DeepFace config
# -----------------------------
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
EMBEDDING_DIM = 512


# -----------------------------
# Utilities
# -----------------------------
def download_image(url: str):
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    if url.startswith("//"):
        url = "https:" + url

    if not url.startswith("http"):
        raise ValueError("Invalid URL")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FaceSearchBot/1.0)",
        "Referer": "https://commons.wikimedia.org/"
    }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.content


import cv2
import numpy as np

def extract_embedding(img_bytes):
    # bytes → numpy image
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return None

    reps = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    if not reps:
        return None

    return np.array(reps[0]["embedding"], dtype="float32")



# -----------------------------
# ONE-TIME REBUILD FUNCTION
# -----------------------------
def rebuild_index_from_urls(image_urls: list[str]):
    """
    ONE-TIME rebuild.
    Call this manually, NOT on app startup.
    """

    print(f"[INDEXER] Rebuilding index from {len(image_urls)} images")

    embeddings = []
    metadata = []

    for i, url in enumerate(image_urls, start=1):
        print(f"[INDEXER] ({i}/{len(image_urls)}) Processing")

        try:
            img = download_image(url)
            emb = extract_embedding(img)

            if emb is None:
                continue

            embeddings.append(emb)
            metadata.append({"url": url})

            print("[INDEXER] Face embedding stored")

        except Exception as e:
            print(f"[INDEXER] Failed — {e}")

    if not embeddings:
        print("[INDEXER] ❌ No embeddings created — aborting save")
        return

    embeddings = np.vstack(embeddings).astype("float32")

    # ✅ IMPORTANT: Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # ✅ Cosine similarity index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # Save everything
    np.save(EMBEDDINGS_FILE, embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print(f"[INDEXER] ✅ Rebuild complete: {len(embeddings)} embeddings saved")


# -----------------------------
# LOAD EXISTING INDEX (SEARCH)
# -----------------------------
def load_index():
    if not os.path.exists(FAISS_INDEX_FILE):
        return None, None

    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    print(f"[INDEXER] Loaded index with {index.ntotal} embeddings")
    return index, metadata

# =========================
# URL NORMALIZATION
# =========================

def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""

    url = url.strip()

    if url.startswith("//"):
        url = "https:" + url

    return url
