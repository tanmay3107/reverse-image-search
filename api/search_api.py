import os
import json
import base64
import requests
import numpy as np
import cv2
import faiss
from flask import Blueprint, request, jsonify
from deepface import DeepFace

# =========================
# Blueprint
# =========================
search_api = Blueprint("search_api", __name__)

# =========================
# Paths
# =========================
BASE_DIR = "data/embeddings"
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

# =========================
# Config
# =========================
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
EMBEDDING_DIM = 512

TOP_K = 50
MIN_SIMILARITY = 60.0  # percent

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# =========================
# Load FAISS index ONCE
# =========================
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"[SEARCH] ✅ Loaded {index.ntotal} embeddings")
else:
    index = None
    metadata = []
    print("[SEARCH] ❌ FAISS index not found")

# =========================
# Helpers
# =========================

def bytes_to_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def get_face_embedding_from_bytes(image_bytes):
    img = bytes_to_image(image_bytes)
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

    emb = np.array(reps[0]["embedding"], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.content


# =========================
# TRUE REVERSE IMAGE SEARCH
# =========================

def reverse_image_search_serpapi(image_bytes, max_results=10):
    """
    Uses Google Reverse Image Search via SerpAPI
    """
    if not SERPAPI_API_KEY:
        return []

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "engine": "google_reverse_image",
        "image_base64": image_b64,
        "api_key": SERPAPI_API_KEY
    }

    try:
        r = requests.post(
            "https://serpapi.com/search.json",
            json=payload,
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[SERPAPI] Error: {e}")
        return []

    urls = []

    for item in data.get("image_results", []):
        if item.get("original"):
            urls.append(item["original"])

    for item in data.get("inline_images", []):
        if item.get("original"):
            urls.append(item["original"])

    # Deduplicate
    return list(dict.fromkeys(urls))[:max_results]


# =========================
# Search Endpoint
# =========================

@search_api.route("/api/search", methods=["POST"])
def search_face():

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["file"].read()

    # --------------------------------------------------
    # STEP 1: TRUE WEB-SCALE REVERSE IMAGE SEARCH
    # --------------------------------------------------
    reverse_urls = reverse_image_search_serpapi(image_bytes)

    verified_results = []

    if reverse_urls:
        query_emb = get_face_embedding_from_bytes(image_bytes)

        if query_emb is not None:
            for url in reverse_urls:
                try:
                    candidate_bytes = download_image(url)
                    candidate_emb = get_face_embedding_from_bytes(candidate_bytes)

                    if candidate_emb is None:
                        continue

                    score = float(np.dot(query_emb, candidate_emb.T))
                    similarity = round(((score + 1) / 2) * 100, 2)

                    if similarity >= MIN_SIMILARITY:
                        verified_results.append({
                            "image_url": url,
                            "similarity": similarity
                        })

                except Exception:
                    continue

    if verified_results:
        verified_results.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        return jsonify({
            "count": len(verified_results),
            "matches": verified_results[:5],
            "mode": "true_reverse_image_search"
        })

    # --------------------------------------------------
    # STEP 2: FALLBACK — LOCAL FACE SIMILARITY SEARCH
    # --------------------------------------------------
    if index is None or index.ntotal == 0:
        return jsonify({
            "count": 0,
            "matches": [],
            "error": "No reverse matches found"
        })

    query_emb = get_face_embedding_from_bytes(image_bytes)
    if query_emb is None:
        return jsonify({
            "count": 0,
            "matches": [],
            "error": "No face detected"
        })

    scores, indices = index.search(query_emb, TOP_K)

    best_per_url = {}

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        url = metadata[idx]["url"]
        similarity = round(((float(score) + 1) / 2) * 100, 2)

        if similarity < MIN_SIMILARITY:
            continue

        if url not in best_per_url or similarity > best_per_url[url]:
            best_per_url[url] = similarity

    results = [
        {"image_url": url, "similarity": sim}
        for url, sim in best_per_url.items()
    ]

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return jsonify({
        "count": len(results),
        "matches": results[:5],
        "mode": "local_face_similarity"
    })
