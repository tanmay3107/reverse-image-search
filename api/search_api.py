import os
import json
import base64
import requests
import numpy as np
import cv2
import faiss
from flask import Blueprint, request, jsonify
from deepface import DeepFace
from dotenv import load_dotenv
from face_engine.image_hash import compute_phash, hamming_distance

load_dotenv()

search_api = Blueprint("search_api", __name__)

# =============================
# CONFIG
# =============================

BASE_DIR = "data/embeddings"
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
EMBEDDING_DIM = 512

TOP_K = 50
MIN_SIMILARITY = 75.0

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# =============================
# LOAD INDEX
# =============================

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"[SEARCH] Loaded {index.ntotal} embeddings")
else:
    index = None
    metadata = []
    print("[SEARCH] ❌ FAISS index not found")


# =============================
# HELPERS
# =============================

def bytes_to_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def get_query_embedding(image_bytes):
    img = bytes_to_image(image_bytes)

    if img is None:
        print("❌ Image decode failed")
        return None

    reps = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True
    )

    if not reps:
        return None

    emb = np.array(reps[0]["embedding"], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


# =============================
# GOOGLE REVERSE SEARCH
# =============================

def google_reverse_search(image_bytes):
    if not SERPAPI_API_KEY:
        return []

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    params = {
        "engine": "google_reverse_image",
        "image_base64": encoded,
        "api_key": SERPAPI_API_KEY
    }

    try:
        response = requests.post(
            "https://serpapi.com/search",
            data=params,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        results = []

        # Visual matches from Google
        for item in data.get("visual_matches", []):
            results.append({
                "image_url": item.get("thumbnail"),
                "source_url": item.get("link"),
                "similarity": 100.0
            })

        return results

    except Exception as e:
        print("SerpAPI error:", e)
        return []


# =============================
# SEARCH ROUTE
# =============================

@search_api.route("/api/search", methods=["POST"])
def search_face():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_bytes = request.files["file"].read()

    # =============================
    # 1️⃣ GOOGLE REVERSE SEARCH
    # =============================
    google_results = google_reverse_search(image_bytes)

    if google_results:
        return jsonify({
            "count": len(google_results),
            "matches": google_results[:5]
        })

    # =============================
    # 2️⃣ LOCAL EXACT MATCH (pHash)
    # =============================
    query_hash = compute_phash(image_bytes)

    for item in metadata:
        if "phash" not in item:
            continue

        dist = hamming_distance(query_hash, item["phash"])

        if dist <= 3:
            return jsonify({
                "count": 1,
                "matches": [{
                    "image_url": item["url"],
                    "similarity": 100.0
                }]
            })

    # =============================
    # 3️⃣ IDENTITY MATCH (FAISS)
    # =============================

    if index is None:
        return jsonify({"error": "Index not available"}), 500

    query_emb = get_query_embedding(image_bytes)

    if query_emb is None:
        return jsonify({
            "count": 0,
            "matches": [],
            "error": "Exactly one face required"
        })

    scores, indices = index.search(query_emb, TOP_K)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        similarity = ((float(score) + 1) / 2) * 100

        if similarity < MIN_SIMILARITY:
            continue

        results.append({
            "image_url": metadata[idx]["url"],
            "similarity": round(similarity, 2)
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return jsonify({
        "count": len(results),
        "matches": results[:5]
    })