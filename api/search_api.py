import os
import json
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
PAGE_SIZE_DEFAULT = 10
MIN_SIMILARITY = 60.0  # percent

# =========================
# Load index ONCE
# =========================

if not os.path.exists(FAISS_INDEX_PATH):
    print("[SEARCH] ❌ FAISS index not found")
    index = None
    metadata = []
else:
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print(f"[SEARCH] ✅ Loaded {index.ntotal} face embeddings")

# =========================
# Helpers
# =========================

def bytes_to_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def get_query_embedding(image_bytes):
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

# =========================
# Search Endpoint
# =========================

@search_api.route("/api/search", methods=["POST"])
def search_face():
    if index is None or index.ntotal == 0:
        return jsonify({
            "count": 0,
            "matches": [],
            "error": "Face index not available"
        }), 500

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["file"].read()

    page = int(request.form.get("page", 1))
    page_size = int(request.form.get("page_size", PAGE_SIZE_DEFAULT))
    offset = (page - 1) * page_size

    query_emb = get_query_embedding(image_bytes)
    if query_emb is None:
        return jsonify({
            "count": 0,
            "matches": [],
            "error": "No face detected in query image"
        }), 200

    # FAISS cosine similarity search
    scores, indices = index.search(query_emb, TOP_K)

    # -----------------------------
    # DEDUPLICATION BY URL
    # -----------------------------

    best_per_url = {}

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        url = metadata[idx]["url"]

        similarity = round(((float(score) + 1) / 2) * 100, 2)
        if similarity < MIN_SIMILARITY:
            continue

        # keep best similarity per URL
        if url not in best_per_url or similarity > best_per_url[url]:
            best_per_url[url] = similarity

    # Convert to list
    results = [
        {"image_url": url, "similarity": sim}
        for url, sim in best_per_url.items()
    ]

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    total_matches = len(results)
    paginated = results[offset: offset + page_size]

    return jsonify({
        "count": total_matches,
        "matches": paginated,
        "page": page,
        "page_size": page_size
    })
