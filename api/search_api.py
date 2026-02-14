import os
import json
import numpy as np
import cv2
import faiss
from flask import Blueprint, request, jsonify
from deepface import DeepFace

from face_engine.image_hash import compute_phash, hamming_distance

search_api = Blueprint("search_api", __name__)

BASE_DIR = "data/embeddings"
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
EMBEDDING_DIM = 512

TOP_K = 50
MIN_SIMILARITY = 75.0   # stricter identity threshold
PHASH_THRESHOLD = 6     # exact / near-exact match

# -----------------------------
# Load index ONCE
# -----------------------------
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
else:
    index = None
    metadata = []

# -----------------------------
# Helpers
# -----------------------------
def bytes_to_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_embedding(image_bytes):
    img = bytes_to_image(image_bytes)
    if img is None:
        return None

    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    # ❗ single-face enforcement
    if len(faces) != 1:
        return None

    rep = DeepFace.represent(
        img_path=faces[0]["face"],
        model_name=MODEL_NAME,
        enforce_detection=False
    )

    emb = np.array(rep[0]["embedding"], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


# -----------------------------
# Search Endpoint
# -----------------------------
@search_api.route("/api/search", methods=["POST"])
def search_face():
    if index is None:
        return jsonify({"error": "Index not available"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["file"].read()

    # =============================
    # 1️⃣ EXACT IMAGE MATCH (TRUE RIS)
    # =============================
    query_hash = compute_phash(image_bytes)

    exact_matches = []
    for item in metadata:
        if "phash" not in item:
            continue

        dist = hamming_distance(query_hash, item["phash"])
        if dist <= PHASH_THRESHOLD:
            exact_matches.append({
                "image_url": item["url"],
                "similarity": 100 - dist * 5,
                "match_type": "exact"
            })

    if exact_matches:
        return jsonify({
            "count": len(exact_matches),
            "matches": sorted(exact_matches, key=lambda x: x["similarity"], reverse=True),
            "mode": "exact_match"
        })

    # =============================
    # 2️⃣ IDENTITY VERIFICATION
    # =============================
    query_emb = extract_embedding(image_bytes)
    if query_emb is None:
        return jsonify({
            "count": 0,
            "matches": [],
            "error": "No single face detected"
        })

    scores, indices = index.search(query_emb, TOP_K)

    best_per_url = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        similarity = ((float(score) + 1) / 2) * 100
        if similarity < MIN_SIMILARITY:
            continue

        url = metadata[idx]["url"]
        if url not in best_per_url or similarity > best_per_url[url]:
            best_per_url[url] = similarity

    results = [
        {
            "image_url": url,
            "similarity": round(sim, 2),
            "match_type": "identity"
        }
        for url, sim in best_per_url.items()
    ]

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return jsonify({
        "count": len(results),
        "matches": results[:10],
        "mode": "identity_verification"
    })
