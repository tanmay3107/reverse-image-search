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
MIN_SIMILARITY = 70.0  # stricter threshold


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
        return None

    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    if len(faces) != 1:
        return None

    face_img = faces[0]["face"]

    rep = DeepFace.represent(
        img_path=face_img,
        model_name=MODEL_NAME,
        enforce_detection=False
    )

    emb = np.array(rep[0]["embedding"], dtype="float32").reshape(1, -1)

    # Normalize query
    faiss.normalize_L2(emb)

    return emb


# =============================
# SEARCH ROUTE
# =============================
@search_api.route("/api/search", methods=["POST"])
def search_face():
    if index is None:
        return jsonify({"error": "Index not available"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_bytes = request.files["file"].read()

    # =========================
    # 1️⃣ EXACT IMAGE MATCH
    # =========================
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

    # =========================
    # 2️⃣ IDENTITY MATCH
    # =========================
    query_emb = get_query_embedding(image_bytes)

    if query_emb is None:
        return jsonify({
            "count": 0,
            "matches": [],
            "error": "Exactly one face required"
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
        {"image_url": url, "similarity": round(sim, 2)}
        for url, sim in best_per_url.items()
    ]

    results.sort(key=lambda x: x["similarity"], reverse=True)
    print("Index total:", index.ntotal)

    return jsonify({
        "count": len(results),
        "matches": results[:5]
    })
