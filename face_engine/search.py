import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
from config import EMBEDDING_DIR

EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, "face_embeddings.npy")
METADATA_FILE = os.path.join(EMBEDDING_DIR, "metadata.json")


def extract_query_embedding(image_path: str):
    """
    High-precision embedding for query image.
    Uses ArcFace.
    """
    try:
        reps = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )

        if not reps:
            return None

        return np.array(reps[0]["embedding"], dtype="float32")

    except Exception:
        return None


def search_similar_faces(query_image_path: str, top_k=5):
    """
    Compare query face with indexed faces.
    """
    if not os.path.exists(EMBEDDING_FILE):
        return []

    # Load index
    embeddings = np.load(EMBEDDING_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Extract query embedding
    query_embedding = extract_query_embedding(query_image_path)
    if query_embedding is None:
        return []

    # Compute cosine similarity
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    # Rank results
    ranked = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []
    for idx, score in ranked:
        results.append({
            "similarity": round(float(score) * 100, 2),
            "image_url": metadata[idx]["image_url"],
            "local_path": metadata[idx]["local_path"]
        })

    return results
