import os
import json
import requests
import numpy as np
import faiss
import cv2
from deepface import DeepFace
from face_engine.image_hash import compute_phash

# =============================
# PATHS
# =============================
DATA_DIR = "data/embeddings"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "face_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

os.makedirs(DATA_DIR, exist_ok=True)

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
EMBEDDING_DIM = 512


# =============================
# DOWNLOAD IMAGE
# =============================
def download_image(url):
    if url.startswith("//"):
        url = "https:" + url

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.content


# =============================
# EXTRACT SINGLE FACE EMBEDDING
# =============================
def extract_embedding(img_bytes):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return None

    # Detect faces explicitly
    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False
    )

    # Only accept exactly 1 face
    if len(faces) != 1:
        return None

    face_img = faces[0]["face"]

    rep = DeepFace.represent(
        img_path=face_img,
        model_name=MODEL_NAME,
        enforce_detection=False
    )

    if not rep:
        return None

    emb = np.array(rep[0]["embedding"], dtype="float32")
    return emb


# =============================
# REBUILD INDEX
# =============================
def rebuild_index_from_urls(image_urls):
    print(f"[INDEXER] Rebuilding from {len(image_urls)} URLs")

    embeddings = []
    metadata = []

    for i, url in enumerate(image_urls, start=1):
        print(f"[INDEXER] ({i}/{len(image_urls)}) Processing")

        try:
            img_bytes = download_image(url)

            emb = extract_embedding(img_bytes)
            if emb is None:
                print("[INDEXER] Skipped (no single face)")
                continue

            phash = compute_phash(img_bytes)

            embeddings.append(emb)
            metadata.append({
                "url": url,
                "phash": phash
            })

            print("[INDEXER] Stored")

        except Exception as e:
            print("[INDEXER] Failed:", e)
            continue

    if not embeddings:
        print("[INDEXER] ❌ No embeddings created")
        return

    embeddings = np.vstack(embeddings).astype("float32")

    # Normalize once
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    np.save(EMBEDDINGS_FILE, embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print(f"[INDEXER] ✅ Rebuild complete: {len(embeddings)} embeddings saved")
