import os
import json
import requests
import numpy as np
from io import BytesIO
from PIL import Image

from config import IMAGE_DIR, EMBEDDING_DIR
from face_engine.detector import has_face
from face_engine.embedder import extract_embedding

# Output files
EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, "face_embeddings.npy")
METADATA_FILE = os.path.join(EMBEDDING_DIR, "metadata.json")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)


# --------------------------------------------------
# Helper: Download image safely
# --------------------------------------------------
def download_image(url: str, save_path: str) -> bool:
    try:
        # Fix protocol-less URLs (e.g. //live.staticflickr.com/...)
        if url.startswith("//"):
            url = "https:" + url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return False

        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(save_path)
        return True

    except Exception:
        return False


# --------------------------------------------------
# Helper: Upscale small thumbnails
# --------------------------------------------------
def upscale_image(path: str):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size

        # Upscale tiny thumbnails for face detection
        if w < 224 or h < 224:
            new_w = max(224, w * 2)
            new_h = max(224, h * 2)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img.save(path)

    except Exception:
        pass


# --------------------------------------------------
# Main: Build face embedding index
# --------------------------------------------------
def build_embedding_index(image_urls: list):
    embeddings = []
    metadata = []

    print(f"[INDEXER] Starting indexing for {len(image_urls)} images")

    for idx, url in enumerate(image_urls):
        img_path = os.path.join(IMAGE_DIR, f"img_{idx}.jpg")

        print(f"[INDEXER] Processing {url}")

        # Download
        if not download_image(url, img_path):
            print("[INDEXER] Download failed — skipped")
            continue

        # Upscale for detection robustness
        upscale_image(img_path)

        # Face presence check (lightweight)
        if not has_face(img_path):
            print("[INDEXER] No face detected — skipped")
            continue

        # Extract ArcFace embedding
        embedding = extract_embedding(img_path)
        if embedding is None:
            print("[INDEXER] Embedding extraction failed — skipped")
            continue

        embeddings.append(embedding)
        metadata.append({
            "image_url": url,
            "local_path": img_path
        })

        print("[INDEXER] Face embedding stored")

    # Save results
    if embeddings:
        embeddings_array = np.vstack(embeddings).astype("float32")
        np.save(EMBEDDING_FILE, embeddings_array)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INDEXER] Stored {len(embeddings)} face embeddings")

    else:
        print("[INDEXER] No embeddings stored")
