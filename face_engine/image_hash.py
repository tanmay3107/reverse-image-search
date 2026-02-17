import imagehash
from PIL import Image
import io

def compute_phash(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None


def hamming_distance(hash1, hash2):
    if not hash1 or not hash2:
        return 999
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
