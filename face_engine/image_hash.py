from PIL import Image
import imagehash
import io

def compute_phash(image_bytes) -> str:
    """
    Compute perceptual hash (pHash) for exact / near-exact image matching.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return str(imagehash.phash(img))


def hamming_distance(h1: str, h2: str) -> int:
    """
    Compute Hamming distance between two hex hashes.
    """
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")
