import imagehash
from PIL import Image
import io

def compute_phash(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return str(imagehash.phash(img))

def hamming_distance(h1, h2):
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
