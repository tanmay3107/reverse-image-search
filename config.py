import os

# ---------------- PROJECT ROOT ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- PATHS ----------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
EMBEDDING_DIR = os.path.join(DATA_DIR, "embeddings")

# ---------------- FILE LIMITS ----------------
MAX_UPLOAD_SIZE_MB = 16
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ---------------- CRAWLER SETTINGS ----------------
CRAWLER_DELAY_SECONDS = 8        # delay between requests (anti-ban)
CRAWLER_COOLDOWN_HOURS = 3       # wait time after CAPTCHA / ban
MAX_IMAGES_PER_SOURCE = 50

# ---------------- FACE ENGINE ----------------
FACE_MODEL = "ArcFace"
FACE_DETECTOR = "retinaface"
SIMILARITY_THRESHOLD = 0.60

# ---------------- API ----------------
API_PORT = 5005
DEBUG = True

# ---------------- CREATE DIRS ----------------
for d in [UPLOAD_FOLDER, IMAGE_DIR, EMBEDDING_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- PAGEINATION ----------------
YAHOO_MAX_PAGES = 5
FLICKR_MAX_PAGES = 5
WIKIMEDIA_MAX_PAGES = 5