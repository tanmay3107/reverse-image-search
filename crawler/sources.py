import requests
import time
import random
from bs4 import BeautifulSoup
from crawler.captcha_detector import is_captcha_page

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

FACE_QUERIES = [
    "portrait photography face close up",
    "close up human face photo",
    "professional headshot face",
    "street photography portrait single person",
    "human face candid portrait"
]

# --------------------------------------------------
# YAHOO IMAGE SOURCE (KEYWORD-BASED)
# --------------------------------------------------
from config import YAHOO_MAX_PAGES

def crawl_yahoo_images():
    collected = []
    captcha = False

    for query in FACE_QUERIES:
        query_encoded = query.replace(" ", "+")

        for page in range(YAHOO_MAX_PAGES):
            start = page * 20

            url = (
                "https://images.search.yahoo.com/search/images"
                f"?p={query_encoded}&b={start}"
            )

            try:
                r = requests.get(url, headers=HEADERS, timeout=10)

                if r.status_code != 200:
                    continue

                if "captcha" in r.text.lower():
                    captcha = True
                    return {"urls": collected, "captcha": True}

                soup = BeautifulSoup(r.text, "html.parser")
                imgs = soup.find_all("img")

                page_urls = 0
                for img in imgs:
                    src = img.get("src") or img.get("data-src")
                    if not src:
                        continue

                    if src.startswith("//"):
                        src = "https:" + src

                    if src.startswith("http"):
                        collected.append(src)
                        page_urls += 1

                print(f"[YAHOO] {query} | Page {page+1}: {page_urls}")

                if page_urls == 0:
                    break

            except Exception:
                continue

            time.sleep(random.uniform(0.8, 1.5))

    return {"urls": collected, "captcha": False}

# --------------------------------------------------
# FLICKR IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------
from config import FLICKR_MAX_PAGES

def crawl_flickr_images():
    collected = []
    captcha = False

    for query in FACE_QUERIES:
        query_encoded = query.replace(" ", "+")

        for page in range(1, FLICKR_MAX_PAGES + 1):
            url = f"https://www.flickr.com/search/?text={query_encoded}&page={page}"

            try:
                r = requests.get(url, headers=HEADERS, timeout=10)

                if r.status_code != 200:
                    continue

                if "captcha" in r.text.lower():
                    captcha = True
                    return {"urls": collected, "captcha": True}

                soup = BeautifulSoup(r.text, "html.parser")
                imgs = soup.find_all("img")

                page_urls = 0
                for img in imgs:
                    src = img.get("src")
                    if not src:
                        continue

                    if src.startswith("//"):
                        src = "https:" + src

                    if "staticflickr.com" in src:
                        collected.append(src)
                        page_urls += 1

                print(f"[FLICKR] {query} | Page {page}: {page_urls}")

                if page_urls == 0:
                    break

            except Exception:
                continue

            time.sleep(random.uniform(1.0, 1.8))

    return {"urls": collected, "captcha": False}

# --------------------------------------------------
# WIKIMEDIA IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------
from config import WIKIMEDIA_MAX_PAGES

def crawl_wikimedia_images():
    collected = []
    captcha = False

    for query in FACE_QUERIES:
        query_encoded = query.replace(" ", "+")

        for page in range(1, WIKIMEDIA_MAX_PAGES + 1):
            url = (
                "https://commons.wikimedia.org/w/index.php"
                f"?search={query_encoded}"
                f"&offset={(page - 1) * 20}"
                "&limit=20"
                "&title=Special:MediaSearch"
                "&type=image"
            )

            try:
                r = requests.get(url, headers=HEADERS, timeout=10)

                if r.status_code != 200:
                    continue

                soup = BeautifulSoup(r.text, "html.parser")
                imgs = soup.find_all("img")

                page_urls = 0
                for img in imgs:
                    src = img.get("src")
                    if src and "upload.wikimedia.org" in src:
                        if src.startswith("//"):
                            src = "https:" + src
                        collected.append(src)
                        page_urls += 1

                print(f"[WIKIMEDIA] {query} | Page {page}: {page_urls}")

                if page_urls == 0:
                    break

            except Exception:
                continue

            time.sleep(1.2)

    return {"urls": collected, "captcha": False}

# --------------------------------------------------
# PEXELS IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------
from config import PEXELS_API_KEY, PEXELS_MAX_PAGES

def crawl_pexels_images():
    collected = []
    captcha = False

    headers = {"Authorization": PEXELS_API_KEY}
    per_page = 30

    for query in FACE_QUERIES:
        for page in range(1, PEXELS_MAX_PAGES + 1):
            url = (
                "https://api.pexels.com/v1/search"
                f"?query={query}"
                f"&page={page}"
                f"&per_page={per_page}"
            )

            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code != 200:
                    continue

                photos = r.json().get("photos", [])
                page_urls = 0

                for photo in photos:
                    src = photo.get("src", {}).get("medium")
                    if src:
                        collected.append(src)
                        page_urls += 1

                print(f"[PEXELS] {query} | Page {page}: {page_urls}")

                if page_urls == 0:
                    break

            except Exception:
                continue

            time.sleep(0.7)

    return {"urls": collected, "captcha": False}

# --------------------------------------------------
# SERPAPI IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------
from config import UNSPLASH_MAX_PAGES, SERPAPI_API_KEY

def crawl_unsplash_images():
    collected = []

    if not SERPAPI_API_KEY:
        print("[UNSPLASH] ❌ SERPAPI_API_KEY not set")
        return {"urls": [], "captcha": False}

    for query in FACE_QUERIES:
        for page in range(UNSPLASH_MAX_PAGES):
            params = {
                "engine": "google_images",
                "q": f"{query} site:unsplash.com",
                "ijn": page,
                "api_key": SERPAPI_API_KEY
            }

            try:
                r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
                r.raise_for_status()
                data = r.json()

                images = data.get("images_results", [])
                page_urls = 0

                for img in images:
                    url = img.get("original")
                    if url and "unsplash.com" in url:
                        collected.append(url)
                        page_urls += 1

                print(f"[UNSPLASH] {query} | Page {page+1}: {page_urls}")

                if page_urls == 0:
                    break

            except Exception as e:
                print(f"[UNSPLASH] Error: {e}")
                continue

            time.sleep(1.0)

    return {"urls": collected, "captcha": False}