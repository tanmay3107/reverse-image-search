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

# --------------------------------------------------
# YAHOO IMAGE SOURCE (KEYWORD-BASED)
# --------------------------------------------------
import requests
from bs4 import BeautifulSoup
from config import YAHOO_MAX_PAGES

def crawl_yahoo_images():
    collected = []
    captcha = False

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for page in range(YAHOO_MAX_PAGES):
        start = page * 20  # Yahoo uses offset

        url = (
            "https://images.search.yahoo.com/search/images"
            f"?p=face+portrait&b={start}"
        )

        try:
            r = requests.get(url, headers=headers, timeout=10)

            if r.status_code != 200:
                continue

            if "captcha" in r.text.lower():
                captcha = True
                break

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

            print(f"[YAHOO] Page {page+1}: {page_urls} images")

            # Stop early if page is empty
            if page_urls == 0:
                break

        except Exception:
            continue

    return {
        "urls": collected,
        "captcha": captcha
    }



# --------------------------------------------------
# FLICKR IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------
import requests
from bs4 import BeautifulSoup
from config import FLICKR_MAX_PAGES

def crawl_flickr_images():
    collected = []
    captcha = False

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for page in range(1, FLICKR_MAX_PAGES + 1):
        url = f"https://www.flickr.com/search/?text=portrait&page={page}"

        try:
            r = requests.get(url, headers=headers, timeout=10)

            if r.status_code != 200:
                continue

            if "captcha" in r.text.lower():
                captcha = True
                break

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

            print(f"[FLICKR] Page {page}: {page_urls} images")

            if page_urls == 0:
                break

        except Exception:
            continue

    return {
        "urls": collected,
        "captcha": captcha
    }

# --------------------------------------------------
# WIKIMEDIA IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------

import requests
from bs4 import BeautifulSoup
from config import WIKIMEDIA_MAX_PAGES


def crawl_wikimedia_images():
    collected = []
    captcha = False

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for page in range(1, WIKIMEDIA_MAX_PAGES + 1):

        # Wikimedia Commons media search (faces / portraits)
        url = (
            "https://commons.wikimedia.org/w/index.php"
            "?search=human+face+portrait"
            f"&offset={(page - 1) * 20}"
            "&limit=20"
            "&title=Special:MediaSearch"
            "&type=image"
        )

        try:
            r = requests.get(url, headers=headers, timeout=10)

            if r.status_code != 200:
                continue

            if "captcha" in r.text.lower():
                captcha = True
                break

            soup = BeautifulSoup(r.text, "html.parser")
            imgs = soup.find_all("img")

            page_urls = 0
            for img in imgs:
                src = img.get("src")
                if not src:
                    continue

                if src.startswith("//"):
                    src = "https:" + src

                # Wikimedia images always come from upload.wikimedia.org
                if "upload.wikimedia.org" in src:
                    collected.append(src)
                    page_urls += 1

            print(f"[WIKIMEDIA] Page {page}: {page_urls} images")

            if page_urls == 0:
                break

        except Exception:
            continue

    return {
        "urls": collected,
        "captcha": captcha
    }

# --------------------------------------------------
# PEXELS IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------

import requests
from config import PEXELS_API_KEY, PEXELS_MAX_PAGES


def crawl_pexels_images():
    collected = []
    captcha = False

    headers = {
        "Authorization": PEXELS_API_KEY
    }

    query = "old man portrait"
    per_page = 30  # Pexels max

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

            data = r.json()
            photos = data.get("photos", [])

            page_urls = 0
            for photo in photos:
                src = photo.get("src", {}).get("medium")
                if src:
                    collected.append(src)
                    page_urls += 1

            print(f"[PEXELS] Page {page}: {page_urls} images")

            if page_urls == 0:
                break

        except Exception:
            continue

    return {
        "urls": collected,
        "captcha": captcha
    }

# --------------------------------------------------
# SERPAPI IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------

import requests
from config import UNSPLASH_MAX_PAGES, SERPAPI_API_KEY

def crawl_unsplash_images():
    collected = []
    captcha = False

    if not SERPAPI_API_KEY:
        print("[UNSPLASH] ❌ SERPAPI_API_KEY not set")
        return {"urls": [], "captcha": False}

    for page in range(0, UNSPLASH_MAX_PAGES):
        params = {
            "engine": "google_images",
            "q": "portrait face site:unsplash.com",
            "ijn": page,  # image page index
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

            print(f"[UNSPLASH] Page {page + 1}: {page_urls} images")

            if page_urls == 0:
                break

        except Exception as e:
            print(f"[UNSPLASH] Error: {e}")
            continue

    return {
        "urls": collected,
        "captcha": False
    }
