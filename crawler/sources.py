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
