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
def crawl_yahoo_images(query="face portrait", limit=20):
    urls = []
    try:
        search_url = "https://images.search.yahoo.com/search/images"
        params = {"p": query}

        r = requests.get(search_url, params=params, headers=HEADERS, timeout=15)

        if is_captcha_page(r.text):
            return {"captcha": True, "urls": []}

        soup = BeautifulSoup(r.text, "html.parser")
        imgs = soup.find_all("img", src=True)

        for img in imgs:
            src = img.get("src")
            if src and src.startswith("http"):
                urls.append(src)
                if len(urls) >= limit:
                    break

        time.sleep(random.uniform(2, 4))

    except Exception:
        pass

    return {"captcha": False, "urls": urls}


# --------------------------------------------------
# FLICKR IMAGE SOURCE (PUBLIC PHOTOS)
# --------------------------------------------------
def crawl_flickr_images(query="portrait", limit=20):
    urls = []
    try:
        search_url = "https://www.flickr.com/search/"
        params = {"text": query}

        r = requests.get(search_url, params=params, headers=HEADERS, timeout=15)

        if is_captcha_page(r.text):
            return {"captcha": True, "urls": []}

        soup = BeautifulSoup(r.text, "html.parser")
        imgs = soup.find_all("img", src=True)

        for img in imgs:
            src = img.get("src")
            if src and ("staticflickr" in src):
                urls.append(src)
                if len(urls) >= limit:
                    break

        time.sleep(random.uniform(2, 4))

    except Exception:
        pass

    return {"captcha": False, "urls": urls}
