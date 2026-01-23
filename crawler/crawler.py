import json
import os

from crawler.sources import (
    crawl_yahoo_images,
    crawl_flickr_images,
    crawl_wikimedia_images,
    crawl_pexels_images
)
from crawler.rate_limiter import RateLimiter
from config import CRAWLER_DELAY_SECONDS, CRAWLER_COOLDOWN_HOURS

# ===============================
# Paths
# ===============================

METADATA_PATH = "data/embeddings/metadata.json"
CRAWLED_URLS_FILE = "data/crawled_urls.json"

os.makedirs("data/embeddings", exist_ok=True)

# ===============================
# Persistent crawled URL storage
# ===============================

if os.path.exists(CRAWLED_URLS_FILE):
    with open(CRAWLED_URLS_FILE, "r") as f:
        crawled_urls = set(json.load(f))
else:
    crawled_urls = set()

# ===============================
# Runtime crawler state (API)
# ===============================

crawler_state = {
    "status": "idle",            # idle | running | paused | completed
    "captcha_required": False,
    "last_source": None,
    "collected_urls": []
}

# ===============================
# Rate limiter
# ===============================

rate_limiter = RateLimiter(
    delay_seconds=CRAWLER_DELAY_SECONDS,
    cooldown_hours=CRAWLER_COOLDOWN_HOURS
)

# ===============================
# Metadata append helper
# ===============================

def append_to_metadata(new_urls):
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_urls = {x["url"] for x in existing}

    added = 0
    for url in new_urls:
        if url not in existing_urls:
            existing.append({"url": url})
            added += 1

    with open(METADATA_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[CRAWLER] 📦 Metadata updated — {added} new URLs added")

# ===============================
# Main crawler entry
# ===============================

def crawl_all_sources():
    global crawler_state, crawled_urls

    print("[CRAWLER] Started")

    crawler_state["status"] = "running"
    crawler_state["captcha_required"] = False
    crawler_state["collected_urls"] = []

    # ===============================
    # Yahoo Images
    # ===============================
    print("[CRAWLER] Crawling Yahoo")
    rate_limiter.wait()

    yahoo = crawl_yahoo_images()

    if yahoo.get("captcha"):
        print("[CRAWLER] CAPTCHA detected on Yahoo")
        crawler_state["captcha_required"] = True
        crawler_state["status"] = "paused"
        rate_limiter.block()
        return

    new_yahoo = [u for u in yahoo.get("urls", []) if u not in crawled_urls]
    crawled_urls.update(new_yahoo)
    crawler_state["collected_urls"].extend(new_yahoo)
    crawler_state["last_source"] = "yahoo"

    append_to_metadata(new_yahoo)
    print(f"[CRAWLER] Yahoo collected {len(new_yahoo)} new images")

    # ===============================
    # Flickr
    # ===============================
    print("[CRAWLER] Crawling Flickr")
    rate_limiter.wait()

    flickr = crawl_flickr_images()

    if flickr.get("captcha"):
        print("[CRAWLER] CAPTCHA detected on Flickr")
        crawler_state["captcha_required"] = True
        crawler_state["status"] = "paused"
        rate_limiter.block()
        return

    new_flickr = [u for u in flickr.get("urls", []) if u not in crawled_urls]
    crawled_urls.update(new_flickr)
    crawler_state["collected_urls"].extend(new_flickr)
    crawler_state["last_source"] = "flickr"

    append_to_metadata(new_flickr)
    print(f"[CRAWLER] Flickr collected {len(new_flickr)} new images")

    # ===============================
    # Wikimedia
    # ===============================
    print("[CRAWLER] Crawling Wikimedia")
    rate_limiter.wait()

    wikimedia = crawl_wikimedia_images()

    new_wiki = [u for u in wikimedia.get("urls", []) if u not in crawled_urls]
    crawled_urls.update(new_wiki)
    crawler_state["collected_urls"].extend(new_wiki)
    crawler_state["last_source"] = "wikimedia"

    append_to_metadata(new_wiki)
    print(f"[CRAWLER] Wikimedia collected {len(new_wiki)} new images")

    # ===============================
    # Pexels
    # ===============================
    print("[CRAWLER] Crawling Pexels")
    rate_limiter.wait()

    pexels = crawl_pexels_images()

    new_pexels = [u for u in pexels.get("urls", []) if u not in crawled_urls]
    crawled_urls.update(new_pexels)
    crawler_state["collected_urls"].extend(new_pexels)
    crawler_state["last_source"] = "pexels"

    append_to_metadata(new_pexels)
    print(f"[CRAWLER] Pexels collected {len(new_pexels)} new images")

    # ===============================
    # Persist crawl results
    # ===============================
    with open(CRAWLED_URLS_FILE, "w") as f:
        json.dump(sorted(crawled_urls), f, indent=2)

    crawler_state["status"] = "completed"
    print("[CRAWLER] Completed")
