import json
import os
from crawler.sources import crawl_wikimedia_images
from crawler.sources import crawl_yahoo_images, crawl_flickr_images
from crawler.rate_limiter import RateLimiter
from config import CRAWLER_DELAY_SECONDS, CRAWLER_COOLDOWN_HOURS

# -------------------------------
# Persistent storage
# -------------------------------
CRAWLED_URLS_FILE = "data/crawled_urls.json"
os.makedirs("data", exist_ok=True)

# Load previously crawled URLs
if os.path.exists(CRAWLED_URLS_FILE):
    with open(CRAWLED_URLS_FILE, "r") as f:
        crawled_urls = set(json.load(f))
else:
    crawled_urls = set()

# -------------------------------
# Runtime crawler state (API)
# -------------------------------
crawler_state = {
    "status": "idle",          # idle | running | paused | completed
    "captcha_required": False,
    "last_source": None,
    "collected_urls": []
}

# -------------------------------
# Rate limiter
# -------------------------------
rate_limiter = RateLimiter(
    delay_seconds=CRAWLER_DELAY_SECONDS,
    cooldown_hours=CRAWLER_COOLDOWN_HOURS
)

# -------------------------------
# Main crawler entry
# -------------------------------
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

    new_yahoo_urls = [
        url for url in yahoo.get("urls", [])
        if url not in crawled_urls
    ]

    crawled_urls.update(new_yahoo_urls)
    crawler_state["collected_urls"].extend(new_yahoo_urls)
    crawler_state["last_source"] = "yahoo"

    print(f"[CRAWLER] Yahoo collected {len(new_yahoo_urls)} new images")

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

    new_flickr_urls = [
        url for url in flickr.get("urls", [])
        if url not in crawled_urls
    ]

    crawled_urls.update(new_flickr_urls)
    crawler_state["collected_urls"].extend(new_flickr_urls)
    crawler_state["last_source"] = "flickr"

    print(f"[CRAWLER] Flickr collected {len(new_flickr_urls)} new images")

    # ===============================
    # wikimedia
    # ===============================
    print("[CRAWLER] Crawling Wikimedia")
    wikimedia = crawl_wikimedia_images()

    new_urls = [u for u in wikimedia["urls"] if u not in crawled_urls]
    crawler_state["collected_urls"].extend(new_urls)
    crawled_urls.update(new_urls)

    crawler_state["last_source"] = "wikimedia"
    print(f"[CRAWLER] Wikimedia collected {len(new_urls)} new images")


    # ===============================
    # Save crawl results
    # ===============================
    with open(CRAWLED_URLS_FILE, "w") as f:
        json.dump(sorted(crawled_urls), f, indent=2)

    crawler_state["status"] = "completed"
    print("[CRAWLER] Completed")
