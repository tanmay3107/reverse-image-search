from crawler.sources import crawl_yahoo_images, crawl_flickr_images
from crawler.rate_limiter import RateLimiter
from config import CRAWLER_DELAY_SECONDS, CRAWLER_COOLDOWN_HOURS
from face_engine.indexer import build_embedding_index


crawler_state = {
    "status": "idle",
    "captcha_required": False,
    "last_source": None,
    "collected_urls": []
}

rate_limiter = RateLimiter(
    delay_seconds=CRAWLER_DELAY_SECONDS,
    cooldown_hours=CRAWLER_COOLDOWN_HOURS
)

def crawl_all_sources():
    global crawler_state

    print("[CRAWLER] Started")

    crawler_state["status"] = "running"
    crawler_state["captcha_required"] = False
    crawler_state["collected_urls"] = []

    # ---------------- Yahoo ----------------
    print("[CRAWLER] Crawling Yahoo")
    rate_limiter.wait()
    yahoo = crawl_yahoo_images()

    if yahoo["captcha"]:
        print("[CRAWLER] CAPTCHA detected on Yahoo")
        crawler_state["captcha_required"] = True
        crawler_state["status"] = "paused"
        rate_limiter.block()
        return

    crawler_state["collected_urls"].extend(yahoo["urls"])
    crawler_state["last_source"] = "yahoo"
    print(f"[CRAWLER] Yahoo collected {len(yahoo['urls'])} images")

    # ---------------- Flickr ----------------
    print("[CRAWLER] Crawling Flickr")
    rate_limiter.wait()
    flickr = crawl_flickr_images()

    if flickr["captcha"]:
        print("[CRAWLER] CAPTCHA detected on Flickr")
        crawler_state["captcha_required"] = True
        crawler_state["status"] = "paused"
        rate_limiter.block()
        return

    crawler_state["collected_urls"].extend(flickr["urls"])
    crawler_state["last_source"] = "flickr"
    print(f"[CRAWLER] Flickr collected {len(flickr['urls'])} images")

    # ---------------- Face Indexing ----------------
    print("[CRAWLER] Building face embedding index")
    try:
        build_embedding_index(crawler_state["collected_urls"])
    except Exception as e:
        print(f"[CRAWLER] Face indexing failed: {e}")

    crawler_state["status"] = "completed"
    print("[CRAWLER] Completed")
