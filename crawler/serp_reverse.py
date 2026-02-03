import requests
from config import SERPAPI_API_KEY

def serp_reverse_image_search(image_url=None, image_bytes=None):
    params = {
        "engine": "google_reverse_image",
        "api_key": SERPAPI_API_KEY
    }

    if image_url:
        params["image_url"] = image_url

    r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = []
    for res in data.get("image_results", []):
        if "link" in res:
            results.append(res["link"])

    return results
