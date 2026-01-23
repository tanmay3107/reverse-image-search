from flask import Flask, jsonify, render_template
import threading
import os
import webbrowser
from dotenv import load_dotenv

from crawler.crawler import crawl_all_sources, crawler_state
from config import API_PORT, DEBUG, UPLOAD_FOLDER
from api.search_api import search_api

load_dotenv()

app = Flask(__name__, template_folder="templates")
app.register_blueprint(search_api)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# UI Routes
# =========================

@app.route("/")
def index():
    # Redirect-style landing page
    return render_template("search.html")


@app.route("/search")
def search_page():
    return render_template("search.html")


# =========================
# Crawler API
# =========================

@app.route("/api/crawl/start", methods=["GET", "POST"])
def start_crawler():
    if crawler_state["status"] == "running":
        return jsonify({"status": "crawler already running"})

    thread = threading.Thread(
        target=crawl_all_sources,
        daemon=True
    )
    thread.start()

    return jsonify({"status": "crawler started"})


@app.route("/api/crawl/status")
def crawler_status():
    return jsonify(crawler_state)


# =========================
# App entry
# =========================

if __name__ == "__main__":
    url = f"http://127.0.0.1:{API_PORT}/search"
    print(f"🚀 Opening browser at {url}")

    # Open browser automatically (once)
    webbrowser.open_new(url)

    app.run(
        debug=DEBUG,
        port=API_PORT,
        use_reloader=False  # prevents double browser open
    )
