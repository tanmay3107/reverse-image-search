from flask import Flask, jsonify, render_template
import threading
import os

from crawler.crawler import crawl_all_sources, crawler_state
from config import API_PORT, DEBUG, UPLOAD_FOLDER
from api.search_api import search_api

app = Flask(__name__, template_folder="templates")
app.register_blueprint(search_api)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/search")
def search_page():
    return render_template("search.html")

@app.route("/")
def index():
    return "<h2>Reverse Image Search with Face Filtering</h2>"

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

if __name__ == "__main__":
    app.run(debug=DEBUG, port=API_PORT)
