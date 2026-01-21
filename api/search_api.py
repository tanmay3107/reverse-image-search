import os
import time
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from face_engine.search import search_similar_faces
from config import UPLOAD_FOLDER

search_api = Blueprint("search_api", __name__)


@search_api.route("/api/search", methods=["POST"])
def face_search():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, f"query_{int(time.time())}_{filename}")
    file.save(path)

    results = search_similar_faces(path)

    return jsonify({
        "matches": results,
        "count": len(results)
    })
