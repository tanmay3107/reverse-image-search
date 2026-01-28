# FaceTrace AI — Reverse Face Search System

FaceTrace AI is a research-grade reverse face search system that identifies visually similar faces across public web images. It combines large-scale image crawling, deep face embeddings, vector similarity search, and a web-based UI.

---

## 🔍 Core Capabilities
* **Reverse Face Search:** Using FaceNet512 embeddings.
* **Multi-Source Crawling:** Yahoo, Flickr, Wikimedia, Pexels, and SerpAPI.
* **Vector Search:** Offline FAISS index construction using Cosine Similarity.
* **Deduplication:** Duplicate-safe metadata handling for efficient storage.
* **Web UI:** Flask-based interface for image uploads and results.

---

## 🧠 System Architecture

1. **Image Crawling:** Collects public images via keyword-based queries.
2. **Offline Indexing:** - Detects faces using **RetinaFace**.
   - Extracts embeddings using **FaceNet512**.
   - Indexes embeddings using **FAISS**.
3. **Search & UI:** Query image embeddings are compared against the index to retrieve Top-K matches.

---

## 📁 Project Structure

```
.
├── api/                # Search API (FAISS + DeepFace)
├── crawler/            # Crawling orchestration & sources
├── face_engine/        # FAISS index rebuild logic
├── templates/          # Web UI (search.html)
├── app.py              # Flask app entry point
├── config.py           # Central configuration
├── requirements.txt
└── README.md
```
---

## ⚙️ Setup & Installation

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Configuration
Create a .env file in the root directory:
PEXELS_API_KEY=your_key_here
SERPAPI_API_KEY=your_key_here

### 3. Start the App
python app.py
Visit: http://127.0.0.1:5005

---

## 🧱 Rebuilding the FAISS Index
To update the search index after crawling new images, run this command:

python -c "from face_engine.indexer import rebuild_index_from_urls; import json; urls = json.load(open('data/embeddings/metadata.json')); rebuild_index_from_urls([x['url'] for x in urls])"

---

## 📊 Similarity Scoring
FAISS retrieves results based on Cosine Similarity, which is converted to a percentage:

similarity = ((cosine_score + 1) / 2) * 100

* **Threshold:** Default minimum 60% confidence.

---

## 👤 Author
**Tanmay Janak** Final Year AIML Student  
*Focus Areas: Computer Vision, Deep Learning, MLOps*

---

## 📜 License
This project is provided for educational and research purposes only. No copyrighted content or pre-built indexes are included in this repository.