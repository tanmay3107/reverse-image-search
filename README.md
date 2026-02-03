# FaceTrace AI — Reverse Face Search System

FaceTrace AI is a research-grade reverse face search system that identifies visually similar faces across public web images. It combines large-scale image crawling, deep face embeddings, vector similarity search, and a web-based UI.

---

## 🔍 Core Capabilities

* **True Reverse Image Search** using Google Reverse Image Search (SerpAPI)
* **Identity Verification** using FaceNet512 + RetinaFace
* **Fallback Local Search** using FAISS (cosine similarity)
* **Multi-source Crawling** (Yahoo, Flickr, Wikimedia, Pexels)
* **Offline Vector Indexing**
* **Web-based UI** for uploads and results
* **Confidence-based ranking**
* **Source URL attribution**
---

## 🧠 System Architecture

1. **Image Crawling (Offline)**
   Public images are collected from multiple sources
   URLs are deduplicated and stored as metadata
2. **Offline Indexing**
   Faces detected using RetinaFace
   Embeddings extracted using FaceNet512
   Embeddings normalized and indexed using FAISS (cosine similarity)
3. **Reverse Search Pipeline (Online)**
   When a user uploads an image:

* **Stage 1** — Web-Scale Reverse Search
   The uploaded image is sent to Google Reverse Image Search via SerpAPI
   Candidate web images are retrieved
* **Stage 2** — Identity Verification
   Face embeddings are extracted from each candidate image
   Embeddings are compared against the query face
   Only same-person matches above a confidence threshold are returned
* **Stage 3** — Fallback (If Needed)
   If no web matches are found, the system falls back to:
   Local FAISS-based face similarity search

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