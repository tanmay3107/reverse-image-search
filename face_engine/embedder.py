from deepface import DeepFace
import numpy as np

def extract_embedding(image_path: str):
    """
    Robust embedding extractor for crawled images.
    Uses Facenet512 (more tolerant than ArcFace).
    """
    try:
        reps = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",      # IMPORTANT CHANGE
            detector_backend="opencv",    # Match crawler detector
            enforce_detection=False
        )

        if not reps:
            return None

        embedding = np.array(reps[0]["embedding"], dtype="float32")
        return embedding

    except Exception as e:
        return None
