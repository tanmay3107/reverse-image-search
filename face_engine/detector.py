from deepface import DeepFace

def has_face(image_path: str) -> bool:
    """
    Lightweight face check for crawled images.
    Optimized for low-resolution thumbnails.
    """
    try:
        detections = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",  # IMPORTANT
            enforce_detection=False
        )
        return len(detections) > 0
    except Exception:
        return False
