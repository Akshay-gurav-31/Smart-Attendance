from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
from modules.embeddings import EmbeddingModel
from api.face_detect_mediapipe import detect_faces_mediapipe
from modules.redis_store import RedisEmbeddings
import numpy as np
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = EmbeddingModel()
redis_store = RedisEmbeddings()

@app.post("/api/detect-and-embed")
async def detect_and_embed(student_id: str = Form(...), photos: list[UploadFile] = []):
    if not photos:
        return {"error": "No images uploaded"}

    face_arrays = []
    for photo in photos:
        contents = await photo.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cropped_faces = detect_faces_mediapipe(img)
        face_arrays.extend(cropped_faces)

    if not face_arrays:
        return {"error": "No valid faces detected"}

    embedding_vector = embedder.get_average_embedding_from_arrays(face_arrays)
    if embedding_vector is None:
        return {"error": "Failed to generate embedding"}

    # Store embedding in Redis
    redis_store.set_embedding(student_id, embedding_vector)

    return {
        "student_id": student_id,
        "faces_detected": len(face_arrays),
        "embedding_length": len(embedding_vector),
        "embedding_preview": embedding_vector[:5].tolist(),
        "message": "Embedding saved in Redis successfully"
    }
