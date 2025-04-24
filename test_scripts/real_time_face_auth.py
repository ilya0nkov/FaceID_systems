import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from FaceID_System import liveness_detector
from utils.db_utils import load_db
from utils.ai_models import face_app, THRESHOLD

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/check_face")
async def check_face(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if not faces:
        return {"access": False, "message": "Лицо не обнаружено!"}

    db = load_db()
    for name, saved_embedding in db.items():
        similarity = cosine_similarity([faces[0].embedding], [saved_embedding])[0][0]
        if similarity >= THRESHOLD:
            return {"access": True, "message": f"Доступ разрешён: {name}"}

    return {"access": False, "message": "Доступ запрещён!"}


@app.post("/api/check_liveness")
async def check_liveness(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    is_real, message = liveness_detector.verify(image)
    return {"is_real": is_real, "message": message}