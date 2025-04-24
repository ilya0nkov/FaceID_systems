import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Инициализация модели
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

# Папка с фото пользователей (структура: /faces/Onkov_I_D/img1.jpg, Nagovitsyna_V_A/img1.jpg ...)
faces_dir = "faces"
embeddings_db = {}

# Извлечение эмбеддингов для всех пользователей
for user in os.listdir(faces_dir):
    user_embeddings = []
    for img_name in os.listdir(f"{faces_dir}/{user}"):
        img = cv2.imread(f"{faces_dir}/{user}/{img_name}")
        faces = app.get(img)
        if len(faces) > 0:
            user_embeddings.append(faces[0].embedding)

    if user_embeddings:
        embeddings_db[user] = np.mean(user_embeddings, axis=0)  # Усредняем эмбеддинги пользователя

# Сохраняем базу
np.save("embeddings_db.npy", embeddings_db)