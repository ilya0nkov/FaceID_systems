import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from utils.db_utils import load_db, save_db
from utils.ai_models import face_app, THRESHOLD

warnings.filterwarnings("ignore", category=FutureWarning)

REGISTRATION_PHOTOS = 7  # Сколько фото использовать для регистрации


# --- Усреднение эмбеддингов ---
def get_average_embedding(image_paths):
    embeddings = []
    for path in image_paths:
        img = cv2.imread(path)
        faces = face_app.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)

    if not embeddings:
        return None
    return np.mean(embeddings, axis=0).tolist()  # Усредняем


# --- Регистрация пользователя ---
def register_user(name, photo_folder):
    image_paths = []
    for img in os.listdir(photo_folder):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(photo_folder, img)
            image_paths.append(full_path)
            if len(image_paths) >= REGISTRATION_PHOTOS:
                break

    if not image_paths:
        print(f"Нет фото для пользователя {name}!")
        return False

    avg_embedding = get_average_embedding(image_paths)
    if avg_embedding is None:
        print(f"Не удалось извлечь лицо из фото пользователя {name}!")
        return False

    db = load_db()
    db[name] = avg_embedding

    if not save_db(db):
        print(f"ОШИБКА: Не удалось сохранить пользователя {name} в базу!")
        return False

    print(f"УСПЕХ: Пользователь {name} зарегистрирован по {len(image_paths)} фото")
    return True


# --- Проверка лица ---
def check_face(image_path):
    db = load_db()
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    if len(faces) == 0:
        print("Лицо не обнаружено!")
        return None

    last_similarity = 0
    for name, saved_embedding in db.items():
        similarity = cosine_similarity([faces[0].embedding], [saved_embedding])[0][0]
        if similarity >= THRESHOLD:
            print(f"Доступ разрешён: {name} (сходство: {similarity:.8f})")
            return name
        last_similarity = similarity

    print(f"Доступ запрещён! Неизвестное лицо.(сходство: {last_similarity:.8f})")
    return None


# Пример использования
def main():
    # Регистрация: каждая папка в 'data' — пользователь с N фото\

    print("началась регистрация")
    for user_folder in os.listdir("data"):
        register_user(user_folder, f"data/{user_folder}")
    print("закончилась регистрация")


if __name__ == "__main__":
    main()

