import cv2
from sklearn.metrics.pairwise import cosine_similarity
from utils.db_utils import load_db
from utils.ai_models import face_app, THRESHOLD
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Проверка лица в реальном времени
def check_live_camera():
    db = load_db()
    cap = cv2.VideoCapture(0)  # Открываем камеру (0 — стандартная камера)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения камеры!")
            break

        # Детекция лиц
        faces = face_app.get(frame)

        # Рисуем рамку и выводим результат
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Сравнение с базой
            max_similarity = 0
            best_match = None
            for name, saved_embedding in db.items():
                similarity = cosine_similarity([face.embedding], [saved_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = name

            if best_match and max_similarity >= THRESHOLD:
                text = f"Access: {best_match} ({max_similarity:.2f})"
                color = (0, 255, 0)  # Зелёный — доступ разрешён
            else:
                text = f"Unknown (max: {max_similarity:.2f})"
                color = (0, 0, 255)  # Красный — доступ запрещён

            cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Вывод кадра
        cv2.imshow("FaceID Test", frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Запуск теста камеры... (Нажмите 'q' для выхода)")
    check_live_camera()