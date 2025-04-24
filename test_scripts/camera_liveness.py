import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Настройки
THRESHOLD = 0.6  # Порог схожести лиц
DB_FILE = "../face_db.json"
LIVENESS_THRESHOLD = 15  # Градусов поворота для проверки движения
EYE_AR_THRESHOLD = 0.2  # Порог закрытия глаз


class LivenessDetector:
    def __init__(self):
        self.prev_head_angles = None
        self.eye_blink_counter = 0
        self.liveness_start_time = None

    def get_head_angles(self, landmarks_3d):
        """Вычисляет углы поворота головы (yaw, pitch) в градусах."""
        nose = landmarks_3d[30]  # Кончик носа
        chin = landmarks_3d[8]  # Подбородок
        left_eye = landmarks_3d[36]
        right_eye = landmarks_3d[45]

        vec_horizontal = right_eye - left_eye
        vec_vertical = chin - nose

        yaw = np.arctan2(vec_horizontal[2], vec_horizontal[0]) * 180 / np.pi
        pitch = np.arctan2(vec_vertical[2], vec_vertical[1]) * 180 / np.pi
        return yaw, pitch

    def check_eye_blink(self, landmarks_2d):
        """Проверяет, закрыты ли глаза (Eye Aspect Ratio)."""

        def ear(eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C)

        left_eye = landmarks_2d[42:48]
        right_eye = landmarks_2d[36:42]
        ear_avg = (ear(left_eye) + ear(right_eye)) / 2.0
        return ear_avg < EYE_AR_THRESHOLD

    def verify(self, face):
        """Проверяет liveness (движение + мигание)."""
        # Первая инициализация
        if self.liveness_start_time is None:
            self.liveness_start_time = time.time()
            self.prev_head_angles = self.get_head_angles(face.landmark_3d_68)
            return False, "Turn your head"

        # Проверка движения головы
        current_yaw, current_pitch = self.get_head_angles(face.landmark_3d_68)
        dyaw = abs(current_yaw - self.prev_head_angles[0])
        dpitch = abs(current_pitch - self.prev_head_angles[1])

        if dyaw > LIVENESS_THRESHOLD or dpitch > LIVENESS_THRESHOLD:
            return True, "Liveness confirmed"

        # Проверка мигания (опционально)
        if self.check_eye_blink(face.landmark_2d_106):
            self.eye_blink_counter += 1
            if self.eye_blink_counter >= 2:
                return True, "Liveness confirmed "

        return False, f"Turn your head (current angle: yaw={dyaw:.1f}, pitch={dpitch:.1f})"


def load_db():
    """Загружает базу лиц."""
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():
    # Инициализация
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    db = load_db()
    liveness_detector = LivenessDetector()
    cap = cv2.VideoCapture(0)

    print("Running the camera test... (Press 'q' to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция лиц
        faces = app.get(frame)
        if len(faces) == 0:
            cv2.putText(frame, "The face was not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("FaceID + Liveness Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        face = faces[0]
        bbox = face.bbox.astype(int)

        # Проверка liveness
        is_real, liveness_msg = liveness_detector.verify(face)

        # Проверка в базе (только если liveness пройден)
        if is_real:
            max_similarity = 0
            best_match = None
            for name, saved_embedding in db.items():
                similarity = cosine_similarity([face.embedding], [saved_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = name

            if best_match and max_similarity >= THRESHOLD:
                result_text = f"Access: {best_match} ({max_similarity:.2f})"
                color = (0, 255, 0)
            else:
                result_text = f"Unknown (max: {max_similarity:.2f})"
                color = (0, 0, 255)
        else:
            result_text = f"Liveness: {liveness_msg}"
            color = (255, 255, 0)  # Жёлтый — ожидание действия

        # Отрисовка
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, result_text, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, liveness_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("FaceID + Liveness Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()