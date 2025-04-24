import cv2
import time
import numpy as np
from utils.ai_models import face_app, THRESHOLD
from utils.db_utils import load_db
from low_power_liveness_detector import LowPowerLivenessDetector
from low_power_anti_spoofing_detector import LowPowerAntiSpoofingDetector
import warnings
import hashlib
import hmac
import base64
from common_data.lake_city import lake

warnings.filterwarnings("ignore", category=FutureWarning)

SECURITY_SALT = lake


class FaceAuthSystem:
    def __init__(self):
        # Проверяем доступность GUI
        self.gui_enabled = self.check_gui_support()
        self.last_console_output_time = 0
        self.console_output_interval = 5

        # Настройки интерфейса
        self.COLORS = {
            "error": (0, 0, 255),
            "success": (0, 255, 0),
            "warning": (0, 255, 255),
            "info": (255, 255, 0)
        }

        # Инициализация компонентов
        self.face_app = face_app
        self.db = load_db()
        self.anti_spoofing = LowPowerAntiSpoofingDetector()
        self.liveness = LowPowerLivenessDetector()

        # Инициализация камеры
        self.cap = self.init_camera()
        self.current_stage = "anti_spoofing"
        self.last_status_message = ""
        self.last_recommendation = ""
        self.last_status_type = "info"
        self.last_update = 0
        self.window_name = "FaceID System (DEBUG MODE)"
        self.last_face_detection_time = 0
        self.face_not_detected_shown = False
        self.last_successful_recognition_time = 0
        self.failed_attempts = 0
        self.max_failed_attempts = 20
        self.lockout_time = 0

        # Настройки окна
        self.window_width = 800
        self.window_height = 600
        self.font_scale = 1.2
        self.font_thickness = 2

    def secure_embedding(self, embedding):
        """Преобразуем эмбеддинг в защищенную форму с HMAC"""
        # Нормализуем эмбеддинг
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)

        # Преобразуем в байты
        embedding_bytes = embedding.tobytes()

        # Создаем HMAC с солью (убедитесь, что SECURITY_SALT - bytes)
        secured = hmac.new(SECURITY_SALT, embedding_bytes, hashlib.sha256).digest()

        # Возвращаем как строку base64
        return base64.b64encode(secured).decode('ascii')

    def check_gui_support(self):
        """Проверяет доступность GUI функций"""
        try:
            test_window = cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test")
            return True
        except:
            print("GUI functions not available - running in console mode")
            return False

    def init_camera(self):
        """Инициализирует камеру"""
        for i in [0, 1, 2]:  # Проверяем несколько индексов камер
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
        raise RuntimeError("Could not open any camera")

    def show_message(self, status_message, recommendation, status_type):
        """Обновляет статус системы"""
        self.last_status_message = status_message
        self.last_recommendation = recommendation
        self.last_status_type = status_type
        self.last_update = time.time()

        print(f"[{status_type.upper()}] {status_message} | {recommendation}")

    def draw_interface(self, frame):
        """Отрисовывает интерфейс с техническими деталями"""
        if frame is None:
            frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        scale_factor = min(self.window_width / w, self.window_height / h)
        resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        current_time = time.time()

        # Блокировка системы
        if self.lockout_time > current_time:
            remaining_time = int(self.lockout_time - current_time)
            lockout_msg = f"System locked (try again in {remaining_time}s)"
            cv2.putText(resized_frame, lockout_msg, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        self.COLORS["error"], self.font_thickness)
            return resized_frame

        # Статус авторизации
        if current_time - self.last_successful_recognition_time < 60:
            time_left = 60 - (current_time - self.last_successful_recognition_time)
            status_text = f"Authorized ({max(0, int(time_left))}s remaining)"
            cv2.putText(resized_frame, status_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        self.COLORS["success"], self.font_thickness)
        else:
            if time.time() - self.last_update < 3.0:
                color = self.COLORS.get(self.last_status_type, (255, 255, 255))
                cv2.putText(resized_frame, self.last_status_message, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.font_thickness)
                cv2.putText(resized_frame, self.last_recommendation, (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, (255, 255, 0), self.font_thickness)

        # Техническая информация (для отладки)
        cv2.putText(resized_frame, f"Stage: {self.current_stage}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(resized_frame, f"Fails: {self.failed_attempts}/{self.max_failed_attempts}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(resized_frame, f"DB size: {len(self.db)}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(resized_frame, f"FPS: {self.get_fps():.1f}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        return resized_frame

    def get_fps(self):
        """Возвращает примерный FPS"""
        if not hasattr(self, 'frame_times'):
            self.frame_times = []

        self.frame_times.append(time.time())
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)

        if len(self.frame_times) > 1:
            return len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        return 0

    def process_frame(self, frame):
        """Обрабатывает кадр"""
        current_time = time.time()

        if self.lockout_time > current_time:
            return frame

        if current_time - self.last_successful_recognition_time < 60:
            return frame

        if current_time - self.last_face_detection_time < 0.5:
            return frame

        faces = self.face_app.get(frame)
        self.last_face_detection_time = current_time

        if not faces:
            if not self.face_not_detected_shown or current_time - self.last_face_detection_time > 3:
                self.show_message(
                    "Face not detected",
                    "Please position your face in the frame",
                    "error"
                )
                self.face_not_detected_shown = True
            return frame

        self.face_not_detected_shown = False
        face = faces[0]
        bbox = face.bbox.astype(int)

        # Отрисовка рамки
        border_color = self.COLORS.get(self.last_status_type, (255, 255, 255))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), border_color, 3)

        if self.current_stage == "anti_spoofing":
            result, status_msg, recommendation = self.anti_spoofing.verify(frame, face)
            if result:
                self.current_stage = "liveness"
                self.show_message(
                    "Anti-spoofing passed",
                    "Now perform liveness check",
                    "success"
                )
            elif result is False:
                self.failed_attempts += 1
                self.check_lockout()
                self.show_message(
                    "Anti-spoofing failed",
                    "Please ensure you're using real face",
                    "error"
                )
                self.current_stage = "anti_spoofing"
        elif self.current_stage == "liveness":
            result, status_msg, recommendation = self.liveness.verify(frame, face)
            if result:
                self.verify_identity(face)
                self.current_stage = "anti_spoofing"
            elif result is False:
                self.failed_attempts += 1
                self.check_lockout()
                self.show_message(
                    "Liveness check failed",
                    "Please move your head or blink",
                    "error"
                )
                self.current_stage = "anti_spoofing"

        return frame

    def check_lockout(self):
        """Проверяет необходимость блокировки"""
        if self.failed_attempts >= self.max_failed_attempts:
            self.lockout_time = time.time() + 300  # 5 минут блокировки
            self.failed_attempts = 0
            self.show_message(
                "Too many failed attempts",
                "System locked for 5 minutes",
                "error"
            )

    def verify_identity(self, face):
        """Проверяет личность"""
        if not self.db:
            self.show_message("Database empty", "Please add users to the database", "error")
            return

        best_match = None
        max_sim = 0

        for name, emb in self.db.items():
            # Для отладки используем прямое сравнение эмбеддингов
            sim = np.dot(face.embedding / np.linalg.norm(face.embedding),
                         np.array(emb) / np.linalg.norm(emb))
            if sim > max_sim:
                max_sim = sim
                best_match = name

        if best_match and max_sim >= THRESHOLD:
            self.last_successful_recognition_time = time.time()
            self.failed_attempts = 0
            self.show_message(
                f"Welcome {best_match}!",
                f"Score: {max_sim:.2f} (threshold: {THRESHOLD})",
                "success"
            )
        else:
            self.failed_attempts += 1
            self.check_lockout()
            self.show_message(
                "Authentication failed",
                f"Best match: {max_sim:.2f} (required {THRESHOLD})",
                "error"
            )

    def run(self):
        """Основной цикл"""
        self.show_message("System ready", "DEBUG MODE", "info")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.show_message("Camera error", "Please check your camera connection", "error")
                    time.sleep(2)
                    continue

                frame = cv2.flip(frame, 1)
                processed = self.process_frame(frame)

                if self.gui_enabled:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
                    final_frame = self.draw_interface(processed)
                    cv2.imshow(self.window_name, final_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.1)

        except Exception as e:
            print(f"System error: {str(e)}")
        finally:
            self.cap.release()
            if self.gui_enabled:
                cv2.destroyAllWindows()
            print("System shutdown")


if __name__ == "__main__":
    system = FaceAuthSystem()
    system.run()

