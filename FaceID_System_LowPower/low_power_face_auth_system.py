import cv2
import time
import numpy as np
from utils.ai_models import face_app, THRESHOLD
from utils.db_utils import load_db
from low_power_liveness_detector import LowPowerLivenessDetector
from low_power_anti_spoofing_detector import LowPowerAntiSpoofingDetector
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


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
        self.last_message = ""
        self.last_status = "info"
        self.last_update = 0
        self.window_name = "FaceID System (Press Q to quit)"
        self.last_face_detection_time = 0
        self.face_not_detected_shown = False
        self.last_successful_recognition_time = 0  # Время последнего успешного распознавания

        # Настройки окна
        self.window_width = 800
        self.window_height = 600
        self.font_scale = 1.2
        self.font_thickness = 2

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
        """Инициализирует камеру с обработкой ошибок"""
        for i in [0, 1, 2]:  # Проверяем несколько индексов камер
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Используем DirectShow для Windows
            if cap.isOpened():
                # Устанавливаем максимально возможное разрешение
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
        raise RuntimeError("Could not open any camera")

    def show_message(self, status_message, recommendation, status_type):
        """Обновляет статус системы с разделением на статус и рекомендацию"""
        self.last_status_message = status_message
        self.last_recommendation = recommendation
        self.last_status_type = status_type
        self.last_update = time.time()

        # Выводим в консоль только если это не повторное сообщение об успешной авторизации
        current_time = time.time()
        if (status_type != "success" or
                current_time - self.last_console_output_time >= self.console_output_interval or
                "approved" not in status_message):
            print(f"[{status_type.upper()}] {status_message} | {recommendation}")
            if status_type == "success" and "approved" in status_message:
                self.last_console_output_time = current_time

    def draw_interface(self, frame):
        """Отрисовывает интерфейс с разделением статуса и рекомендации"""
        if frame is None:
            frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        # Масштабирование кадра
        h, w = frame.shape[:2]
        scale_factor = min(self.window_width / w, self.window_height / h)
        resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        current_time = time.time()
        time_left = 60 - (current_time - self.last_successful_recognition_time)

        # Если авторизация активна, показываем оставшееся время
        if current_time - self.last_successful_recognition_time < 60:
            status_text = f"Authorized ({max(0, int(time_left))}s remaining)"
            cv2.putText(resized_frame, status_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        self.COLORS["success"], self.font_thickness)
            # Обновляем консоль раз в 5 секунд
            if current_time - self.last_console_output_time >= self.console_output_interval:
                print(f"[STATUS] Authorization active - {max(0, int(time_left))} seconds remaining")
                self.last_console_output_time = current_time
        else:
            # Обычный вывод сообщений
            if time.time() - self.last_update < 3.0:
                color = self.COLORS.get(self.last_status_type, (255, 255, 255))

                # Статус (первая строка)
                cv2.putText(resized_frame, self.last_status_message, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.font_thickness)

                # Рекомендация (вторая строка)
                cv2.putText(resized_frame, self.last_recommendation, (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, (255, 255, 0), self.font_thickness)
        '''
        # Отображение текущего этапа (третья строка)
        cv2.putText(resized_frame, f"Stage: {self.current_stage}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7, (255, 0, 0), self.font_thickness)
        '''
        return resized_frame

    def process_frame(self, frame):
        """Обрабатывает кадр с лицом"""
        faces = self.face_app.get(frame)
        current_time = time.time()

        # Проверка, прошло ли меньше минуты с момента последнего успешного распознавания
        if current_time - self.last_successful_recognition_time < 60:
            return frame

        if not faces:
            if not self.face_not_detected_shown or current_time - self.last_face_detection_time > 3:
                self.show_message(
                    "Face not detected",
                    "Please position your face in the frame",
                    "error"
                )
                self.face_not_detected_shown = True
                self.last_face_detection_time = current_time
            return frame

        self.face_not_detected_shown = False
        face = faces[0]
        bbox = face.bbox.astype(int)

        # Отрисовка рамки
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      self.COLORS.get(self.last_status_type, (255, 255, 255)), 3)

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
                self.show_message(
                    "Liveness check failed",
                    "Please move your head or blink",
                    "error"
                )
                self.current_stage = "anti_spoofing"

        return frame

    def verify_identity(self, face):
        """Проверяет личность"""
        if not self.db:
            self.show_message("Database empty", "Please add users to the database", "error")
            return

        best_match = None
        max_sim = 0

        for name, emb in self.db.items():
            sim = np.dot(face.embedding / np.linalg.norm(face.embedding),
                         np.array(emb) / np.linalg.norm(emb))
            if sim > max_sim:
                max_sim = sim
                best_match = name

        if best_match and max_sim >= THRESHOLD:
            self.last_successful_recognition_time = time.time()
            self.show_message(f"Welcome {best_match}!", f"Authorized for 60 seconds (Score: {max_sim:.2f})", "success")
        else:
            self.show_message("Authentication failed", f"Best match score: {max_sim:.2f} (required {THRESHOLD})",
                              "error")

    def run(self):
        """Основной цикл с GUI"""
        self.show_message("System ready", "", "info")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.show_message("Camera error", "Please check your camera connection", "error")
                    break

                frame = cv2.flip(frame, 1)
                processed = self.process_frame(frame)

                if self.gui_enabled:
                    # Создаем окно с возможностью изменения размера
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

                    # Отрисовываем интерфейс и отображаем
                    final_frame = self.draw_interface(processed)
                    cv2.imshow(self.window_name, final_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.1)

        finally:
            self.cap.release()
            if self.gui_enabled:
                cv2.destroyAllWindows()
            print("System shutdown")


if __name__ == "__main__":
    system = FaceAuthSystem()
    system.run()
