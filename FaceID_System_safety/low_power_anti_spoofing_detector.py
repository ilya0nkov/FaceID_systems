import cv2
import time


class LowPowerAntiSpoofingDetector:
    def __init__(self):
        self.last_processed_time = 0
        self.processing_interval = 1.0  # Уменьшен интервал проверки
        self.consecutive_frames = 0
        self.required_consecutive_frames = 2  # Уменьшено количество требуемых подтверждений
        self.failed_attempts = 0

    def enhanced_spoof_check(self, face_roi):
        if face_roi.size == 0:
            return False

        # Улучшенные параметры проверки
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Проверка размытия с адаптивным порогом
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 50 + min(self.failed_attempts * 5, 30)  # Адаптивный порог

        # Проверка цветовых каналов
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()

        # Проверка текстур
        std_dev = gray.std()

        # Более мягкие условия для реальных лиц
        if blur < blur_threshold and saturation < 40 and std_dev < 25:
            return False
        return True

    def verify(self, frame, face):
        current_time = time.time()
        if current_time - self.last_processed_time < self.processing_interval:
            return None, "Checking...", "Please wait"

        face_roi = frame[int(face.bbox[1]):int(face.bbox[3]),
                   int(face.bbox[0]):int(face.bbox[2])]

        is_real = self.enhanced_spoof_check(face_roi)

        if is_real:
            self.consecutive_frames += 1
            if self.consecutive_frames >= self.required_consecutive_frames:
                self.consecutive_frames = 0
                self.failed_attempts = max(0, self.failed_attempts - 1)
                self.last_processed_time = current_time
                return True, "Real face confirmed", "Now perform liveness check"
            return None, "Verifying authenticity", "Please hold still"
        else:
            self.consecutive_frames = 0
            self.failed_attempts += 1
            self.last_processed_time = current_time
            recommendation = "Try better lighting"
            if self.failed_attempts > 2:
                recommendation = "Ensure good lighting and remove glasses if worn"
            return False, "Spoofing detected", recommendation