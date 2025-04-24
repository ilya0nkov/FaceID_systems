import time
import numpy as np


class LowPowerLivenessDetector:
    def __init__(self):
        self.eye_blink_threshold = 0.30  # Увеличен порог для более легкого обнаружения
        self.prev_head_pos = None
        self.last_processed_time = 0
        self.processing_interval = 0.8  # Уменьшен интервал
        self.required_movement = 12  # Уменьшено требуемое движение
        self.consecutive_blinks = 0
        self.required_blinks = 1  # Требуется только 1 моргание
        self.last_blink_time = 0
        self.blink_cooldown = 0.8
        self.failed_attempts = 0

    def simple_eye_blink_check(self, eye_landmarks):
        vertical_dist = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5]) + \
                        np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        ear = vertical_dist / (2.0 * horizontal_dist)
        return ear < (self.eye_blink_threshold - min(self.failed_attempts * 0.02, 0.1))

    def verify(self, frame, face):
        current_time = time.time()
        if current_time - self.last_processed_time < self.processing_interval:
            return None, "Processing liveness", "Please wait"

        self.last_processed_time = current_time

        # Получаем landmarks для глаз
        left_eye = face.landmark_2d_106[36:42]
        right_eye = face.landmark_2d_106[42:48]

        # Проверка моргания
        left_blink = self.simple_eye_blink_check(left_eye)
        right_blink = self.simple_eye_blink_check(right_eye)
        blink_detected = left_blink or right_blink

        if blink_detected and (current_time - self.last_blink_time > self.blink_cooldown):
            self.consecutive_blinks += 1
            self.last_blink_time = current_time

        # Проверка движения головы
        nose_tip = face.landmark_2d_106[55]
        if self.prev_head_pos is None:
            self.prev_head_pos = nose_tip
            return False, "Initializing liveness check", "Please move your head slightly"

        movement = np.linalg.norm(nose_tip - self.prev_head_pos)
        self.prev_head_pos = nose_tip

        # Комбинированная проверка
        if self.consecutive_blinks >= self.required_blinks or movement > self.required_movement:
            self.consecutive_blinks = 0
            self.failed_attempts = max(0, self.failed_attempts - 1)
            return True, "Liveness confirmed", "Proceeding to authentication"

        # Генерация рекомендации
        self.failed_attempts += 1
        recommendation = "Please "
        if self.consecutive_blinks < self.required_blinks:
            recommendation += f"blink ({self.consecutive_blinks}/{self.required_blinks}) "
        if movement < self.required_movement:
            if self.consecutive_blinks < self.required_blinks:
                recommendation += "or "
            recommendation += "move your head"

        if self.failed_attempts > 2:
            recommendation += " more noticeably"

        return False, "Liveness verification", recommendation.strip()