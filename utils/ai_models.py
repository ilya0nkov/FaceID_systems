from insightface.app import FaceAnalysis


THRESHOLD = 0.6  # Порог схожести

# Инициализация модели
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
# face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_app.prepare(ctx_id=0)
print("модель загружена")