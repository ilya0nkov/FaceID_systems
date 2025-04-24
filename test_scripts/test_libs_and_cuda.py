import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis


print(np.__version__)  # Должно быть <2.0
print(onnxruntime.get_device())  # Должно быть 'GPU'
app = FaceAnalysis(providers=['CUDAExecutionProvider'])

'''
bash:
nvcc --version  # Должно быть 11.8
where cudnn64_8.dll  # Должен показать путь к CUDA 11.8
'''
