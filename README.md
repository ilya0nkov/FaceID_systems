FaceID_System_LowPower for using with embeddings in json format.
FaceID_System_safety for for encrypted embeddings.

Create a "data" folder with folders with user photos in it, then run registration.py.
For safety mode, use db_utils_safety instead of db_utils.

You can adjust the number of photos for registration by changing REGISTRATION_PHOTOS in registration.py.
You can choose between CPU and CUDA if your device allows it. To do this, in the utils/ai_models file, replace providers=["CPUExecutionProvider"] on providers=["CUDAExecutionProvider"]
