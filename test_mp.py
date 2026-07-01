import sys
import mediapipe as mp

print("Python exe:", sys.executable)
print("Python version:", sys.version)
print("Mediapipe file:", mp.__file__)
print("Mediapipe version:", getattr(mp, "__version__", "no version"))
print("Has solutions:", hasattr(mp, "solutions"))
print("Dir sample:", dir(mp)[:20])