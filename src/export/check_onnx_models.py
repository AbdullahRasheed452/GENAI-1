import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models_onnx"
NAMES = [
    "task1_universal_ae", "task2_classifier", "task2_specialist_salt_pepper",
    "task2_specialist_blur", "task2_specialist_occlusion", "task3_soft_moe", "task4_generator",
]

photo = np.random.rand(1, 3, 128, 128).astype(np.float32)
for name in NAMES:
    path = MODELS / f"{name}.onnx"
    if not path.exists():
        print(f"{name}: MISSING")
        continue
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {"input": photo}
    if name == "task4_generator":
        feeds = {"photo": photo * 2 - 1, "style": np.array([1], dtype=np.int64)}
    started = time.time()
    outputs = session.run(None, feeds)
    ms = (time.time() - started) * 1000
    print(f"{name}: OK, outputs {[o.shape for o in outputs]}, {ms:.0f} ms")
