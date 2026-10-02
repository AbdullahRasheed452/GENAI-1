import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app import corruptions

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[2] / "models_onnx"))
MODEL_FILES = {
    "task1": "task1_universal_ae.onnx",
    "classifier": "task2_classifier.onnx",
    "salt_pepper": "task2_specialist_salt_pepper.onnx",
    "blur": "task2_specialist_blur.onnx",
    "occlusion": "task2_specialist_occlusion.onnx",
    "task3": "task3_soft_moe.onnx",
    "task4": "task4_generator.onnx",
}
BRANCHES = ("identity", "salt_pepper", "blur", "occlusion")
SEVERITIES = ("low", "medium", "high")
MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000

sessions = {}


@asynccontextmanager
async def lifespan(application):
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
    for key, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            sessions[key] = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            print(f"loaded {key} from {path}")
        else:
            print(f"MISSING model file: {path}")
    yield


app = FastAPI(title="Generative AI Assignment 1 API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def read_image(upload):
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "The file is not a valid image.")
    if img.format not in ("JPEG", "PNG", "WEBP"):
        raise HTTPException(415, "Only JPEG, PNG, and WebP images are accepted.")
    img = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8).copy()


def data_url(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def to_input(image):
    return (image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def to_image(tensor):
    return np.clip(np.rint(tensor[0].transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def run(key, feeds):
    session = sessions.get(key)
    if session is None:
        raise HTTPException(503, f"Model '{key}' is not loaded on the server.")
    started = time.perf_counter()
    outputs = session.run(None, feeds)
    return outputs, (time.perf_counter() - started) * 1000.0


def corrupt(image, kind, severity, seed):
    if kind == "none":
        return image, {"type": "none"}
    if kind not in corruptions.TEST_LEVELS:
        raise HTTPException(422, "corruption must be none, salt_pepper, blur, or occlusion.")
    if severity not in SEVERITIES:
        raise HTTPException(422, "severity must be low, medium, or high.")
    level = corruptions.TEST_LEVELS[kind][SEVERITIES.index(severity)]
    if kind == "salt_pepper":
        params = {"type": kind, "prob": level["prob"], "seed": int(seed)}
    elif kind == "blur":
        params = {"type": kind, **level}
    else:
        rng = np.random.default_rng(seed)
        rects = corruptions.sample_rects(rng, level["target_coverage"], level["count"], tolerance=0.005)
        params = {"type": kind, "rects": rects, "target_coverage": level["target_coverage"], "coverage": corruptions.coverage(rects)}
    return corruptions.apply(image, params), {**params, "severity": severity}


def quality(original, corrupted, restored, kind):
    if kind == "none":
        return {"psnr_input_vs_original": None, "psnr_restored_vs_original": None}
    return {
        "psnr_input_vs_original": round(psnr(corrupted, original), 2),
        "psnr_restored_vs_original": round(psnr(restored, original), 2),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": sorted(sessions),
        "models_missing": [k for k in MODEL_FILES if k not in sessions],
    }


@app.post("/api/universal")
async def universal(image: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(0)):
    original = await read_image(image)
    corrupted, settings = corrupt(original, corruption, severity, seed)
    outputs, ms = run("task1", {"input": to_input(corrupted)})
    restored = to_image(outputs[0])
    return {
        "task": "universal",
        "input": data_url(original),
        "corrupted": data_url(corrupted),
        "restored": data_url(restored),
        "settings": settings,
        "inference_ms": round(ms, 1),
        **quality(original, corrupted, restored, corruption),
    }


@app.post("/api/hard")
async def hard(image: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(0)):
    original = await read_image(image)
    corrupted, settings = corrupt(original, corruption, severity, seed)
    x = to_input(corrupted)
    outputs, classifier_ms = run("classifier", {"input": x})
    probs = outputs[0][0]
    index = int(np.argmax(probs))
    predicted = corruptions.CLASSES[index]
    if index == 0:
        restored, expert, expert_ms = corrupted, "identity bypass", 0.0
    else:
        outputs, expert_ms = run(predicted, {"input": x})
        restored, expert = to_image(outputs[0]), f"{predicted} specialist"
    return {
        "task": "hard_routing",
        "input": data_url(original),
        "corrupted": data_url(corrupted),
        "restored": data_url(restored),
        "settings": settings,
        "probabilities": {name: round(float(p), 4) for name, p in zip(corruptions.CLASSES, probs)},
        "predicted": predicted,
        "expert": expert,
        "classifier_ms": round(classifier_ms, 1),
        "expert_ms": round(expert_ms, 1),
        "inference_ms": round(classifier_ms + expert_ms, 1),
        **quality(original, corrupted, restored, corruption),
    }


@app.post("/api/soft")
async def soft(image: UploadFile = File(...), corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(0)):
    original = await read_image(image)
    corrupted, settings = corrupt(original, corruption, severity, seed)
    outputs, ms = run("task3", {"input": to_input(corrupted)})
    restored = to_image(outputs[0])
    weights = outputs[1][0]
    return {
        "task": "soft_moe",
        "input": data_url(original),
        "corrupted": data_url(corrupted),
        "restored": data_url(restored),
        "settings": settings,
        "weights": {name: round(float(w), 4) for name, w in zip(BRANCHES, weights)},
        "dominant": BRANCHES[int(np.argmax(weights))],
        "inference_ms": round(ms, 1),
        **quality(original, corrupted, restored, corruption),
    }


@app.post("/api/sketch")
async def sketch(image: UploadFile = File(...), style: int = Form(1)):
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2, or 3.")
    photo = await read_image(image)
    tensor = to_input(photo) * 2.0 - 1.0
    outputs, ms = run("task4", {"photo": tensor.astype(np.float32), "style": np.array([style - 1], dtype=np.int64)})
    sketch_image = np.clip(np.rint((outputs[0][0].transpose(1, 2, 0) + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return {
        "task": "face_to_sketch",
        "photo": data_url(photo),
        "sketch": data_url(sketch_image),
        "style": style,
        "inference_ms": round(ms, 1),
    }
