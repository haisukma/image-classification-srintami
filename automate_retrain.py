from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path

import tensorflow as tf
import numpy as np
import shutil
import uuid
import os

from PIL import Image

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = FastAPI()

BASE_DIR = Path(__file__).parent

model_path = BASE_DIR / "saved_model"

model = tf.saved_model.load(str(model_path))

infer = model.signatures["serving_default"]

class_names = [
    "aksesoris cold",
    "aksesoris hot",
    "arcing horn",
    "bracing",
    "isolator",
    "jumper",
    "other",
    "pondasi"
]


UPLOAD_DIR = BASE_DIR / "uploaded_images"
NEW_DATA_DIR = BASE_DIR / "new_data"

UPLOAD_DIR.mkdir(exist_ok=True)
NEW_DATA_DIR.mkdir(exist_ok=True)

class FeedbackRequest(BaseModel):
    image_id: str
    correct_label: str

@app.get("/")
def home():
    return {
        "message": "API jalan"
    }

@app.post("/classify")
async def classify(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    image_id = str(uuid.uuid4())

    image_path = UPLOAD_DIR / f"{image_id}.jpg"

    image.save(image_path)

    image = image.resize((256, 256))

    img_array = np.array(image)

    img_array = np.expand_dims(
        img_array,
        axis=0
    ).astype(np.float32)

    prediction = infer(
        tf.constant(img_array)
    )

    prediction = list(
        prediction.values()
    )[0].numpy()

    prediction = prediction[0]

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)
    )

    return {
        "image_id": image_id,
        "prediction": predicted_class,
        "confidence": confidence
    }

@app.post("/feedback")
async def feedback(data: FeedbackRequest):

    if data.correct_label not in class_names:
        return {
            "status": "error",
            "message": "Label tidak valid"
        }

    source_file = UPLOAD_DIR / f"{data.image_id}.jpg"

    if not source_file.exists():
        return {
            "status": "error",
            "message": "Gambar tidak ditemukan"
        }

    target_dir = NEW_DATA_DIR / data.correct_label

    target_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    target_file = target_dir / f"{data.image_id}.jpg"

    shutil.move(
        str(source_file),
        str(target_file)
    )

    return {
        "status": "success",
        "message": "Feedback berhasil disimpan"
    }