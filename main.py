import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from pathlib import Path
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

@app.get("/")
def home():
    return {
        "message": "API jalan"
    }

@app.post("/classify")
async def classify(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

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

    # print("jumlah output model:", len(prediction))

    # print("jumlah class_names:", len(class_names)) 

    # for i in range(len(class_names)):
    #     print(f"{class_names[i]}: {prediction[i]:.4f}")

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)
    )

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }