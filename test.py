import tensorflow as tf
import numpy as np
from PIL import Image
import time
from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).parent

model_path = BASE_DIR / "saved_model5"

model = tf.saved_model.load(str(model_path))

infer = model.signatures["serving_default"]

class_names = [
    "aksesoris cold",
    "aksesoris hot",
    "arcing horn",
    "bracing",
    "isolator",
    "jumper",
    "pondasi"
]

image_path = input("Masukkan path gambar: ")

image = Image.open(image_path).convert("RGB")
# image = Image.open(
#     "/Users/diajeng/Documents/image-classification-newsrintami/demo/AJAX_PDKBSLG016833d7ae31fcf.JPG"
# ).convert("RGB")

image = image.resize((256, 256))

img_array = np.array(image)

# img_array = img_array / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
).astype(np.float32)

start_time = time.time()

prediction = infer(
    tf.constant(img_array)
)

prediction = list(
    prediction.values()
)[0].numpy()

prediction = prediction[0]

# print("Hasil Prediksi:", prediction)

for i in range(len(class_names)):
    print(f"{class_names[i]}: {prediction[i]:.4f}")

predicted_class = class_names[
    np.argmax(prediction)
]

confidence = float(
    np.max(prediction)
)

end_time = time.time()

processing_time = end_time - start_time

print("Classification :", predicted_class)

print("Confidence :", confidence)
print("Processing Time: {:.4f} seconds".format(processing_time))