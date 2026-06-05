import pandas as pd
from PIL import Image
import io
import os

df = pd.read_parquet("/Users/diajeng/Documents/newsrintami/validation-00000-of-00001-02016d89fba12263.parquet")

os.makedirs("images", exist_ok=True)

for idx, row in df.iterrows():

    image_dict = row["image"]

    image_bytes = image_dict["bytes"]

    image = Image.open(io.BytesIO(image_bytes))

    image.save(f"images/{row['image_id']}.jpg")

print(f"Berhasil menyimpan {len(df)} gambar")