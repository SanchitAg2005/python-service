import insightface
import numpy as np
from PIL import Image
import requests
from io import BytesIO

model = None

def get_model():
    global model
    if model is None:
        model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0)
    return model

def load_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)
