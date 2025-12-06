import requests
import numpy as np
from PIL import Image
from io import BytesIO

def download_image(url: str):
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)
