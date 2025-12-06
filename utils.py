import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import requests

def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
