import requests
import cv2
import numpy as np

def download_image(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error downloading image:", e)
        return None
