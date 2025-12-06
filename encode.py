import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils import download_image

# load model once
import os
os.environ["INSIGHTFACE_DISABLE_TRT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])

app.prepare(ctx_id=0, det_size=(640, 640))

def augment(img):
    h, w = img.shape[:2]
    aug = []

    aug.append(img)
    aug.append(cv2.flip(img, 1))

    M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1)
    aug.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))

    M = cv2.getRotationMatrix2D((w/2, h/2), -5, 1)
    aug.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE))

    aug.append(cv2.convertScaleAbs(img, alpha=1.0, beta=30))
    aug.append(cv2.convertScaleAbs(img, alpha=0.9, beta=-30))
    aug.append(cv2.convertScaleAbs(img, alpha=1.2, beta=0))
    aug.append(cv2.convertScaleAbs(img, alpha=0.9, beta=0))

    zx, zy = int(w*0.08), int(h*0.08)
    crop = img[zy:h-zy, zx:w-zx]
    crop = cv2.resize(crop, (w, h))
    aug.append(crop)

    aug.append(cv2.GaussianBlur(img, (5,5), 0))

    return aug

def generate_embeddings(image_url):
    img = download_image(image_url)
    aug_images = augment(img)

    embeddings = []
    for a in aug_images:
        faces = app.get(a)
        if len(faces) == 0:
            continue
        embeddings.append(faces[0].embedding.tolist())

    return embeddings
