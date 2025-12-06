import cv2
import numpy as np
import onnxruntime as ort
from utils import download_image
from numpy.linalg import norm

# -----------------------------
# Load ONNX models
# -----------------------------

# MobileFaceNet ONNX (embedding)
embed_sess = ort.InferenceSession("models/mobilefacenet.onnx", providers=["CPUExecutionProvider"])

# SCRFD lightweight detector (face detection)
det_sess = ort.InferenceSession("models/scrfd_2.5g.onnx", providers=["CPUExecutionProvider"])


def detect_face(img):
    blob = cv2.resize(img, (640, 640))
    blob = blob.transpose(2, 0, 1)[None].astype(np.float32)

    outputs = det_sess.run(None, {"data": blob})
    bboxes = outputs[0]

    if len(bboxes) == 0:
        return None

    x1, y1, x2, y2, score = bboxes[0]
    if score < 0.5:
        return None

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    return img[y1:y2, x1:x2]


def get_embedding(face_img):
    if face_img is None:
        return None

    face = cv2.resize(face_img, (112, 112))
    face = face[:, :, ::-1]  # BGR → RGB
    face = face.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    emb = embed_sess.run(None, {"input": face})[0]
    emb = emb[0]
    emb = emb / (norm(emb) + 1e-10)
    return emb.tolist()


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
        face = detect_face(a)
        emb = get_embedding(face)

        if emb is not None:
            embeddings.append(emb)

    return embeddings
