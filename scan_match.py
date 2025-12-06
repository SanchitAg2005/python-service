import numpy as np
import cv2
import onnxruntime as ort
from utils import download_image
from numpy.linalg import norm

# -----------------------------
# Load models
# -----------------------------
embed_sess = ort.InferenceSession("models/mobilefacenet.onnx", providers=["CPUExecutionProvider"])
det_sess = ort.InferenceSession("models/scrfd_2.5g.onnx", providers=["CPUExecutionProvider"])

# Threshold rules (unchanged)
STRICT = 0.40
LOOSE  = 0.46
DELTA  = 0.03


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
    face = face[:, :, ::-1]
    face = face.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    emb = embed_sess.run(None, {"input": face})[0]
    emb = emb[0]
    emb = emb / (norm(emb) + 1e-10)
    return emb


def scan_and_match(image_urls, friend_embeddings):
    results = {name: [] for name in friend_embeddings.keys()}

    for img_url in image_urls:
        img = download_image(img_url)

        face = detect_face(img)
        f_emb = get_embedding(face)
        if f_emb is None:
            continue

        f_norm = norm(f_emb)

        best_person = None
        best_dist = 99
        second = 99

        for person, embs in friend_embeddings.items():
            arr = np.array(embs)
            norms = norm(arr, axis=1)

            sims = np.dot(arr, f_emb) / (norms * f_norm + 1e-12)
            dists = 1 - sims

            m = float(np.min(dists))

            if m < best_dist:
                second = best_dist
                best_dist = m
                best_person = person
            elif m < second:
                second = m

        if best_dist <= STRICT:
            results[best_person].append(img_url)
        elif best_dist <= LOOSE and (second - best_dist) >= DELTA:
            results[best_person].append(img_url)

    return results
