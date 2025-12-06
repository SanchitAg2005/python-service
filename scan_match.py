import numpy as np
from utils import download_image
from insightface.app import FaceAnalysis

# Load ArcFace model once
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Threshold rules
STRICT = 0.40
LOOSE  = 0.46
DELTA  = 0.03

def scan_and_match(image_urls, friend_embeddings):
    results = {name: [] for name in friend_embeddings.keys()}

    for img_url in image_urls:
        img = download_image(img_url)
        faces = app.get(img)

        for face in faces:
            f_emb = face.embedding
            f_norm = np.linalg.norm(f_emb)

            best_person = None
            best_dist = 99
            second = 99

            for person, embs in friend_embeddings.items():
                arr = np.array(embs)
                norms = np.linalg.norm(arr, axis=1)

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
