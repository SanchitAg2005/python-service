from fastapi import FastAPI
from pydantic import BaseModel
from encode import generate_embeddings
from scan_match import scan_and_match

app = FastAPI()

class EncodeRequest(BaseModel):
    image_url: str

class ScanMatchRequest(BaseModel):
    image_urls: list
    friend_embeddings: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/encode")
def encode_face(req: EncodeRequest):
    emb = generate_embeddings(req.image_url)
    return {"embeddings": emb}

@app.post("/scan-match")
def scan_match(req: ScanMatchRequest):
    result = scan_and_match(req.image_urls, req.friend_embeddings)
    return {"results": result}
