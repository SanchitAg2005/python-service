from fastapi import FastAPI
from pydantic import BaseModel
from encode import generate_embeddings
from scan_match import scan_and_match
from pymongo import MongoClient
import os

app = FastAPI()

# --- MongoDB Setup ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

@app.get("/db-test")
def db_test():
    try:
        client.admin.command("ping")
        return {"mongo": "connected"}
    except Exception as e:
        return {"mongo": "failed", "error": str(e)}

# --- Request Models ---
class EncodeRequest(BaseModel):
    image_url: str

class ScanMatchRequest(BaseModel):
    image_urls: list
    friend_embeddings: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/encode")
def encode_route(req: EncodeRequest):
    embeddings = generate_embeddings(req.image_url)
    return {"embeddings": embeddings}

@app.post("/scan-match")
def scan_match_route(req: ScanMatchRequest):
    results = scan_and_match(req.image_urls, req.friend_embeddings)
    return {"results": results}
