import os
import sys

sys.path.insert(0, "/workspace/workit-runpod/rag")

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel

from law_rag_pipeline import load_embed_model, load_reranker

app = FastAPI()

print("[startup] loading embed model (BAAI/bge-m3)...", flush=True)
embed_model = load_embed_model()
print("[startup] loading reranker (bge-reranker-v2-m3)...", flush=True)
reranker = load_reranker()
print("[startup] embed+reranker server ready.", flush=True)

API_KEY = os.environ.get("LLM_API_KEY", "")


def verify_key(authorization: str = Header(None)):
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


class EmbedRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    query: str
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "message": "embed+rerank server is running"}


@app.post("/embed", dependencies=[Depends(verify_key)])
def embed(req: EmbedRequest):
    vectors = embed_model.encode(req.texts, return_dense=True, return_sparse=True)
    dense = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors["dense_vecs"]]
    sparse = [{str(k): float(v) for k, v in lw.items()} for lw in vectors["lexical_weights"]]
    return {"dense_vecs": dense, "lexical_weights": sparse}


@app.post("/rerank", dependencies=[Depends(verify_key)])
def rerank(req: RerankRequest):
    pairs = [[req.query, t] for t in req.texts]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]
    return {"scores": scores}
