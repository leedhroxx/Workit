import os
import sys
import threading

sys.path.insert(0, "/workspace/workit-runpod/rag")

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel

from inference import (
    load_model as load_llm_model,
    predict as llm_predict,
    predict_pep as llm_predict_pep,
    predict_rpt as llm_predict_rpt,
)

app = FastAPI()

print("[startup] loading LLM (kanana-1.5-8b + LoRA)...", flush=True)
llm_model, tokenizer = load_llm_model()
print("[startup] LLM server ready.", flush=True)

# 모델 인스턴스 하나를 여러 요청이 동시에 generate()하면 CUDA 상태가 꼬여
# "device-side assert triggered"로 죽고 그 뒤 요청까지 전부 실패한다.
# FastAPI의 sync 엔드포인트는 스레드풀에서 동시 실행되므로 반드시 직렬화해야 한다.
# CON(/predict)과 PEP/RPT(/compare-*)는 같은 모델 인스턴스를 공유하므로 락도 공유한다.
_predict_lock = threading.Lock()

API_KEY = os.environ.get("LLM_API_KEY", "")


def verify_key(authorization: str = Header(None)):
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


class PredictRequest(BaseModel):
    item: dict


class CompareRequest(BaseModel):
    item: dict


@app.get("/health")
def health():
    return {"status": "ok", "message": "LLM server is running"}


@app.post("/predict", dependencies=[Depends(verify_key)])
def predict_endpoint(req: PredictRequest):
    with _predict_lock:
        result = llm_predict(req.item, llm_model, tokenizer)
    return {"prediction": result}


@app.post("/compare-pep", dependencies=[Depends(verify_key)])
def compare_pep_endpoint(req: CompareRequest):
    with _predict_lock:
        result = llm_predict_pep(req.item, llm_model, tokenizer)
    return {"result": result}


@app.post("/compare-rpt", dependencies=[Depends(verify_key)])
def compare_rpt_endpoint(req: CompareRequest):
    with _predict_lock:
        result = llm_predict_rpt(req.item, llm_model, tokenizer)
    return {"result": result}
