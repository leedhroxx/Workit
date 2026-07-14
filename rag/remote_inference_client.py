# -*- coding: utf-8 -*-
"""
remote_inference_client.py — RunPod 추론 서버(FastAPI) 호출용 로컬 프록시.

CPU 로컬 환경에서 BGE-M3 / bge-reranker-v2-m3 / kanana-1.5-8b를 직접 로드하는 대신,
RunPod에 띄운 추론 서버(main.py, /embed·/rerank·/predict)에 HTTP로 위임한다.

law_rag_pipeline.search_jo()는 model.encode()/reranker.compute_score() 인터페이스만
알면 되므로, 아래 두 클래스는 그 인터페이스를 그대로 흉내 낸다 — search_jo() 쪽
코드는 한 줄도 바꿀 필요가 없다.
"""
import os
import time
import numpy as np
import requests

def _auth_headers():
    # 주의: 이름에 "RUNPOD_" 접두사를 쓰면 안 됨 — RunPod이 그 접두사의 환경변수를
    # 플랫폼 자체 용도로 예약해서 pod 안에서 자동으로 다른 값이 주입돼버린다.
    key = os.environ.get("LLM_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _post_with_logging(label: str, url: str, json_payload: dict, timeout: int) -> requests.Response:
    """RunPod 호출 공통 래퍼 — 성공/실패/응답시간을 로그로 남긴다(TC-NF-021).

    이 프로세스의 표준 출력은 gunicorn/celery 로그 파일로 잡혀서 CloudWatch Logs로 올라간다.
    """
    started = time.monotonic()
    try:
        resp = requests.post(url, json=json_payload, headers=_auth_headers(), timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f'[remote_inference_client] {label} 실패 ({elapsed:.2f}s) — {url}: {exc}')
        raise
    elapsed = time.monotonic() - started
    print(f'[remote_inference_client] {label} 성공 ({elapsed:.2f}s) — {url}')
    return resp

class RemoteEmbedModel:
    """BGEM3FlagModel.encode()와 같은 인터페이스로 /embed를 호출한다."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def encode(self, texts, return_dense: bool = True, return_sparse: bool = True):
        resp = _post_with_logging(
            "embed", f"{self.base_url}/embed", {"texts": list(texts)}, self.timeout
        )
        data = resp.json()
        return {
            "dense_vecs": np.array(data["dense_vecs"]),
            "lexical_weights": data["lexical_weights"],
        }


class RemoteReranker:
    """FlagReranker.compute_score()와 같은 인터페이스로 /rerank를 호출한다."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def compute_score(self, pairs, normalize: bool = True):
        if not pairs:
            return []
        query = pairs[0][0]
        texts = [p[1] for p in pairs]
        resp = _post_with_logging(
            "rerank", f"{self.base_url}/rerank", {"query": query, "texts": texts}, self.timeout
        )
        return resp.json()["scores"]


def remote_predict(item: dict, base_url: str, timeout: int = 120) -> dict:
    """jihye_inference.predict(item, model, tokenizer)와 같은 반환 형식으로 /predict를 호출한다.

    임베딩/리랭커 서버와 LLM 서버는 transformers 버전이 서로 호환되지 않아
    RunPod에서 별도 venv·별도 포트로 띄운다(embed_server:8000, llm_server:8002).
    그래서 base_url이 RemoteEmbedModel/RemoteReranker와 다를 수 있다.
    """
    resp = _post_with_logging(
        "predict", f"{base_url.rstrip('/')}/predict", {"item": item}, timeout
    )
    return resp.json()["prediction"]

def remote_compare_pep(item: dict, base_url: str, timeout: int = 180) -> dict:
    """RFP ↔ 사업수행계획서(PEP) 대응비교. jihye_inference.predict_pep()와 같은 반환 형식."""
    resp = _post_with_logging(
        "compare-pep", f"{base_url.rstrip('/')}/compare-pep", {"item": item}, timeout
    )
    return resp.json()["result"]


def remote_compare_rpt(item: dict, base_url: str, timeout: int = 180) -> dict:
    """사업수행계획서(PEP) ↔ 사업추진결과보고서(RPT) 대응비교. jihye_inference.predict_rpt()와 같은 반환 형식."""
    resp = _post_with_logging(
        "compare-rpt", f"{base_url.rstrip('/')}/compare-rpt", {"item": item}, timeout
    )
    return resp.json()["result"]
