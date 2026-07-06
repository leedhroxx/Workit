"""
Workit - JoRAG 검색 모듈 (최종 확정판)
파일명: rag/yoonha_law_rag.py

법령 지식베이스(law_kb_jo_fixedid)에서 텍스트 질의 하나에 대해 관련 조문을
찾아주는 순수 검색 모듈이다. 계약서 조항은 호출하는 쪽에서 이미 하나씩
분리해서 넘겨준다고 가정한다 — 이 모듈은 그 텍스트를 어디서 어떻게
쪼갰는지 몰라도 되고, 질의 텍스트 하나를 받아 검색만 한다.

검증된 최종 하이퍼파라미터 (gold_standard_v4 100개 기준, MRR 0.8837):
    alpha=0.3, rrf_k=20, fetch_k=80, rerank_k=20, reranker=bge-reranker-v2-m3, top_k=3

필요 패키지 (버전 꼭 맞출 것):
    FlagEmbedding==1.3.2, transformers==4.44.2
    이 두 버전 조합 밖에서는 임베더/리랭커 중 하나가 깨진다 — FlagEmbedding의
    리랭커 코드는 구식 tokenizer API(prepare_for_model)를 쓰고, 임베더 코드는
    최신 transformers 전용 인자(dtype)를 요구하는 등 내부적으로 버전 가정이
    엇갈려 있어서, 위 조합 밖에서는 둘 중 하나가 항상 깨진다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector


# ── 상수 ──────────────────────────────────────────────────────────

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_JO = "law_kb_jo_fixedid"

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# fp16은 GPU 전용 최적화라 CPU 환경에서는 자동으로 꺼진다 (켜두면 에러가 나거나
# 오히려 느려질 수 있음).
USE_FP16 = torch.cuda.is_available()

DEFAULT_ALPHA = 0.3
DEFAULT_RRF_K = 20
DEFAULT_FETCH_K = 80
DEFAULT_RERANK_K = 20
DEFAULT_TOP_K = 3


# ── 데이터 모델 ────────────────────────────────────────────────────

@dataclass
class LawRef:
    """검색된 법령 조문 하나."""

    chunk_id: str
    law_name: str
    article_id: str
    text: str
    score: float


# ── 모델 로더 ─────────────────────────────────────────────────────

def get_qdrant_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def load_embed_model() -> BGEM3FlagModel:
    return BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=USE_FP16)


def load_reranker() -> FlagReranker:
    return FlagReranker(RERANKER_MODEL_NAME, use_fp16=USE_FP16)


# ── 검색 파이프라인 ────────────────────────────────────────────────

def _hybrid_search(
    query_text: str,
    client: QdrantClient,
    model: BGEM3FlagModel,
    fetch_k: int,
    alpha: float,
    rrf_k: int,
) -> list[dict]:
    """dense(의미 유사도) + sparse(키워드 매칭)를 RRF로 결합해 상위 fetch_k개를 가져온다."""
    vectors = model.encode([query_text], return_dense=True, return_sparse=True)
    dense_vec = vectors["dense_vecs"][0]
    sparse_weights = vectors["lexical_weights"][0]
    sparse_vec = SparseVector(
        indices=[int(k) for k in sparse_weights.keys()],
        values=[float(v) for v in sparse_weights.values()],
    )

    results = client.query_points(
        collection_name=COLLECTION_JO,
        prefetch=[
            Prefetch(query=dense_vec.tolist(), using="dense", limit=fetch_k),
            Prefetch(query=sparse_vec, using="sparse", limit=fetch_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_k,
        with_payload=True,
    )

    return [
        {
            "chunk_id": point.payload["chunk_id"],
            "law_name": point.payload["law_name"],
            "article_id": point.payload["article_id"],
            "text": point.payload["text"],
            "rrf_score": point.score,
        }
        for point in results.points
    ]


def _rerank(
    query_text: str,
    candidates: list[dict],
    reranker: FlagReranker,
    rerank_k: int,
) -> list[dict]:
    """상위 rerank_k개만 쿼리와 1:1 비교해 재채점하고 점수 내림차순으로 정렬한다."""
    top_candidates = candidates[:rerank_k]
    pairs = [[query_text, c["text"]] for c in top_candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    for candidate, score in zip(top_candidates, scores):
        candidate["rerank_score"] = score

    return sorted(top_candidates, key=lambda c: c["rerank_score"], reverse=True)


def _build_law_refs(candidates: list[dict], top_k: int, use_reranker: bool) -> list[LawRef]:
    score_key = "rerank_score" if use_reranker else "rrf_score"
    return [
        LawRef(
            chunk_id=c["chunk_id"],
            law_name=c["law_name"],
            article_id=c["article_id"],
            text=c["text"],
            score=c[score_key],
        )
        for c in candidates[:top_k]
    ]


# ── 공개 API ──────────────────────────────────────────────────────

def search_jo(
    query_text: str,
    client: QdrantClient,
    model: BGEM3FlagModel,
    reranker: FlagReranker | None = None,
    use_reranker: bool = True,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    fetch_k: int = DEFAULT_FETCH_K,
    rerank_k: int = DEFAULT_RERANK_K,
) -> list[LawRef]:
    """
    임의의 텍스트 질의에 대해 관련 법령 조문을 검색한다. 이 모듈의 유일한 진입점.
    흐름: 하이브리드 검색(fetch_k개) → [옵션] 리랭크(rerank_k개 중) → 상위 top_k개 반환.
    """
    candidates = _hybrid_search(query_text, client, model, fetch_k, alpha, rrf_k)

    if use_reranker:
        if reranker is None:
            raise ValueError("use_reranker=True인데 reranker가 전달되지 않았습니다.")
        candidates = _rerank(query_text, candidates, reranker, rerank_k)

    return _build_law_refs(candidates, top_k, use_reranker)


# ── 실행 진입점 ────────────────────────────────────────────────────
# 호출 예시: 조항 텍스트 하나를 그대로 넘기면 관련 조문을 찾아준다.

def main() -> None:
    sample_clause = "제7조(지체상금) 계약상대자가 준공기한 내에 계약을 이행하지 못한 경우 지체일수 1일당 계약금액의 1천분의 1에 해당하는 금액을 지체상금으로 징수한다."

    client = get_qdrant_client()
    model = load_embed_model()
    reranker = load_reranker()

    law_refs = search_jo(sample_clause, client, model, reranker)

    print(f"질의(계약서 조항): {sample_clause}")
    for ref in law_refs:
        print(f"  - {ref.chunk_id} ({ref.law_name} {ref.article_id}) score={ref.score:.4f}")


if __name__ == "__main__":
    main()