"""
Workit - 계약서 검토 RAG 파이프라인
파일명: yoonha_contract_rag.py
위치:   Workit/rag/yoonha_contract_rag.py

변경사항 (2026-06-25):
  FlagReranker → CrossEncoderReranker (transformers 직접 사용)
  이유: FlagReranker.compute_score()가 최신 transformers에서
        BertTokenizer.prepare_for_model() 제거로 AttributeError 발생.
        transformers AutoModel 기반으로 교체해 호환성 문제 우회.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_THIS_DIR     = Path(__file__).resolve().parent
_DATA_DIR     = _THIS_DIR.parent / "data"
LAWS_REF_PATH = _DATA_DIR / "hn_seed" / "law_refs.json"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_HO = "law_kb_ho"
COLLECTION_JO = "law_kb_jo"

EMBED_MODEL = "BAAI/bge-m3"
RRF_ALPHA   = 1.0

FETCH_K   = 50
RERANK1_K = 12
RERANK2_K = 7
TOP_K     = 10

RERANKER1_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER2_MODEL = "BAAI/bge-reranker-v2-m3"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reranker 래퍼 (transformers 직접 사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CrossEncoderReranker:
    """
    transformers AutoModel 기반 Cross-encoder reranker.
    FlagReranker 대체용 — 외부 인터페이스는 동일하게 유지.

    FlagReranker가 내부적으로 tokenizer.prepare_for_model()을 호출하는데
    최신 transformers에서 BertTokenizer에 해당 메서드가 제거되어
    AttributeError가 발생함. tokenizer() 직접 호출 방식으로 우회.

    compute_score(pairs) → list[float]
      pairs: [[query, doc], [query, doc], ...]
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def compute_score(
        self,
        pairs     : list[list[str]],
        batch_size: int = 32,
        normalize : bool = True,
    ) -> list[float]:
        """
        쿼리-문서 쌍 리스트의 관련도 점수 반환.

        Args:
            pairs     : [[query, doc], ...]
            batch_size: 배치 크기 (CPU 메모리 절약)
            normalize : True면 sigmoid 적용 (0~1 범위)
        """
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            encoded = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits

            # 이진 분류: (batch, 1) 또는 (batch, 2)
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1)
            else:
                scores = logits[:, 1]

            if normalize:
                scores = torch.sigmoid(scores)

            all_scores.extend(scores.cpu().tolist())

        return all_scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 데이터 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LawRef:
    """검색된 법령 조문 1건."""
    chunk_id    : str
    article     : str
    category    : str
    law_name    : str
    chunk_text  : str
    score       : float
    is_risk_ref : bool
    parent_id   : str = ""


@dataclass
class ClauseResult:
    """계약서 조항(또는 항) 1건의 검색 결과"""
    clause_number : str
    clause_text   : str
    page          : int = 0
    bbox          : dict | None = None
    law_refs      : list[LawRef] = field(default_factory=list)
    categories    : list[str]   = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. laws_ref.json 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_laws_ref(path: Path = LAWS_REF_PATH) -> dict[str, dict]:
    if not path.exists():
        print(f"  ⚠️  laws_ref.json 없음: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model(model_name: str = EMBED_MODEL, use_fp16: bool = True) -> BGEM3FlagModel:
    print(f"📦 임베딩 모델 로드: {model_name}")
    return BGEM3FlagModel(model_name, use_fp16=use_fp16)


def load_rerankers(device: str = "cpu") -> tuple[CrossEncoderReranker, CrossEncoderReranker]:
    """
    2-stage Cross-encoder reranker 로드.
    GPU 있으면 device="cuda", 없으면 device="cpu".
    """
    print(f"📦 Re-ranker 1단계 로드: {RERANKER1_MODEL}")
    reranker1 = CrossEncoderReranker(RERANKER1_MODEL, device=device)
    print(f"📦 Re-ranker 2단계 로드: {RERANKER2_MODEL}")
    reranker2 = CrossEncoderReranker(RERANKER2_MODEL, device=device)
    return reranker1, reranker2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. BGE-M3 Dense + Sparse 벡터 추출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_vectors(
    text  : str,
    model : BGEM3FlagModel,
) -> tuple[list[float], dict[int, float]]:
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vector    = output["dense_vecs"][0].tolist()
    lexical_weights = output["lexical_weights"][0]

    sparse_vector: dict[int, float] = {}
    for token_str, weight in lexical_weights.items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int):
            sparse_vector[token_id] = sparse_vector.get(token_id, 0.0) + float(weight)

    return dense_vector, sparse_vector


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 계약서 조항+항 단위 청킹
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_contract(text: str) -> list[dict]:
    HANG_MAP = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}
    HO_SPLIT_PATTERN = r"(?:^|\s)(\d{1,2}\.\s)"

    text = text.strip()
    header_pattern = re.compile(r"제(\d+)조(?:의(\d+))?\s*\(([^)]*)\)")
    raw_matches = list(header_pattern.finditer(text))

    candidates = []
    for m in raw_matches:
        prefix = text[max(0, m.start() - 5):m.start()]
        if re.search(r"법\s*$", prefix):
            continue
        num = int(m.group(1))
        sub = m.group(2)
        clause_number = f"제{m.group(1)}조" + (f"의{sub}" if sub else "")
        candidates.append((num, clause_number, m.start()))

    header_spans = []
    last_num = 0
    for num, clause_number, start in candidates:
        if num >= last_num and num <= last_num + 5:
            header_spans.append((clause_number, start))
            last_num = num

    def split_into_ho(parent_number: str, unit_text: str) -> list[dict]:
        ho_splits = re.split(HO_SPLIT_PATTERN, unit_text)
        if len(ho_splits) <= 1:
            return [{"clause_number": parent_number, "clause_text": unit_text}]

        head = ho_splits[0].strip()
        chunks = []
        if head:
            chunks.append({"clause_number": parent_number, "clause_text": head})

        k = 1
        last_ho_num = 0
        while k < len(ho_splits) - 1:
            marker = ho_splits[k].strip()
            ho_num_match = re.match(r"(\d{1,2})\.", marker)
            ho_num = int(ho_num_match.group(1)) if ho_num_match else (k // 2 + 1)
            ho_body = ho_splits[k + 1].strip() if k + 1 < len(ho_splits) else ""

            if ho_num == last_ho_num + 1 and ho_body:
                chunks.append({
                    "clause_number": f"{parent_number}제{ho_num}호",
                    "clause_text":   re.sub(r"\s+", " ", f"{marker} {ho_body}").strip(),
                })
                last_ho_num = ho_num
            elif ho_body:
                if chunks:
                    chunks[-1]["clause_text"] += f" {marker} {ho_body}"
                else:
                    chunks.append({"clause_number": parent_number, "clause_text": f"{marker} {ho_body}"})
            k += 2

        return chunks if chunks else [{"clause_number": parent_number, "clause_text": unit_text}]

    clauses = []
    for idx, (clause_number, start) in enumerate(header_spans):
        end = header_spans[idx + 1][1] if idx + 1 < len(header_spans) else len(text)
        raw_block = text[start:end].strip()

        m = header_pattern.match(raw_block)
        raw_header = m.group(0) if m else clause_number
        body = raw_block[m.end():].strip() if m else raw_block

        if not body:
            continue

        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", body)

        if len(hang_splits) <= 1:
            clause_text = re.sub(r"\s+", " ", f"{raw_header} {body}").strip()
            clauses.extend(split_into_ho(clause_number, clause_text))
        else:
            j = 1
            while j < len(hang_splits) - 1:
                hang_char = hang_splits[j]
                hang_body = hang_splits[j + 1].strip() if j + 1 < len(hang_splits) else ""
                hang_num  = HANG_MAP.get(hang_char, j)
                if hang_body:
                    hang_number = f"{clause_number}제{hang_num}항"
                    hang_text   = re.sub(r"\s+", " ", f"{raw_header} {hang_char}{hang_body}").strip()
                    clauses.extend(split_into_ho(hang_number, hang_text))
                j += 2

    if not clauses:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [
            {"clause_number": f"단락{i + 1}", "clause_text": para}
            for i, para in enumerate(paragraphs)
        ]

    return clauses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Qdrant 하이브리드 검색 (Weighted RRF)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _qdrant_hybrid_search(
    clause_text : str,
    client      : QdrantClient,
    model       : BGEM3FlagModel,
    collection  : str,
    fetch_k     : int   = FETCH_K,
    alpha       : float = RRF_ALPHA,
) -> list[dict]:
    dense_vec, sparse_vec = get_vectors(clause_text, model)
    indices = list(sparse_vec.keys())
    values  = list(sparse_vec.values())
    RRF_K   = 60

    try:
        dense_results = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense",
            limit=fetch_k,
            with_payload=True,
        ).points

        sparse_results = client.query_points(
            collection_name=collection,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            limit=fetch_k,
            with_payload=True,
        ).points

    except Exception as e:
        print(f"  ⚠️  sparse 검색 실패, dense만 사용: {e}")
        dense_results = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense",
            limit=fetch_k,
            with_payload=True,
        ).points
        sparse_results = []

    scores: dict[str, dict] = {}

    for rank, point in enumerate(dense_results, 1):
        cid = point.payload.get("chunk_id", str(point.id))
        scores[cid] = {
            "payload":     point.payload,
            "dense_rank":  rank,
            "sparse_rank": len(dense_results) + 1,
        }

    for rank, point in enumerate(sparse_results, 1):
        cid = point.payload.get("chunk_id", str(point.id))
        if cid in scores:
            scores[cid]["sparse_rank"] = rank
        else:
            scores[cid] = {
                "payload":     point.payload,
                "dense_rank":  len(sparse_results) + 1,
                "sparse_rank": rank,
            }

    results = []
    for cid, info in scores.items():
        rrf_score = (
            alpha         * (1 / (RRF_K + info["dense_rank"]))
            + (1 - alpha) * (1 / (RRF_K + info["sparse_rank"]))
        )
        results.append({
            "chunk_id":  cid,
            "payload":   info["payload"],
            "rrf_score": rrf_score,
        })

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 2-stage Re-ranking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _rerank(
    query     : str,
    candidates: list[dict],
    reranker  : CrossEncoderReranker,
    top_k     : int,
) -> list[dict]:
    """CrossEncoderReranker로 후보 재순위 후 상위 top_k개 반환."""
    if not candidates:
        return []

    texts = [c["payload"].get("text", c["payload"].get("chunk_text", "")) for c in candidates]
    pairs = [[query, t] for t in texts]
    scores = reranker.compute_score(pairs, normalize=True)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    return [item for _, item in ranked[:top_k]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Hierarchical RAG — parent fetch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_parent_texts(
    candidates : list[dict],
    client     : QdrantClient,
) -> list[dict]:
    parent_ids = list({
        c["payload"].get("parent_id")
        for c in candidates
        if c["payload"].get("parent_id")
    })

    if not parent_ids:
        return candidates

    parent_texts: dict[str, str] = {}
    try:
        for parent_id in parent_ids:
            results = client.scroll(
                collection_name=COLLECTION_JO,
                scroll_filter={
                    "must": [
                        {"key": "chunk_id", "match": {"value": parent_id}}
                    ]
                },
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            if points:
                payload = points[0].payload
                parent_texts[parent_id] = payload.get("text", payload.get("chunk_text", ""))
    except Exception as e:
        print(f"  ⚠️  parent fetch 실패, child 텍스트 유지: {e}")
        return candidates

    updated = []
    for c in candidates:
        parent_id = c["payload"].get("parent_id")
        if parent_id and parent_id in parent_texts:
            updated_payload = dict(c["payload"])
            updated_payload["text"] = parent_texts[parent_id]
            updated.append({**c, "payload": updated_payload})
        else:
            updated.append(c)

    return updated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. 단일 청크 → 법령 검색 (전체 파이프라인)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def search_law_for_clause(
    clause_text : str,
    client      : QdrantClient,
    model       : BGEM3FlagModel,
    laws_ref    : dict[str, dict],
    reranker1   : CrossEncoderReranker | None = None,
    reranker2   : CrossEncoderReranker | None = None,
    top_k       : int   = TOP_K,
    alpha       : float = RRF_ALPHA,
) -> list[LawRef]:
    candidates = _qdrant_hybrid_search(
        clause_text=clause_text,
        client=client,
        model=model,
        collection=COLLECTION_HO,
        fetch_k=FETCH_K,
        alpha=alpha,
    )

    if reranker1 is not None and candidates:
        candidates = _rerank(
            query=clause_text,
            candidates=candidates,
            reranker=reranker1,
            top_k=RERANK1_K,
        )

    if reranker2 is not None and candidates:
        candidates = _rerank(
            query=clause_text,
            candidates=candidates,
            reranker=reranker2,
            top_k=RERANK2_K,
        )

    candidates = _fetch_parent_texts(candidates, client)

    law_refs: list[LawRef] = []
    for c in candidates[:top_k]:
        payload  = c["payload"]
        chunk_id = payload.get("chunk_id", "")
        ref_meta = laws_ref.get(chunk_id, {})

        law_refs.append(LawRef(
            chunk_id    = chunk_id,
            article     = ref_meta.get("article",  payload.get("article", "")),
            category    = ref_meta.get("category", payload.get("category", "")),
            law_name    = payload.get("law_name",  ""),
            chunk_text  = payload.get("text", payload.get("chunk_text", "")),
            score       = round(float(c.get("rrf_score", 0.0)), 4),
            is_risk_ref = bool(payload.get("is_risk_ref", False)),
            parent_id   = payload.get("parent_id", ""),
        ))

    return law_refs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. 전체 계약서 검토 (메인 인터페이스)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def review_contract(
    contract_text : str,
    client        : QdrantClient,
    model         : BGEM3FlagModel,
    laws_ref      : dict[str, dict] | None = None,
    reranker1     : CrossEncoderReranker | None = None,
    reranker2     : CrossEncoderReranker | None = None,
    top_k         : int   = TOP_K,
    alpha         : float = RRF_ALPHA,
) -> list[ClauseResult]:
    if laws_ref is None:
        laws_ref = load_laws_ref()

    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []

    print(f"  총 {len(clauses)}개 청크 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} 검색 중...", end="\r")

        law_refs = search_law_for_clause(
            clause_text = clause["clause_text"],
            client      = client,
            model       = model,
            laws_ref    = laws_ref,
            reranker1   = reranker1,
            reranker2   = reranker2,
            top_k       = top_k,
            alpha       = alpha,
        )

        categories = list(dict.fromkeys(
            ref.category for ref in law_refs if ref.category
        ))

        results.append(ClauseResult(
            clause_number = clause["clause_number"],
            clause_text   = clause["clause_text"],
            law_refs      = law_refs,
            categories    = categories,
        ))

    print("\n  ✅ 검색 완료")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. JSON 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def results_to_json(results: list[ClauseResult]) -> list[dict]:
    return [asdict(result) for result in results]