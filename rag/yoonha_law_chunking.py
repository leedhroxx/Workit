"""
═══════════════════════════════════════════════════════════════════
Workit RAG Pipeline — 법령 KB 구축
파일명: yoonha_law_chunking.py
위치:   Workit/rag/law_chunking.py
═══════════════════════════════════════════════════════════════════

■ 이 파일의 역할
──────────────────────────────────────────────────────────────────
법령 JSON 파일을 읽어 청킹 → taxonomy 리스크 매핑 → Qdrant 저장
까지의 전체 파이프라인을 담당합니다.

처리 흐름:
    1. yoonha_qdrant_manager    → Docker 컨테이너 자동 기동
    2. load_law_json()          → 법령 JSON 파싱 (조문 리스트)
    3. chunk_articles()         → RSC 방식으로 조문 청킹
                                  + taxonomy 리스크 태그 부착
    4. store_chunks()           → BGE-M3 임베딩 후 Qdrant 저장
    (+) yoonha_token_stats_logger → 청킹 전 토큰 분포 측정 및 로그 저장

■ 법령 파일 목록 (data/structured/ 위치)
──────────────────────────────────────────────────────────────────
  - 소프트웨어_진흥법.json
  - 지방계약법.json
  - 지방계약법_시행규칙.json
  - 지방계약법_시행령.json
  - 지방자치단체 용역계약 일반조건 (행안부 예규).json

■ 리스크 taxonomy
──────────────────────────────────────────────────────────────────
  9개 리스크 유형 (RISK_001 ~ RISK_009) 이 법령 조문과 매핑됩니다.
  매핑된 조문 청크에는 risk_ids, risk_names, is_risk_ref=True 가 부착됩니다.
  이를 통해 계약서 분석 시 리스크 조문만 필터링하여 검색할 수 있습니다.

■ Qdrant 컬렉션
──────────────────────────────────────────────────────────────────
  컬렉션명 : law_kb
  벡터 차원 : BGE-M3 자동 감지
  거리 방식 : Cosine Similarity

■ 실행 방법
──────────────────────────────────────────────────────────────────
  # 전체 법령 적재
  python law_chunking.py

  # 컬렉션 초기화 후 재적재
  python law_chunking.py --reset

■ 실행 전 설치
──────────────────────────────────────────────────────────────────
  pip install qdrant-client sentence-transformers numpy
"""

import argparse
import hashlib
import json
import uuid
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from yoonha_qdrant_manager import ensure_qdrant_running
from yoonha_token_stats_logger import log_token_stats_from_texts


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
DATA_DIR    = Path("data/structured")   # 법령 JSON 파일 위치
QDRANT_HOST = "localhost"               # Qdrant Docker 호스트
QDRANT_PORT = 6333                      # Qdrant REST API 포트
COLLECTION  = "law_kb"                  # Qdrant 컬렉션명
EMBED_MODEL = "BAAI/bge-m3"            # 임베딩 모델 (한국어 특화)
MAX_TOKENS  = 1024                      # 청크 최대 토큰 수 (BGE-M3 컨텍스트 길이)
MIN_TOKENS  = 30                        # 이 이하 토큰의 조문은 청킹 대상 제외

# 법령 파일명 → (law_name, law_type) 매핑
LAW_META = {
    "소프트웨어_진흥법.json": {
        "law_name": "소프트웨어 진흥법",
        "law_type": "법률",
    },
    "지방계약법.json": {
        "law_name": "지방계약법",
        "law_type": "법률",
    },
    "지방계약법_시행규칙.json": {
        "law_name": "지방계약법 시행규칙",
        "law_type": "시행규칙",
    },
    "지방계약법_시행령.json": {
        "law_name": "지방계약법 시행령",
        "law_type": "시행령",
    },
    "지방자치단체 용역계약 일반조건 (행안부 예규).json": {
        "law_name": "지방자치단체 용역계약 일반조건",
        "law_type": "예규",
    },
}

# taxonomy: 9개 리스크 유형 + 관련 법령 조문 매핑
# 계약서에서 해당 리스크가 감지되면 이 조문들이 근거로 활용됩니다
TAXONOMY = [
    {
        "risk_id":   "RISK_001",
        "risk_name": "지연배상금 상한 미설정",
        "law_refs":  [
            {"filename": "지방계약법_시행령.json", "article_id": "제90조"},
        ],
    },
    {
        "risk_id":   "RISK_002",
        "risk_name": "지연배상금률 과다 설정",
        "law_refs":  [
            {"filename": "지방계약법_시행규칙.json", "article_id": "제75조"},
        ],
    },
    {
        "risk_id":   "RISK_003",
        "risk_name": "대금 지급 기한 미설정",
        "law_refs":  [
            {"filename": "지방계약법.json",       "article_id": "제18조"},
            {"filename": "지방계약법_시행령.json", "article_id": "제67조"},
        ],
    },
    {
        "risk_id":   "RISK_004",
        "risk_name": "합의 없는 일방적 과업 추가",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_나"},
            {"filename": "소프트웨어_진흥법.json",                          "article_id": "제50조"},
        ],
    },
    {
        "risk_id":   "RISK_005",
        "risk_name": "추가 과업 비용 미지급",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_라"},
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_마"},
            {"filename": "지방계약법_시행령.json",                          "article_id": "제74조"},
        ],
    },
    {
        "risk_id":   "RISK_006",
        "risk_name": "갑의 일방적 해지권",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제7절_제4항_다"},
        ],
    },
    {
        "risk_id":   "RISK_007",
        "risk_name": "을의 해제권 배제/제한",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제7절_제5항_가"},
        ],
    },
    {
        "risk_id":   "RISK_008",
        "risk_name": "계약금액 조정 없는 과업변경",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_라"},
            {"filename": "지방계약법.json",       "article_id": "제22조"},
            {"filename": "지방계약법_시행령.json", "article_id": "제74조"},
        ],
    },
    {
        "risk_id":   "RISK_009",
        "risk_name": "손해배상 범위 일방적 제한",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제8절_제7항_가"},
            {"filename": "소프트웨어_진흥법.json",                          "article_id": "제38조"},
        ],
    },
]


# ──────────────────────────────────────────
# 1. taxonomy 역인덱스 빌드
# ──────────────────────────────────────────
def build_risk_index(taxonomy: list[dict]) -> dict:
    """
    taxonomy를 (파일명, 조문ID) → 리스크 목록 형태의 역인덱스로 변환합니다.

    청킹 시 각 조문이 어떤 리스크와 연관되는지 O(1) 로 조회하기 위해
    미리 빌드해두는 구조입니다.

    Args:
        taxonomy : TAXONOMY 상수 (리스크 유형 + 법령 조문 매핑 목록)

    Returns:
        dict: { (filename, article_id): [{"risk_id": ..., "risk_name": ...}, ...] }

    예시:
        key   = ("지방계약법_시행령.json", "제90조")
        value = [{"risk_id": "RISK_001", "risk_name": "지연배상금 상한 미설정"}]
    """
    index = defaultdict(list)
    for risk in taxonomy:
        for ref in risk["law_refs"]:
            key = (ref["filename"], ref["article_id"])
            index[key].append({
                "risk_id":   risk["risk_id"],
                "risk_name": risk["risk_name"],
            })
    return dict(index)


# ──────────────────────────────────────────
# 2. JSON 로드
# ──────────────────────────────────────────
def load_law_json(filepath: Path) -> list[dict]:
    """
    법령 JSON 파일을 읽어 조문(article) 리스트를 반환합니다.

    JSON 구조: { "articles": [ { "article_id": ..., "text": ... }, ... ] }

    Args:
        filepath : 법령 JSON 파일 경로

    Returns:
        list[dict]: 조문 딕셔너리 리스트
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("articles", [])


# ──────────────────────────────────────────
# 3. RSC 청킹
# ──────────────────────────────────────────
# 재귀적으로 시도할 구분자 우선순위 (큰 단위 → 작은 단위 순)
SEPARATORS = ["\n\n", "\n", ". ", " "]


def count_tokens(text: str, tokenizer) -> int:
    """
    텍스트의 토큰 수를 반환합니다.

    BGE-M3 모델의 tokenizer를 사용하므로
    실제 임베딩 시 적용되는 토큰 수와 동일합니다.

    Args:
        text      : 토큰 수를 셀 텍스트
        tokenizer : SentenceTransformer 모델의 내장 tokenizer

    Returns:
        int: 토큰 수
    """
    return len(tokenizer.encode(text))


def recursive_split(text: str, tokenizer, sep_index: int = 0) -> list[str]:
    """
    텍스트를 MAX_TOKENS 이하가 될 때까지 재귀적으로 분할합니다. (RSC)

    Recursive Splitting by Character 방식:
    SEPARATORS 순서대로 구분자를 시도하며, 각 분할된 조각이
    여전히 MAX_TOKENS를 초과하면 다음 구분자로 재귀 분할합니다.

    Args:
        text      : 분할할 텍스트
        tokenizer : 토큰 수 측정용 tokenizer
        sep_index : 현재 시도 중인 구분자 인덱스 (재귀 호출 시 증가)

    Returns:
        list[str]: MAX_TOKENS 이하 크기의 텍스트 청크 리스트
    """
    if count_tokens(text, tokenizer) <= MAX_TOKENS:
        return [text.strip()] if text.strip() else []
    if sep_index >= len(SEPARATORS):
        return [text.strip()]
    sep   = SEPARATORS[sep_index]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return recursive_split(text, tokenizer, sep_index + 1)
    result = []
    for part in parts:
        if count_tokens(part, tokenizer) > MAX_TOKENS:
            result.extend(recursive_split(part, tokenizer, sep_index + 1))
        else:
            result.append(part.strip())
    return result


# ──────────────────────────────────────────
# 4. chunk_id 생성 (해시 기반)
# ──────────────────────────────────────────
def make_chunk_id(law_name: str, article_id: str, sub_index: int) -> str:
    """
    청크 단위 고유 ID를 UUID 형태로 생성합니다.

    law_name + article_id + sub_index 조합을 MD5 해싱하여
    동일 조문의 동일 청크는 항상 동일한 ID를 갖습니다. (멱등성)
    Qdrant upsert 시 동일 ID면 덮어쓰므로 재실행해도 중복이 생기지 않습니다.

    Args:
        law_name   : 법령명 (예: "지방계약법 시행령")
        article_id : 조문 ID (예: "제90조")
        sub_index  : 조문 내 청크 순번 (조문이 분할되지 않으면 항상 0)

    Returns:
        str: UUID 형태의 문자열 ID
    """
    raw = f"{law_name}::{article_id}::{sub_index}"
    return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


# ──────────────────────────────────────────
# 5. 청크 생성 + 메타데이터 태깅
# ──────────────────────────────────────────
def chunk_articles(
    articles:   list[dict],
    tokenizer,
    law_meta:   dict,
    filename:   str,
    risk_index: dict,
) -> list[dict]:
    """
    조문 리스트를 청킹하고 taxonomy 리스크 태그를 부착합니다.

    각 조문에 대해:
        1. risk_index 에서 해당 조문의 리스크 매핑 조회
        2. 조문 텍스트를 MAX_TOKENS 이하로 RSC 청킹
        3. 각 청크에 법령 정보 + 리스크 태그 메타데이터 부착

    is_risk_ref=True 인 청크는 계약서 리스크 분석 시
    필터링하여 우선 검색 대상으로 활용됩니다.

    Args:
        articles   : load_law_json() 반환값
        tokenizer  : BGE-M3 내장 tokenizer
        law_meta   : LAW_META 의 단일 항목 {"law_name": ..., "law_type": ...}
        filename   : 법령 JSON 파일명 (risk_index 키 조회용)
        risk_index : build_risk_index() 반환값

    Returns:
        list[dict]: 메타데이터가 부착된 청크 딕셔너리 리스트
    """
    chunks = []

    for article in articles:
        text = article.get("text", "").strip()
        if not text:
            continue

        article_id     = article.get("article_id", "")
        article_number = article.get("article_number", "")
        title          = article.get("title", "")

        # taxonomy 역인덱스에서 이 조문의 리스크 매핑 조회
        key        = (filename, article_id)
        risk_hits  = risk_index.get(key, [])
        risk_ids   = [r["risk_id"]   for r in risk_hits]
        risk_names = [r["risk_name"] for r in risk_hits]

        # RSC 청킹 (MAX_TOKENS 이하면 분할 없이 그대로)
        sub_chunks = (
            [text] if count_tokens(text, tokenizer) <= MAX_TOKENS
            else recursive_split(text, tokenizer)
        )

        for idx, chunk_text in enumerate(sub_chunks):
            if not chunk_text:
                continue

            chunk = {
                # ── 식별 ──────────────────────────────────────────────
                "chunk_id":       make_chunk_id(law_meta["law_name"], article_id, idx),
                "source_type":    "law",            # form 컬렉션과 구분하는 식별자
                # ── 법령 정보 ──────────────────────────────────────────
                "law_name":       law_meta["law_name"],
                "law_type":       law_meta["law_type"],
                "article_id":     article_id,
                "article_number": article_number,
                "article_title":  title,
                "source_full":    f"{law_meta['law_name']} {article_number}",
                # ── 리스크 매핑 ────────────────────────────────────────
                "risk_ids":       risk_ids,
                "risk_names":     risk_names,
                "is_risk_ref":    len(risk_hits) > 0,   # 리스크 관련 조문 여부
                # ── 청크 정보 ──────────────────────────────────────────
                "text":           chunk_text,
                "chunk_tokens":   count_tokens(chunk_text, tokenizer),
                "sub_index":      idx,
            }
            chunks.append(chunk)

    return chunks


# ──────────────────────────────────────────
# 6. Qdrant 초기화
# ──────────────────────────────────────────
def ensure_collection(client: QdrantClient, embed_dim: int, reset: bool = False) -> None:
    """
    law_kb 컬렉션이 없으면 생성하고, 있으면 그대로 사용합니다.

    reset=True 이면 기존 컬렉션을 삭제 후 재생성합니다.
    법령 개정 등으로 전체 재적재가 필요할 때 사용합니다.

    Args:
        client    : QdrantClient 인스턴스
        embed_dim : 임베딩 벡터 차원 수 (BGE-M3 자동 감지값)
        reset     : True이면 기존 컬렉션 삭제 후 재생성
    """
    existing = [c.name for c in client.get_collections().collections]

    if reset and COLLECTION in existing:
        client.delete_collection(collection_name=COLLECTION)
        print(f"🗑️  컬렉션 초기화: {COLLECTION}")
        existing = []

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        print(f"✅ 컬렉션 생성: {COLLECTION} (dim={embed_dim})")
    else:
        print(f"✅ 컬렉션 기존 사용: {COLLECTION}")


# ──────────────────────────────────────────
# 7. 청크 저장 — 배치 임베딩
# ──────────────────────────────────────────
def store_chunks(
    client:     QdrantClient,
    chunks:     list[dict],
    model:      SentenceTransformer,
    batch_size: int = 32,
) -> None:
    """
    청크 리스트를 배치 임베딩 후 Qdrant에 저장합니다.

    메모리 효율을 위해 batch_size 단위로 나누어 임베딩합니다.
    chunk_id가 해시 기반 UUID이므로 재실행 시 중복 없이 upsert됩니다.

    Args:
        client     : QdrantClient 인스턴스
        chunks     : chunk_articles() 반환값
        model      : SentenceTransformer 임베딩 모델
        batch_size : 한 번에 임베딩할 청크 수 (GPU 메모리 조절용)
    """
    texts      = [c["text"] for c in chunks]
    all_points = []

    print(f"  임베딩 생성 중... ({len(texts)}개)")
    for i in range(0, len(texts), batch_size):
        batch_vecs = model.encode(
            texts[i:i + batch_size],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for j, vec in enumerate(batch_vecs):
            chunk   = chunks[i + j]
            payload = {k: v for k, v in chunk.items() if k != "chunk_id"}
            all_points.append(PointStruct(
                id=chunk["chunk_id"],
                vector=vec.tolist(),
                payload=payload,
            ))

    # 100개 단위로 나누어 upsert (Qdrant 권장 배치 크기)
    for i in range(0, len(all_points), 100):
        client.upsert(collection_name=COLLECTION, points=all_points[i:i + 100])
    print(f"  ✅ {len(all_points)}개 청크 저장")


# ──────────────────────────────────────────
# 8. 메인
# ──────────────────────────────────────────
def main():
    # Qdrant Docker 컨테이너 자동 기동
    ensure_qdrant_running()

    ap = argparse.ArgumentParser(description="Workit 법령 KB 구축")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="컬렉션 삭제 후 재구축 (기본값: 기존 컬렉션 유지하며 upsert)",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("Workit 법령 KB 구축")
    print("=" * 60)

    # 모델 로드 — SentenceTransformer 내장 tokenizer 재사용
    print(f"\n📦 모델 로드: {EMBED_MODEL}")
    print("  (첫 실행 시 약 2GB 다운로드)")
    embed_model = SentenceTransformer(EMBED_MODEL)
    tokenizer   = embed_model.tokenizer
    embed_dim   = embed_model.get_sentence_embedding_dimension()
    print(f"  임베딩 차원: {embed_dim}")

    # taxonomy 역인덱스
    risk_index = build_risk_index(TAXONOMY)
    print(f"\n📋 taxonomy 로드: {len(TAXONOMY)}개 리스크 유형")
    print(f"   매핑된 조문 수: {len(risk_index)}개")

    # Qdrant 초기화
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client, embed_dim, reset=args.reset)

    total       = 0
    risk_tagged = 0

    # 파일별 처리
    for filename, law_meta in LAW_META.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"\n⚠️  파일 없음: {filepath}")
            continue

        print(f"\n{'─' * 50}")
        print(f"📄 {law_meta['law_name']} ({law_meta['law_type']})")

        articles = load_law_json(filepath)
        print(f"  조문 수: {len(articles)}개")

        # 토큰 통계 측정 및 로그 저장
        texts_for_stats = [a["text"] for a in articles if a.get("text")]
        log_token_stats_from_texts(f"law:{law_meta['law_name']}", texts_for_stats, tokenizer)

        chunks = chunk_articles(articles, tokenizer, law_meta, filename, risk_index)
        tagged = sum(1 for c in chunks if c["is_risk_ref"])
        print(f"  청크 수: {len(chunks)}개  |  리스크 태깅: {tagged}개")

        store_chunks(client, chunks, embed_model)
        total       += len(chunks)
        risk_tagged += tagged

    # 결과 요약
    print(f"\n{'=' * 60}")
    print(f"✅ 완료!")
    print(f"   총 청크: {total}개")
    print(f"   리스크 태깅된 청크: {risk_tagged}개")
    count = client.count(collection_name=COLLECTION)
    print(f"   Qdrant 저장: {count.count}개")
    print(f"   Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")

    # 검색 테스트
    print(f"\n🔍 검색 테스트: '지연배상금 한도' (리스크 조문만)")
    query_vec = embed_model.encode("지연배상금 한도", normalize_embeddings=True)
    results   = client.query_points(
        collection_name=COLLECTION,
        query=query_vec.tolist(),
        query_filter=Filter(
            must=[FieldCondition(key="is_risk_ref", match=MatchValue(value=True))]
        ),
        limit=3,
    ).points

    for i, r in enumerate(results, 1):
        risk_str = ", ".join(r.payload.get("risk_names", [])) or "—"
        print(f"  [{i}] score={r.score:.4f} | {r.payload['source_full']}")
        print(f"       리스크: {risk_str}")
        print(f"       {r.payload['text'][:80]}...")


if __name__ == "__main__":
    main()