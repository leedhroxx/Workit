"""
Workit - 임베딩 결과 Qdrant 업로드 (upsert)
파일명: rag/law_upsert_qdrant.py

임베딩(vectors_*.npz, sparse_weights_*.json) + 파싱 결과(chunks_*.json)를
합쳐 Qdrant에 upsert. 검색은 law_rag_pipeline.py 담당 (이 스크립트가 먼저 실행돼
컬렉션이 존재해야 함).

입력: project_root/data/merged/ 안의 chunks_*.json, vectors_*.npz,
      sparse_weights_*.json (jo/ho 각각)
출력: Qdrant collection "law_kb_jo", "law_kb_ho"

RECREATE_COLLECTIONS=True(기본값): 매번 컬렉션 삭제 후 재생성 → 소스 JSON에서
빠진 chunk가 Qdrant에 고아로 안 남고 항상 동기화됨. False면 있으면 skip(빠른
반복 실험용).

Point id는 chunk_id 해시값(결정론적) — 배열 순서 아님, 매칭은 항상 chunk_id로.

사용법:
    pip install qdrant-client
    python rag/law_upsert_qdrant.py

--- 2026-07-14 수정 사항 ---
1. chunk_id_to_point_id: 내장 hash()는 문자열에 대해 PYTHONHASHSEED에 따라
   프로세스마다 값이 달라질 수 있어(PEP 456) "결정론적"이 아니었음. hashlib.sha256
   기반으로 교체 — 실행 환경/시점과 무관하게 같은 chunk_id는 항상 같은 point_id.
   (RECREATE_COLLECTIONS=True로만 써왔다면 지금까지는 겉으로 드러나지 않았을 버그.
   RECREATE_COLLECTIONS=False로 skip-upsert 모드를 쓸 계획이 있다면 반드시 필요한 수정.)
2. (취소됨) 애초엔 article_number를 PAYLOAD_FIELDS에서 제거하려 했음 — law_kb_jo에서
   article_id와 1,186/1,186 완전 일치해 중복 필드로 보였기 때문. 하지만 grep 결과
   inference.py, pdfver_yoonha_contract_rag.py가 payload["article_number"]를
   직접 읽고 있는 게 확인되어 제거 시 두 파일이 조용히 깨짐(에러 없이 값 누락).
   또한 law_chunk_reference.py(항/호/목 파서)를 보면 ho 레벨에서는 article_number가
   article_id보다 세분화된 값(예: "...제3항제2호")을 가질 걸로 보여, jo 레벨 완전
   일치는 우연일 가능성이 높음. → article_number는 그대로 유지.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "merged"

QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.environ.get('QDRANT_PORT', '6333'))

# True: 매번 컬렉션 삭제 후 재생성 (Qdrant를 소스 JSON과 100% 동기화, 고아 데이터 방지)
# False: 있으면 건너뛰고 upsert만 (빠른 반복 실험용)
RECREATE_COLLECTIONS = True

DATASETS = {
    "jo": {
        "chunks_file": DATA_DIR / "chunks_jo.json",
        "vectors_file": DATA_DIR / "vectors_jo.npz",
        "sparse_file": DATA_DIR / "sparse_weights_jo.json",
        "collection": "law_kb_jo",
    },
    "ho": {
        "chunks_file": DATA_DIR / "chunks_ho.json",
        "vectors_file": DATA_DIR / "vectors_ho.npz",
        "sparse_file": DATA_DIR / "sparse_weights_ho.json",
        "collection": "law_kb_ho",
    },
}

# 임베딩에 쓴 payload 필드 (text는 인덱싱만 하고 payload에도 남겨서 리랭크/표시에 사용)
# article_number 제거 (2026-07-14, 재확정) — article_id와 jo/ho 전수 100% 일치하는 중복 필드.
# inference.py / pdfver_yoonha_contract_rag.py는 참조하지만 미사용 파일이라 무관.
# eval 스크립트에서 article_number를 쓰는 부분은 별도로 article_id 참조로 수정 예정
# (담당: 전윤하) — 이 코드 반영 전에 eval 쪽 수정이 먼저 들어가 있어야 함.
PAYLOAD_FIELDS = [
    "chunk_id",
    "law_name",
    "article_id",
    "title",
    "text",
    "hierarchy",
    "parent_chunk_id",
    "cross_refs",
    "is_ref_article",
    "is_upper_law",
]


def chunk_id_to_point_id(chunk_id: str) -> int:
    """chunk_id -> 결정론적 정수 id. Qdrant point id는 int/UUID만 허용.

    hashlib.sha256 기반 — 내장 hash()와 달리 PYTHONHASHSEED 영향을 받지 않아
    실행 시점/환경이 달라져도 같은 chunk_id는 항상 같은 값을 반환한다.
    % (2**63)로 signed 64-bit 정수 범위 안으로 제한 (일부 클라이언트가 unsigned
    64-bit를 signed로 잘못 해석해 오버플로/음수 에러를 낼 수 있어 원래 코드의
    범위 제한 관례를 그대로 유지).
    """
    digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**63)


def load_chunks_by_id(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


def load_sparse_by_id(
    path: Path, chunk_ids_order: list[str]
) -> dict[str, dict[int, float]]:
    """sparse_weights_*.json은 chunk_ids_order(npz의 chunk_ids)와 같은 순서라고 가정하고 매칭."""
    with open(path, encoding="utf-8") as f:
        sparse_list = json.load(f)

    if len(sparse_list) != len(chunk_ids_order):
        raise ValueError(
            f"sparse_weights 길이({len(sparse_list)})와 chunk_ids 길이({len(chunk_ids_order)})가 다릅니다. "
            "임베딩 노트북에서 저장한 순서가 어긋났을 가능성이 있습니다."
        )

    return {
        cid: {int(tok): float(w) for tok, w in sw.items()}
        for cid, sw in zip(chunk_ids_order, sparse_list)
    }


def ensure_collection(
    client: QdrantClient, name: str, dense_dim: int, recreate: bool
) -> None:
    exists = client.collection_exists(name)

    if exists and not recreate:
        print(f"  [SKIP] 컬렉션 이미 존재: {name}")
        return

    if exists:
        client.delete_collection(name)
        print(f"  [DELETE] 기존 컬렉션 삭제: {name}")

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"  [CREATE] 컬렉션 생성: {name} (dense_dim={dense_dim})")


def build_points(
    chunk_ids: list[str],
    dense_vectors: np.ndarray,
    sparse_by_id: dict[str, dict[int, float]],
    chunks_by_id: dict[str, dict],
) -> list[PointStruct]:
    points = []
    missing_payload = 0

    for idx, cid in enumerate(chunk_ids):
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            missing_payload += 1
            continue

        payload = {k: chunk.get(k) for k in PAYLOAD_FIELDS if k in chunk}
        sparse = sparse_by_id.get(cid, {})

        points.append(
            PointStruct(
                id=chunk_id_to_point_id(cid),
                vector={
                    "dense": dense_vectors[idx].tolist(),
                    "sparse": SparseVector(
                        indices=list(sparse.keys()), values=list(sparse.values())
                    ),
                },
                payload=payload,
            )
        )

    if missing_payload:
        print(
            f"  ⚠️  chunks json에 없는 chunk_id {missing_payload}개는 스킵됨 (dense/sparse는 있는데 payload가 없는 경우)"
        )

    return points


def upload_dataset(
    client: QdrantClient, name: str, cfg: dict, batch_size: int = 256
) -> int:
    print(f"\n=== {name} ===")

    npz = np.load(cfg["vectors_file"], allow_pickle=True)
    dense_vectors = npz["vectors"]
    chunk_ids = [str(c) for c in npz["chunk_ids"]]

    chunks_by_id = load_chunks_by_id(cfg["chunks_file"])
    sparse_by_id = load_sparse_by_id(cfg["sparse_file"], chunk_ids)

    ensure_collection(
        client, cfg["collection"], dense_vectors.shape[1], recreate=RECREATE_COLLECTIONS
    )

    points = build_points(chunk_ids, dense_vectors, sparse_by_id, chunks_by_id)
    print(f"  업로드할 point 수: {len(points)}")

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=cfg["collection"], points=batch)
        print(f"    upsert {i + len(batch)}/{len(points)}", end="\r")

    count = client.count(cfg["collection"]).count
    print(
        f"\n  ✅ 완료: {cfg['collection']} (업로드 {len(points)}개 / 컬렉션 총 {count}개)"
    )
    return count


def missing_input_files(cfg: dict) -> list[Path]:
    return [
        cfg[key]
        for key in ("chunks_file", "vectors_file", "sparse_file")
        if not cfg[key].exists()
    ]


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for name, cfg in DATASETS.items():
        missing = missing_input_files(cfg)
        if missing:
            print(
                f"[ERROR] {name}: 다음 파일이 없어 건너뜁니다 -> {[str(p) for p in missing]}"
            )
            continue
        upload_dataset(client, name, cfg)

    print("\nDone!")


if __name__ == "__main__":
    main()