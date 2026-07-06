# -*- coding: utf-8 -*-
"""
기술적용결과표(tech_apply) PDF의 적용/부분적용/미적용/해당없음 체크 표를
룰베이스로 검증한다. (행정기관 및 공공기관 정보시스템 구축·운영 지침 별지 서식)

검증 규칙:
1. 4개 체크 칸(적용/부분적용/미적용/해당없음) 중 아무것도 체크되지 않음 → 오류
2. 체크 칸이 2개 이상 체크됨 → 오류
3. '부분적용' 또는 '미적용'이 체크됐는데 '부분적용/미적용시 사유 및 대체기술'란이
   비어있음 → 오류

설계 원칙 (기존 파이프라인과 동일 — deliverable_date_extractor.py 참고):
- 파싱/검증 단계에는 LLM을 사용하지 않는다 (순수 rule-based)
- 표 헤더는 fuzzy 앵커 매칭으로 탐지 → 표가 여러 페이지에 걸쳐 반복돼도 동작
- 체크 대상이 아닌 '그룹 라벨' 행(예: "기본 지침", "o 웹브라우저 관련") 판별은
  config로 관리 → 실제 문서로 검증 후 오탐이 있으면 이 목록만 수정하면 됨

⚠ 주의: 실제 서식의 표 구조(병합 셀 등)는 완성된 샘플 PDF로 검증되지 않았다.
   오탐/누락이 발견되면 아래 SKIP_LABEL_* 설정과 _find_header_indices()의
   앵커 텍스트를 조정할 것.
"""

import re
from dataclasses import dataclass
from typing import Optional

import pdfplumber

CHECKBOX_COLUMNS = ("적용", "부분적용", "미적용", "해당없음")
REASON_REQUIRED_FOR = ("부분적용", "미적용")

# 체크 대상이 아닌 '그룹 라벨' 행 판별용 설정
#  - 표 안에 구분선처럼 들어가는 정확한 라벨
SKIP_LABELS_EXACT = {"기본 지침", "세부 기술 지침"}
#  - "o 웹브라우저 관련"처럼 하위 항목을 묶기만 하는 소제목 (접미어로 판별)
SKIP_LABEL_SUFFIXES = ("관련",)

# 항목 행은 항상 불릿/대시로 시작한다 (o / ㅇ / ｏ / ○ / -)
BULLET_PREFIX_RE = re.compile(r"^[ㅇoO○ｏ\-–—]\s*")


def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\s\.\,\·\-\(\)\[\]]", "", text)


def _clean(cell) -> str:
    return (cell or "").strip()


@dataclass
class TechApplyCheckItem:
    page: int
    item_text: str
    checked: dict            # {"적용": bool, "부분적용": bool, "미적용": bool, "해당없음": bool}
    checked_count: int
    reason_filled: bool
    error: Optional[str] = None  # None이면 정상


def _find_header_indices(row) -> Optional[dict]:
    """표 헤더 행에서 항목/체크박스4개/사유 컬럼 인덱스를 fuzzy 매칭으로 탐색."""
    idx = {}
    for i, cell in enumerate(row):
        norm = _normalize(cell)
        if not norm:
            continue
        if norm in ("항목", "항목명") and "item" not in idx:
            idx["item"] = i
        elif norm == "적용" and "적용" not in idx:
            idx["적용"] = i
        elif norm == "부분적용" and "부분적용" not in idx:
            idx["부분적용"] = i
        elif norm == "미적용" and "미적용" not in idx:
            idx["미적용"] = i
        elif norm == "해당없음" and "해당없음" not in idx:
            idx["해당없음"] = i
        elif "사유" in norm and "대체기술" in norm and "사유" not in idx:
            idx["사유"] = i

    required = ("item", "적용", "부분적용", "미적용", "해당없음", "사유")
    if all(k in idx for k in required):
        return idx
    return None


def _is_checkable_item(item_text: str) -> bool:
    """항목 셀 텍스트가 실제 체크 대상 행인지(불릿으로 시작) 판별."""
    text = item_text.strip()
    if not text:
        return False
    if text in SKIP_LABELS_EXACT:
        return False
    if not BULLET_PREFIX_RE.match(text):
        return False
    bare = BULLET_PREFIX_RE.sub("", text).strip()
    if bare.endswith(SKIP_LABEL_SUFFIXES):
        return False
    return True


def _validate(checked: dict, reason_filled: bool) -> Optional[str]:
    checked_count = sum(1 for v in checked.values() if v)
    if checked_count == 0:
        return "체크된 항목이 없습니다."
    if checked_count >= 2:
        return "체크박스가 2개 이상 선택되어 있습니다."
    for col in REASON_REQUIRED_FOR:
        if checked.get(col) and not reason_filled:
            return f"'{col}' 선택 시 사유 및 대체기술을 작성해야 합니다."
    return None


def check_tech_apply(pdf_path: str) -> dict:
    """
    기술적용결과표 PDF를 페이지별로 순회하며 체크 표를 검증한다.

    Returns:
        {
          "total": int,
          "error_count": int,
          "items": [
            {"page": int, "item_text": str, "checked": dict,
             "checked_count": int, "reason_filled": bool, "error": str|None},
            ...
          ],
        }
    """
    items: list = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            })
            for table in tables:
                header_idx = None
                for row in table:
                    if header_idx is None:
                        header_idx = _find_header_indices(row)
                        continue

                    item_i = header_idx["item"]
                    item_cell = _clean(row[item_i]) if item_i < len(row) else ""
                    if not _is_checkable_item(item_cell):
                        continue

                    checked = {}
                    for col in CHECKBOX_COLUMNS:
                        ci = header_idx[col]
                        cell_val = row[ci] if ci < len(row) else None
                        checked[col] = bool(_clean(cell_val))

                    reason_i = header_idx["사유"]
                    reason_filled = bool(_clean(row[reason_i])) if reason_i < len(row) else False

                    error = _validate(checked, reason_filled)

                    items.append(TechApplyCheckItem(
                        page=page_no,
                        item_text=item_cell,
                        checked=checked,
                        checked_count=sum(1 for v in checked.values() if v),
                        reason_filled=reason_filled,
                        error=error,
                    ))

    error_items = [it for it in items if it.error]

    return {
        "total": len(items),
        "error_count": len(error_items),
        "items": [
            {
                "page": it.page,
                "item_text": it.item_text,
                "checked": it.checked,
                "checked_count": it.checked_count,
                "reason_filled": it.reason_filled,
                "error": it.error,
            }
            for it in items
        ],
    }
