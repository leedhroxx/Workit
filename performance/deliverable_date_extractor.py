# -*- coding: utf-8 -*-
"""
사업수행계획서(PEP) PDF의 "산출물계획" 표를 룰베이스로 파싱해
각 산출물의 제출일자를 추출하고, Workit의 deliverable_type
(kickoff / tech_apply / final)에 매칭한다.

설계 원칙 (기존 파이프라인과 동일):
- 파싱 단계에는 LLM을 사용하지 않는다 (순수 rule-based)
- 산출물명 매칭은 config(dict)로 관리 → 코드 수정 없이 별칭 추가 가능
- 모든 결과는 source="rule" 로 표시 → 감사 추적성 확보
- 표 헤더("산출물명", "제출일정" 등)를 fuzzy 앵커 매칭으로 탐지
  (숫자/기호 제거 후 비교) → 문서마다 표 위치(페이지)가 달라도 동작
"""

import re
from dataclasses import dataclass
from typing import Optional

import pdfplumber

# 1. 설정 : deliverable_type 별 산출물명 별칭 목록 (fuzzy 매칭용)
#    - 새 산출물/별칭이 필요하면 이 dict만 수정하면 됨 (코드 변경 불필요)
DELIVERABLE_ALIASES = {
    "kickoff": [
        "사업수행계획서",
        "과업수행계획서",
        "착수보고서",
    ],
    "tech_apply": [
        "기술적용결과표",
        "기술적용계획표",
        "기술적용 결과표",
        "기술 적용 결과표",
    ],
    "final": [
        "사업추진결과보고서",
        "최종결과보고서",
        "최종 결과 보고서",
        "사업추진 결과 보고서",
    ],
}

# 표 헤더 판별용 앵커 (공백/특수문자 제거 후 비교). 회사 템플릿마다 열 이름이
# 달라서("산출물명"/"제출일정" vs "산출물"/"제출시기" 등) 여러 앵커 조합을 허용한다.
# 페이지/표에 그중 한 조합이 전부 포함되면 매칭으로 본다.
HEADER_ANCHOR_SETS = [
    ["산출물명", "제출일정"],
    ["산출물", "제출시기"],
    ["산출물", "제출일자"],
]
HEADER_ANCHORS = HEADER_ANCHOR_SETS[0]  # 하위 호환용 (기존 코드에서 참조하는 곳이 있을 수 있음)

# "2026. 3. 20" 형식과 "2026년 3월 20일" 형식을 모두 지원한다 (회사 템플릿마다 다름).
DATE_PATTERN = re.compile(
    r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})"
    r"|(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
)


def _match_header_anchor_set(normalized_text: str) -> Optional[list]:
    """normalized_text 안에 HEADER_ANCHOR_SETS 중 하나가 전부 포함되면 그 세트를 반환."""
    for anchor_set in HEADER_ANCHOR_SETS:
        if all(_normalize(a) in normalized_text for a in anchor_set):
            return anchor_set
    return None


def _normalize(text: str) -> str:
    """숫자·공백·특수문자를 제거해 fuzzy 비교에 사용."""
    if not text:
        return ""
    return re.sub(r"[\s\.\,\·\-\(\)\[\]]", "", text)


@dataclass
class DeliverablePlanItem:
    raw_name: str                # PDF에서 재구성한 산출물명
    due_date: Optional[str]      # YYYY-MM-DD
    copies: Optional[str]        # 제출부수
    doc_form: Optional[str]      # 유형 (PDF+책자 등)
    matched_type: Optional[str]  # kickoff / tech_apply / final / None
    source: str = "rule"         # 감사 추적용 고정값


def _find_output_plan_page(pdf: "pdfplumber.PDF") -> Optional[int]:
    """'산출물계획' 표가 있는 페이지를 헤더 앵커(여러 조합 중 하나)로 탐색."""
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        normalized = _normalize(text)
        if _match_header_anchor_set(normalized):
            return i
    return None


def _extract_raw_rows(page) -> list:
    """lines 전략으로 표를 추출 (셀 병합/줄바꿈 원형 유지)."""
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    }
    tables = page.extract_tables(table_settings)
    if not tables:
        return []
    # 헤더 앵커 조합 중 하나를 포함한 표를 선택
    for t in tables:
        flat = " ".join(str(c) for row in t for c in row if c)
        if _match_header_anchor_set(_normalize(flat)):
            return t
    return tables[0]


def parse_output_plan(pdf_path: str) -> list:
    """
    사업수행계획서 PDF에서 산출물계획 표를 파싱해
    DeliverablePlanItem 리스트로 반환.

    표 구조 특성상 산출물명이 셀 줄바꿈으로 두 물리적 행에
    걸쳐 쪼개지는 경우가 있어, 날짜(DATE_PATTERN)가 등장하는
    행을 기준으로 그 앞뒤 행의 파편을 이어붙여 이름을 복원한다.
    """
    items: list = []

    with pdfplumber.open(pdf_path) as pdf:
        page_idx = _find_output_plan_page(pdf)
        if page_idx is None:
            raise ValueError(
                "'산출물계획' 표를 찾을 수 없습니다. "
                "(헤더 '산출물명'/'제출일정' 미탐지)"
            )
        rows = _extract_raw_rows(pdf.pages[page_idx])

    pending_name_frag = ""  # 이전 행에서 넘어온 산출물명 파편

    for row in rows:
        cells = [(c or "").strip() for c in row]
        row_text = " ".join(c for c in cells if c)
        if not row_text:
            continue

        # 열 폭이 좁으면 "2025.03.10" 같은 날짜가 셀 안에서 "2025.03.1\n0"처럼 숫자 중간에 줄바꿈이 들어가 잘린다. 
        # 날짜 매칭만큼은 공백/줄바꿈을 전부 제거한 텍스트로 하되(원본 cells는 산출물명 표시용으로 그대로 둠).
        cells_compact = [re.sub(r"\s+", "", c) for c in cells]
        row_text_compact = re.sub(r"\s+", "", row_text)

        date_match = DATE_PATTERN.search(row_text_compact)

        if date_match:
            # 이 행에 날짜가 있음 = 산출물 레코드의 첫 물리적 행 앞쪽(0,1번 열 등)에 있는 비-날짜 텍스트를 산출물명 후보로 수집
            name_frags = []
            copies = None
            doc_form = None
            for c, c_compact in zip(cells, cells_compact):
                if not c or DATE_PATTERN.search(c_compact):
                    continue
                if re.match(r"^\d+식$|^\d*부$|^각\s*\d+부$", c):
                    copies = c
                elif "PDF" in c or "책자" in c or "Git" in c or "Figma" in c or "CD" in c:
                    doc_form = c
                else:
                    name_frags.append(c)

            # 후보가 여러 개면(예: '구분'+'산출물'처럼 앞에 카테고리 열이 따로 있는 템플릿) 알려진 산출물명과 매칭되는 후보를 우선 채택한다.
            # 매칭되는 게 없으면 기존처럼 첫 파편을 쓴다(원래 템플릿 - 산출물명 열이 맨 앞 - 과의 하위 호환).
            if len(name_frags) > 1:
                matched_frag = next(
                    (f for f in name_frags if classify_deliverable(re.sub(r"\s+", "", f))),
                    None,
                )
                chosen_frag = matched_frag if matched_frag is not None else name_frags[0]
            else:
                chosen_frag = name_frags[0] if name_frags else ""

            candidate_name = (pending_name_frag + " " + chosen_frag).strip()
            pending_name_frag = ""

            # DATE_PATTERN은 "2026. 3. 20"과 "2026년 3월 20일" 두 형식을 각각 별도 캡처 그룹으로 잡는다 - 어느 쪽이 매칭됐는지에 따라 선택.
            g = date_match.groups()
            y, m, d = (g[0], g[1], g[2]) if g[0] is not None else (g[3], g[4], g[5])
            due_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

            items.append({
                "candidate_name": candidate_name,
                "due_date": due_date,
                "copies": copies,
                "doc_form": doc_form,
            })
        else:
            # 날짜 없는 행 = 산출물명 두 번째 파편 (다음/이전 항목과 연결)
            frag_candidates = [c for c in cells if c]
            if frag_candidates and items:
                # 방금 추가된 항목의 이름 뒤에 이어붙임
                items[-1]["candidate_name"] = (
                    items[-1]["candidate_name"] + frag_candidates[0]
                ).strip()

    # 산출물명 재구성 결과 정제 + 타입 매칭
    results = []
    for it in items:
        raw_name = re.sub(r"\s+", "", it["candidate_name"])
        matched_type = classify_deliverable(raw_name)
        results.append(DeliverablePlanItem(
            raw_name=it["candidate_name"].strip(),
            due_date=it["due_date"],
            copies=it["copies"],
            doc_form=it["doc_form"],
            matched_type=matched_type,
        ))
    return results


def classify_deliverable(raw_name: str) -> Optional[str]:
    """산출물명을 정규화 후 별칭 dict와 fuzzy 매칭."""
    normalized = _normalize(raw_name)
    for dtype, aliases in DELIVERABLE_ALIASES.items():
        for alias in aliases:
            if _normalize(alias) in normalized or normalized in _normalize(alias):
                return dtype
    return None

