"""
LLM/qa_agent/bbox_locator.py

PDF에서 각 소제목(SectionSpec)의 "내용 전체"가 차지하는 영역을 찾는다 — 소제목
텍스트 자체의 위치가 아니라, 그 소제목부터 다음 소제목 직전까지의 구간을
페이지 번호 + 페이지 대비 비율(%) bbox로 반환한다(여러 페이지에 걸치면 fragments로
나뉜다). 소제목 클릭 시 좌측 뷰어에 하이라이트를 표시하는 기능(contracts의 clause
하이라이트, performance의 tech_apply 하이라이트와 같은 목적)에 쓰인다.

engine.py의 _sequential_search와 같은 전략으로, 코드 순서대로 진행하면서 이전에 찾은
페이지보다 앞으로는 돌아가지 않는다 — 목차 등에 같은 문구가 먼저 나와도 실제 본문
위치를 안정적으로 잡기 위함이다.
"""
from __future__ import annotations

from typing import Dict, List

import fitz

from qa_agent.registry import get_sections


def _page_pct_bbox(page_rect, top_pt: float, bottom_pt: float) -> dict:
    """세로 구간(top_pt~bottom_pt)을 페이지 폭 전체(0~100%) 가로 밴드로 변환한다.
    표/문단마다 좌측 여백이 달라 텍스트 폭에 맞추면 잘려 보일 수 있어, 소제목
    아래 내용 전체를 안전하게 덮도록 폭은 항상 페이지 전체로 잡는다."""
    height = page_rect.height
    top = max(0.0, min(100.0, top_pt / height * 100))
    bottom = max(0.0, min(100.0, bottom_pt / height * 100))
    return {
        'left': 0.0,
        'top': round(top, 3),
        'width': 100.0,
        'height': round(max(0.0, bottom - top), 3),
    }


def _fragments_between(doc: "fitz.Document", start_page: int, start_y: float,
                        end_page: int, end_y: float) -> List[dict]:
    """(start_page, start_y) ~ (end_page, end_y) 사이의 영역을 페이지별로 잘라
    fragments로 반환한다 (rag/clause_locator.py의 페이지 분할 방식과 동일한 개념)."""
    if start_page == end_page:
        return [{
            'page': start_page + 1,
            'bbox': _page_pct_bbox(doc[start_page].rect, start_y, end_y),
        }]

    fragments = [{
        'page': start_page + 1,
        'bbox': _page_pct_bbox(doc[start_page].rect, start_y, doc[start_page].rect.height),
    }]
    for mid in range(start_page + 1, end_page):
        fragments.append({
            'page': mid + 1,
            'bbox': _page_pct_bbox(doc[mid].rect, 0, doc[mid].rect.height),
        })
    fragments.append({
        'page': end_page + 1,
        'bbox': _page_pct_bbox(doc[end_page].rect, 0, end_y),
    })
    return fragments


def locate_section_bboxes(pdf_path: str, document_type: str) -> Dict[str, dict]:
    """
    document_type(rfp/pep/rpt)의 각 소제목이 PDF에서 차지하는 영역(소제목부터 다음
    소제목 직전까지)을 찾는다.

    반환: {code: {"fragments": [{"page": int(1-base), "bbox": {...}}, ...]}}
    bbox 값은 페이지 폭/높이 대비 0~100 비율(%) — performance/tech_apply_checker.py와
    같은 방식이라 DPI 변환 없이 그대로 이미지 픽셀에 곱해 쓸 수 있다.

    PDF를 열 수 없거나 소제목을 못 찾으면 해당 항목은 결과에서 빠진다
    (하이라이트 없이 텍스트만 표시됨 — 조용히 성능을 낮추는 기존 패턴과 동일).
    """
    sections = get_sections(document_type)
    result: Dict[str, dict] = {}

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return result

    try:
        # 1. 각 소제목의 시작 위치(등장 순서대로)를 찾는다.
        starts: List[tuple] = []  # [(spec, page_idx, y0), ...]
        start_page = 0
        for spec in sections:
            found = None
            for page_idx in range(start_page, len(doc)):
                page = doc[page_idx]
                for candidate in spec.title_candidates():
                    if not candidate:
                        continue
                    rects = page.search_for(candidate)
                    if rects:
                        found = (page_idx, rects[0].y0)
                        break
                if found:
                    break

            if not found:
                continue

            starts.append((spec, found[0], found[1]))
            start_page = found[0]

        # 2. 각 소제목의 영역 = 자기 시작 위치 ~ 다음 소제목 시작 위치 직전(마지막이면 문서 끝).
        last_page_idx = len(doc) - 1
        for i, (spec, page_idx, y0) in enumerate(starts):
            if i + 1 < len(starts):
                end_page_idx, end_y = starts[i + 1][1], starts[i + 1][2]
            else:
                end_page_idx, end_y = last_page_idx, doc[last_page_idx].rect.height

            result[spec.code] = {
                'fragments': _fragments_between(doc, page_idx, y0, end_page_idx, end_y),
            }
    finally:
        doc.close()

    return result
