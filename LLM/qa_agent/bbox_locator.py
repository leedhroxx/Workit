"""
LLM/qa_agent/bbox_locator.py

PDF에서 각 소제목(SectionSpec)이 실제로 등장하는 위치를 페이지 번호 + 페이지 대비
비율(%) bbox로 찾는다. 소제목 클릭 시 좌측 뷰어에 하이라이트를 표시하는 기능
(contracts의 clause 하이라이트, performance의 tech_apply 하이라이트와 같은 목적)에 쓰인다.

engine.py의 _sequential_search와 같은 전략으로, 코드 순서대로 진행하면서 이전에 찾은
페이지보다 앞으로는 돌아가지 않는다 — 목차 등에 같은 문구가 먼저 나와도 실제 본문
위치를 안정적으로 잡기 위함이다.
"""
from __future__ import annotations

from typing import Dict

import fitz

from qa_agent.registry import get_sections


def locate_section_bboxes(pdf_path: str, document_type: str) -> Dict[str, dict]:
    """
    document_type(rfp/pep/rpt)의 소제목이 PDF에서 처음 등장하는 위치를 찾는다.

    반환: {code: {"page": int(1-base), "bbox": {"left","top","width","height"}}}
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
                        found = (page_idx, rects[0])
                        break
                if found:
                    break

            if not found:
                continue

            page_idx, rect = found
            page = doc[page_idx]
            result[spec.code] = {
                'page': page_idx + 1,
                'bbox': {
                    'left': round(rect.x0 / page.rect.width * 100, 3),
                    'top': round(rect.y0 / page.rect.height * 100, 3),
                    'width': round((rect.x1 - rect.x0) / page.rect.width * 100, 3),
                    'height': round((rect.y1 - rect.y0) / page.rect.height * 100, 3),
                },
            }
            start_page = page_idx
    finally:
        doc.close()

    return result
