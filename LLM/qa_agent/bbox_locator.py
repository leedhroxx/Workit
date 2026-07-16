"""
LLM/qa_agent/bbox_locator.py

PDF에서 각 소제목(SectionSpec)의 "내용 전체"가 차지하는 영역을 찾는다 — 소제목
텍스트 자체의 위치가 아니라, 그 소제목부터 다음 소제목 직전까지의 구간을
페이지 번호 + 페이지 대비 비율(%) bbox로 반환한다(여러 페이지에 걸치면 fragments로
나뉜다). 소제목 클릭 시 좌측 뷰어에 하이라이트를 표시하는 기능(contracts의 clause
하이라이트, performance의 tech_apply 하이라이트와 같은 목적)에 쓰인다.

위치 탐색 자체는 두 가지를 engine.py/rag.clause_locator.py와 맞춘다:
1. PDF에서 추출한 글자 사이에 불규칙한 공백이 섞이는 경우가 흔해서(예: "사 업 명"),
   PyMuPDF의 page.search_for()로 그냥 찾으면 실제 위치를 놓치고 우연히 비슷한 문구가
   있는 엉뚱한 위치를 짚어버릴 수 있다 — 그래서 rag/clause_locator.py처럼 글자 단위
   좌표(rawdict)를 모아 공백을 제거한 정규화 문자열로 찾은 뒤, 원래 글자의 좌표로
   되돌린다(qa_agent.text_utils.normalize_compare_text와 동일한 정규화 규칙).
2. 목차에 소제목이 본문보다 먼저 나열되는 문서가 많아, 찾은 위치가 문서 앞부분
   좁은 페이지 구간에 몰려 있으면(=목차로 추정) 그 구간 다음부터 다시 탐색한다
   (LLM/qa_agent/engine.py._find_section_positions와 같은 전략).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import fitz

from qa_agent.registry import get_sections

_STRIP_PATTERN = re.compile(r"[()（）\[\]【】ㆍ·.,:;]")


def _normalize_for_match(s: str) -> str:
    """qa_agent.text_utils.normalize_compare_text와 동일한 정규화 규칙."""
    s = (s or "").replace(" ", " ")
    s = re.sub(r"\s+", "", s)
    s = _STRIP_PATTERN.sub("", s)
    return s


def _page_char_stream(page: "fitz.Page") -> List[Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]]:
    """페이지의 모든 글자를 읽기 순서대로 (글자, x0, y0, x1, y1) 목록으로 반환한다.
    rag/clause_locator.py의 reconstruct_line_text와 같은 방식을 페이지 전체로 확장한 것."""
    chars: List[Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    blocks = page.get_text("rawdict")["blocks"]
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if "chars" in span:
                    for c in span["chars"]:
                        x0, y0, x1, y1 = c["bbox"]
                        chars.append((c["c"], x0, y0, x1, y1))
                else:
                    x0, y0, x1, y1 = span["bbox"]
                    chars.append((span["text"], x0, y0, x1, y1))
            # 줄이 바뀌는 자리에 구분자를 넣어서, 서로 다른 줄의 글자가 우연히
            # 이어붙어 엉뚱하게 매칭되는 일이 없게 한다(정규화 시 공백은 제거되므로
            # 매칭 자체에는 영향 없이, 아래 index_map에서 위치만 구분해준다).
            chars.append((" ", None, None, None, None))
    return chars


class _PageIndex:
    """한 페이지 안에서 정규화된 텍스트 검색을 반복할 수 있도록 미리 계산해두는 캐시."""

    def __init__(self, page: "fitz.Page"):
        self.chars = _page_char_stream(page)
        norm_chars: List[str] = []
        index_map: List[int] = []
        for i, (ch, *_rest) in enumerate(self.chars):
            norm = _normalize_for_match(ch)
            if not norm:
                continue
            norm_chars.append(norm)
            index_map.extend([i] * len(norm))
        self.norm_text = "".join(norm_chars)
        self.index_map = index_map

    def find(self, candidate: str) -> Optional[Tuple[float, float, float, float]]:
        cand_norm = _normalize_for_match(candidate)
        if not cand_norm:
            return None
        idx = self.norm_text.find(cand_norm)
        if idx < 0:
            return None

        start_i = self.index_map[idx]
        end_i = self.index_map[idx + len(cand_norm) - 1]
        xs0, ys0, xs1, ys1 = [], [], [], []
        for i in range(start_i, end_i + 1):
            _ch, x0, y0, x1, y1 = self.chars[i]
            if x0 is None:
                continue
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        if not xs0:
            return None
        return (min(xs0), min(ys0), max(xs1), max(ys1))


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
        page_index_cache: Dict[int, _PageIndex] = {}

        def get_page_index(idx: int) -> _PageIndex:
            cached = page_index_cache.get(idx)
            if cached is None:
                cached = _PageIndex(doc[idx])
                page_index_cache[idx] = cached
            return cached

        def search_from(start_page: int) -> List[Tuple[object, int, float]]:
            starts: List[Tuple[object, int, float]] = []
            cursor_page = start_page
            for spec in sections:
                found = None
                for page_idx in range(cursor_page, len(doc)):
                    pidx = get_page_index(page_idx)
                    for candidate in spec.title_candidates():
                        if not candidate:
                            continue
                        bbox = pidx.find(candidate)
                        if bbox:
                            found = (page_idx, bbox[1])  # y0
                            break
                    if found:
                        break
                if not found:
                    continue
                starts.append((spec, found[0], found[1]))
                cursor_page = found[0]
            return starts

        starts = search_from(0)

        # 목차 등에서 소제목이 본문보다 먼저 몰려서 잡히는 경우 방지 — 찾은 위치의
        # 상당수가 문서 앞부분 좁은 페이지 구간에 몰려 있으면, 그 구간 다음부터
        # 다시 탐색한다 (LLM/qa_agent/engine.py._find_section_positions와 같은 전략).
        if starts:
            total_pages = len(doc)
            toc_window = max(1, int(total_pages * 0.2))
            clustered = [p for _, p, _ in starts if p <= toc_window]
            if len(starts) >= 3 and len(clustered) >= len(starts) * 0.5:
                retry_start = max(clustered) + 1
                if retry_start < total_pages:
                    retried = search_from(retry_start)
                    if retried:
                        starts = retried

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
