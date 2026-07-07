# -*- coding: utf-8 -*-
"""
계약서 PDF 1~2페이지의 "항목 : 내용" 요약 표(계약 조건 요약 + 계약당사자)를
룰베이스로 검사해, 내용이 비어 있는 항목을 찾는다. (LLM 미사용)

전제:
- 표는 "라벨|값" 쌍이 반복되는 구조다 (한 행에 1쌍: 라벨|값,
  또는 2쌍: 라벨1|값1|라벨2|값2 — 예: "지연이자율"/"지체상금요율",
  계약당사자 표의 "발주자"/"공급자").
- 항목명은 계약서 양식마다 다를 수 있어 고정 목록을 쓰지 않는다. 표에서
  발견되는 모든 라벨:값 쌍을 대상으로 값이 비어 있는지만 검사한다.
- EXCLUDE_LABELS에 있는 라벨은 값이 비어 있어도 오류로 잡지 않는다
  ('지식재산권'/'검사의 기준및 방법'은 장문 조항형 항목, '기타사항'은
  원래 선택 기재 항목이라 비어 있는 게 정상이기 때문).

좌표(bbox)는 rag/clause_locator.py가 legal_issues에 쓰는 것과 동일하게
PDF 포인트 기준 {x, y(top), width, height}로 반환한다 — 그래야
templates/contracts/document_analyze.html의 jumpToClause()가 그대로 쓸 수 있다.
"""

import pdfplumber

EXCLUDE_LABELS = {"지식재산권", "검사의 기준및 방법", "기타사항"}

TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}


def _clean(cell) -> str:
    return (cell or "").strip()


def check_contract_fields(pdf_path: str, max_pages: int = 2) -> list:
    """
    계약서 PDF 앞 max_pages 페이지에서 라벨:값 표를 찾아 값이 빈 항목을 검사한다.

    PDF가 아니면(hwp/docx 등) 표 좌표 기반 파싱을 지원하지 않으므로 빈 리스트를
    반환한다 (LLM 검토는 별도 파이프라인에서 계속 수행됨).

    Returns:
        AIReviewResult.blanks / document_analyze.html 렌더링과 호환되는
        문제 항목 리스트:
        [{"location": str, "description": str, "text": "",
          "page": int, "bbox": {"x","y","width","height"} | None}, ...]
    """
    if not pdf_path.lower().endswith('.pdf'):
        return []

    blanks: list = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages[:max_pages], start=1):
            for table in page.find_tables(TABLE_SETTINGS):
                grid = table.extract()
                for row_i, row in enumerate(grid):
                    row_cells = table.rows[row_i].cells if row_i < len(table.rows) else []

                    # 한 행에 라벨:값 쌍이 1개 또는 여러 개(짝수 컬럼) 들어있을 수 있음
                    for pair_start in range(0, len(row) - 1, 2):
                        label = _clean(row[pair_start])
                        if not label or label in EXCLUDE_LABELS:
                            continue

                        value = _clean(row[pair_start + 1]) if pair_start + 1 < len(row) else ""
                        if value:
                            continue

                        value_bbox = row_cells[pair_start + 1] if pair_start + 1 < len(row_cells) else None
                        label_bbox = row_cells[pair_start] if pair_start < len(row_cells) else None
                        bbox_src = value_bbox or label_bbox

                        bbox = None
                        if bbox_src:
                            x0, top, x1, bottom = bbox_src
                            bbox = {
                                "x": round(x0, 2),
                                "y": round(top, 2),
                                "width": round(x1 - x0, 2),
                                "height": round(bottom - top, 2),
                            }

                        blanks.append({
                            "location": label,
                            "description": f"'{label}' 항목의 내용이 비어 있습니다.",
                            "text": "",
                            "page": page_no,
                            "bbox": bbox,
                        })

    return blanks
