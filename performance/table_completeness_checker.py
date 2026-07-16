# -*- coding: utf-8 -*-
"""
performance/table_completeness_checker.py

사업수행계획서(PEP) PDF의 표에서, 미리 등록해둔 "이 표의 이 열은 항상
채워져 있어야 한다"는 규칙에 따라 빈 셀을 찾는다.

소제목 단위 QA(LLM/qa_agent)는 소제목 블록 전체를 하나의 텍스트로 보기
때문에, 표 안 셀 하나가 비어 있어도 같은 소제목 안에 다른 내용이 많으면
"정상"으로 통과해버린다. 이 체커는 그 사각지대를 표 단위로 메운다.

표마다 어느 열이 "항상 채워지는 열"인지는 자동으로 판단할 수 없다 — 실제
PEP 문서로 검증해본 결과, 순수 통계(열별 채움 비율)나 PDF 구조(셀 좌표)만
으로는 정상적으로 병합된 열(예: "대분류"처럼 같은 값이 여러 행에 걸쳐 한
번만 표시되는 열)과 실제로 비어 있으면 안 되는 열을 구분할 수 없었다 —
병합 열의 채움 비율(예: 70%)이 실제 빈칸이 있는 열의 채움 비율(예: 62.5%)
보다 오히려 더 높은 경우가 실제로 있었기 때문이다. 그래서 표마다 "이 열은
항상 채워진다"를 TABLE_SPECS에 직접 등록해두는 방식을 쓴다 — LLM/qa_agent의
SectionSpec 키워드 매칭과 같은 설계 방식이다.

표를 식별하는 기준은 헤더 행 전체(공백 제거 후)의 정확한 일치다. 헤더가
조금이라도 다른 문서(다른 회사의 PEP 양식 등)는 그냥 매칭되지 않고 조용히
건너뛴다 — 오탐(false positive)보다 미탐(false negative)이 안전하기 때문.
"""

import re

import pdfplumber

TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}

MIN_DATA_ROWS = 2


def _norm(s: str) -> str:
    """공백(줄바꿈 포함) 제거 — PDF 추출 시 헤더 셀 글자 사이에 공백이나
    줄바꿈이 섞여 들어가는 경우가 흔해서, 이를 무시하고 비교해야 한다."""
    return re.sub(r'\s+', '', s or '')


# 표 이름, 정규화된 헤더(전체 열, 순서대로), 항상 채워져야 하는 열의 인덱스.
TABLE_SPECS = [
    {
        'name': '개발대상업무',
        'header': ['대분류', '중분류', '주요기능'],
        'required': [2],
    },
    {
        'name': '개발 및 운영환경',
        'header': ['구분', '항목', '개발기간', '운영단계(개발후)'],
        'required': [1, 2, 3],
    },
    {
        'name': '총괄추진체계',
        'header': ['구분', '기관/담당자', '역할'],
        'required': [0, 2],
    },
    {
        'name': '사업자 추진체계 업무분장',
        'header': ['구분', '성명', '주요임무'],
        'required': [0, 1, 2],
    },
    {
        'name': '참여인력 총괄표',
        'header': ['성명', '소속', '담당업무', '주요산출물'],
        'required': [0, 1, 2, 3],
    },
    {
        'name': '사업추진절차',
        'header': ['단계(Phase)', '세그먼트(Segment)', '단위업무(Task)', '주요산출물'],
        'required': [0, 1, 2, 3],
    },
    {
        'name': '산출물계획',
        'header': ['산출물명', '주요내용', '제출일정', '제출부수', '유형'],
        'required': [0, 1, 2, 4],
    },
    {
        'name': '공정별 투입인력계획',
        'header': ['작업단계(Task)', '기간', 'PL', 'PM', '백엔드', '프론트엔드', 'AI/인프라', 'QA/이관'],
        'required': [1],
    },
    {
        'name': '보고계획',
        'header': ['보고유형', '주기/일정', '보고대상', '주요내용'],
        'required': [0, 2, 3],
    },
    {
        'name': '표준화 항목',
        'header': ['구분', '항목', '사업수행내역'],
        # '구분'은 행정업무표준/공통서비스 등 여러 행에 걸쳐 한 번만 표시되는
        # 병합열이라 제외 — 항목/사업수행내역만 항상 채워져야 한다.
        'required': [1, 2],
    },
    {
        'name': '품질목표',
        'header': ['품질지표', '목표값'],
        'required': [0, 1],
    },
    {
        'name': '위험관리계획',
        'header': ['위험ID', '위험내용', '등급(확률×영향)', '대응방안'],
        'required': [0, 1, 3],
    },
    {
        'name': '교육계획',
        'header': ['교육과목', '일정', '대상', '기간/방법', '교육내용및지원사항'],
        'required': [0, 2, 4],
    },
    {
        'name': '발주기관 협조요청사항',
        'header': ['구분', '협조내용', '필요시기및세부사항'],
        'required': [0, 1, 2],
    },
]

_NORMALIZED_SPECS = [
    {**spec, '_header_norm': [_norm(h) for h in spec['header']]}
    for spec in TABLE_SPECS
]


def _match_spec(header_row: list) -> dict | None:
    header_norm = [_norm(c) for c in header_row]
    for spec in _NORMALIZED_SPECS:
        if header_norm == spec['_header_norm']:
            return spec
    return None


def _row_label(row: list) -> str:
    for cell in row:
        text = (cell or '').strip()
        if text:
            return text.split('\n')[0]
    return ''


def find_empty_required_cells(pdf_path: str) -> list[dict]:
    """PEP PDF의 표에서 TABLE_SPECS에 등록된 표를 찾아, 항상 채워져야 하는
    열이 비어 있는 셀을 찾는다.

    표가 페이지 경계에서 잘리면(예: 5페이지 끝에서 시작해 6페이지로 이어짐),
    pdfplumber는 페이지별로 표를 따로 반환해서 이어지는 뒷부분에는 헤더 행이
    없다 — 그래서 문서 순서대로 표를 훑으면서, 헤더가 매칭되지 않는 표를
    만나도 직전에 매칭된 표와 열 개수가 같으면 "이어지는 표"로 보고 계속
    같은 규칙을 적용한다.

    반환: [{"page": int, "table": str, "column": str, "row_label": str,
            "message": str, "bbox": {left,top,width,height}(0~100, 페이지 대비 %)}, ...]
    bbox는 tech_apply_checker.py와 같은 방식(페이지 폭/높이 대비 비율)이라
    프런트에서 클릭 시 좌측 뷰어에 그대로 하이라이트할 수 있다.
    """
    results = []
    active_spec = None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    # extract_tables()는 텍스트만 주고 셀 좌표가 없어 하이라이트를 못 만든다 —
                    # find_tables()는 같은 표를 Table 객체로 줘서 row.cells의 셀별 bbox를 그대로 쓸 수 있다.
                    tables = page.find_tables(TABLE_SETTINGS) or []
                except Exception:
                    tables = []
                for table in tables:
                    issues, active_spec = _check_table(table, page, page_number, active_spec)
                    results.extend(issues)
    except Exception:
        return []
    return results


def _check_table(table, page, page_number: int, active_spec: dict | None):
    text_rows = table.extract()
    if not text_rows:
        return [], active_spec

    cell_rows = table.rows
    header, *rows = text_rows
    spec = _match_spec(header)

    if spec:
        data_text = rows
        data_cells = cell_rows[1:]
    elif active_spec and len(header) == len(active_spec['header']):
        # 헤더가 매칭되지 않았지만 직전 표와 열 개수가 같음 — 페이지가 넘어가며
        # 잘린 이어지는 표로 보고, 이번 표의 첫 행도 데이터 행으로 취급한다.
        spec = active_spec
        data_text = text_rows
        data_cells = cell_rows
    else:
        return [], None

    paired = [
        (row, cells) for row, cells in zip(data_text, data_cells)
        if row and any((c or '').strip() for c in row)
    ]

    n_cols = len(spec['header'])
    issues = []
    for row, cellgroup in paired:
        row_label = _row_label(row)
        for col_idx in spec['required']:
            if col_idx >= n_cols or col_idx >= len(row):
                continue
            cell = (row[col_idx] or '').strip()
            if cell:
                continue
            col_name = spec['header'][col_idx]

            # 병합 등으로 이 칸만의 좌표가 없으면(None) 행 전체 영역으로 대신한다.
            cell_bbox = cellgroup.cells[col_idx] if col_idx < len(cellgroup.cells) else None
            x0, top, x1, bottom = cell_bbox or cellgroup.bbox

            issues.append({
                'page': page_number,
                'table': spec['name'],
                'column': col_name,
                'row_label': row_label,
                'message': (
                    f"'{row_label}' 행의 '{col_name}' 칸이 비어 있습니다."
                ),
                'bbox': {
                    'left': round(x0 / page.width * 100, 3),
                    'top': round(top / page.height * 100, 3),
                    'width': round((x1 - x0) / page.width * 100, 3),
                    'height': round((bottom - top) / page.height * 100, 3),
                },
            })
    return issues, spec
