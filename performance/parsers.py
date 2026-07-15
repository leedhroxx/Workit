"""
performance/parsers.py

과업수행계획서(PEP)·사업추진결과보고서(RPT)를 정규식·키워드 기반으로 파싱하고,
RFP 파싱 결과와 구조적으로 비교한다. LLM 없이 동작한다.

PEP 코드 체계: PEP-01 ~ PEP-16 (노션 파싱 코드 기준)
RPT 코드 체계: RPT-01-01 ~ RPT-03-02 (노션 파싱 코드 기준)
"""

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# PEP 섹션 코드 → 탐지 키워드 매핑
# ─────────────────────────────────────────────────────────────────────────────

_PEP_SECTION_KEYWORDS: list[tuple[str, str, list[str]]] = [
    ('PEP-01',    '사업명',                   ['사업명']),
    ('PEP-02',    '사업기간',                  ['사업기간', '사업 기간']),
    ('PEP-03-01', '추진배경',                  ['추진배경', '추진 배경']),
    ('PEP-03-02', '목적',                      ['사업 목적', '추진목적', '목적']),
    ('PEP-04-01', '개발대상업무',               ['개발대상업무', '개발 대상 업무', '개발대상']),
    ('PEP-04-02', '개발 및 운영환경',           ['개발 및 운영환경', '개발환경', '운영환경']),
    # label(2번째 값)은 qa_agent/configs/pep.py의 SectionSpec.title과 반드시 같아야 한다
    # (다르면 qa_agent가 "section_title_mismatch"로 오탐한다). title="기타"/group="사업범위"로
    # 나뉘어 있으므로 label도 "기타"만 쓴다 — 키워드 목록(내용 탐지용)은 그대로 넉넉하게 둔다.
    ('PEP-04-03', '기타',                      ['기타 사업범위', '사업 범위 기타', '다. 기타', '기타']),
    ('PEP-05-01', '총괄추진체계',               ['총괄추진체계', '총괄 추진체계', '추진체계']),
    ('PEP-05-02', '사업자 추진체계',            ['사업자 추진체계', '수행체계']),
    ('PEP-06',    '사업추진절차',               ['사업추진절차', '사업 추진 절차', '추진절차']),
    ('PEP-07',    '산출물계획',                 ['산출물계획', '산출물 계획', '산출물관리']),
    ('PEP-08',    '일정계획',                   ['일정계획', '일정 계획', '추진일정', '개발일정']),
    ('PEP-09',    '공정별 투입인력계획',         ['투입인력', '투입 인력', '인력계획', '인력 계획']),
    ('PEP-10',    '보고계획',                   ['보고계획', '보고 계획']),
    ('PEP-11-01', '표준화 항목',                ['표준화 계획', '표준화항목', '표준화 항목']),
    ('PEP-11-02', '정보화기반표준',              ['정보화기반표준', '정보화기반 표준']),
    ('PEP-11-03', '공공기관 DB표준화 지침',      ['DB표준화', 'DB 표준화', '데이터베이스 표준']),
    ('PEP-11-04', '전자정부 웹사이트 품질관리 지침', ['웹사이트 품질', '웹 품질', '전자정부 표준']),
    ('PEP-12',    '품질관리계획',               ['품질관리계획', '품질 관리 계획', '품질관리 계획', '품질보증계획', '품질 보증 계획']),
    ('PEP-13',    '위험관리계획',               ['위험관리계획', '위험 관리', '리스크 관리']),
    ('PEP-14',    '보안대책',                   ['보안대책', '보안 대책', '보안계획']),
    ('PEP-15',    '교육계획',                   ['교육계획', '교육 계획', '사용자 교육']),
    ('PEP-16',    '발주기관 협조요청사항',       ['협조요청', '협조 요청', '발주기관 협조']),
]


# ─────────────────────────────────────────────────────────────────────────────
# RPT(사업추진결과보고서) 섹션 코드 → 탐지 키워드 매핑
# [별지 제23호 서식] 전자정부지원사업 사업추진 결과보고서 양식(제1장 사업개요 5개절,
# 제2장 사업내용(개발사업) 11개절, 제3장 운영계획 및 발전방향 6개절, 부록 8개 항목)
# 기준. 제목 앞의 "제N장"은 장이 바뀌면 절 번호(제1절 등)가 다시 1부터 시작해 여러
# 장에서 중복되므로, 키워드는 그 장 안에서만 유일한 절 제목(내용어) 위주로 잡는다.
# ─────────────────────────────────────────────────────────────────────────────

_RPT_SECTION_KEYWORDS: list[tuple[str, str, list[str]]] = [
    ('RPT-01-01', '제1절 사업 개요',              ['사업 개요', '사업개요']),
    ('RPT-01-02', '제2절 추진체계 및 조직',        ['추진체계 및 조직']),
    ('RPT-01-03', '제3절 추진경과',               ['추진경과', '추진 경과']),
    ('RPT-01-04', '제4절 주요 산출물',            ['주요 산출물']),
    ('RPT-01-05', '제5절 위험관리 및 변경관리',    ['위험관리 및 변경관리']),

    ('RPT-02-01', '제1절 적용방법론',             ['적용방법론', '적용 방법론']),
    ('RPT-02-02', '제2절 세부 개발내역',          ['세부 개발내역']),
    ('RPT-02-03', '제3절 시스템 구성도',          ['시스템 구성도', '시스템구성도']),
    ('RPT-02-04', '제4절 시스템 연계 결과',        ['시스템 연계 결과']),
    ('RPT-02-05', '제5절 표준화 적용결과',         ['표준화 적용결과', '표준화적용결과']),
    ('RPT-02-06', '제6절 보안 강화 내역',         ['보안 강화 내역']),
    ('RPT-02-07', '제7절 법·제도 정비 실적',       ['법·제도 정비 실적', '법제도 정비 실적', '법제도 정비실적']),
    ('RPT-02-08', '제8절 테스트 수행 결과 요약',   ['테스트 수행 결과 요약']),
    ('RPT-02-09', '제9절 데이터베이스 및 화면 설계 결과', ['데이터베이스 및 화면 설계 결과']),
    ('RPT-02-10', '제10절 데이터 마이그레이션 결과', ['데이터 마이그레이션 결과']),
    ('RPT-02-11', '제11절 비기능요구사항 충족현황', ['비기능요구사항 충족현황']),

    ('RPT-03-01', '제1절 운영조직 및 인력계획',    ['운영조직 및 인력계획']),
    ('RPT-03-02', '제2절 운영예산계획',           ['운영예산계획', '운영 예산계획']),
    ('RPT-03-03', '제3절 단기 발전방향',          ['단기 발전방향']),
    ('RPT-03-04', '제4절 중장기 발전방향',        ['중장기 발전방향']),
    ('RPT-03-05', '제5절 정량적 성과지표 종합',    ['정량적 성과지표 종합']),
    ('RPT-03-06', '제6절 예산 집행 내역',         ['예산 집행 내역']),

    ('RPT-04-01', '산출물 목록',                 ['산출물 목록']),
    ('RPT-04-02', '용어 정리',                   ['용어 정리']),
    ('RPT-04-03', '참고 법령 및 표준',            ['참고 법령 및 표준']),
    ('RPT-04-04', '시범운영 만족도조사 결과',      ['시범운영 만족도조사 결과']),
    ('RPT-04-05', '하자보수 및 향후 일정',        ['하자보수 및 향후 일정']),
    ('RPT-04-06', '외부 감리 및 자체점검 결과',    ['외부 감리 및 자체점검 결과']),
    ('RPT-04-07', '교육 및 기술이전 결과',        ['교육 및 기술이전 결과']),
    ('RPT-04-08', '사업 추진 시사점 및 교훈',      ['사업 추진 시사점 및 교훈', '사업추진 시사점 및 교훈']),
]


def _normalize_for_match(s: str) -> str:
    """공백 제거. PDF 추출 시 '사 업 명'처럼 글자 사이에 공백이 섞여 들어가는
    경우가 흔해서, 공백을 무시하고 키워드를 매칭해야 실제 헤더를 놓치지 않는다."""
    return re.sub(r'\s+', '', s)


def _build_normalized_index(text: str) -> tuple[str, list[int]]:
    """
    공백을 제거한 정규화 텍스트와, 그 정규화 텍스트의 각 글자가 원본 text의
    몇 번째 글자였는지 알려주는 index_map을 만든다. 정규화 텍스트에서 찾은
    위치를 원본 text의 위치로 되돌려야 줄바꿈이 살아있는 원본 그대로 슬라이싱할 수 있다.
    """
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append(ch)
        index_map.append(i)
    return ''.join(chars), index_map


# 목차 탐지 기준값 (둘 다 만족해야 "목차가 있다"고 판단)
_TOC_WINDOW_RATIO = 0.15       # 문서 앞부분 이 비율 이내에 몰려 있어야 목차 후보
_TOC_MIN_CLUSTER_RATIO = 0.5   # 매칭된 소제목의 이 비율 이상이 그 구간에 몰려 있어야 함

_TOC_LISTING_WINDOW = 500      # '목차' 글자 뒤로 이 범위 안에서 나열의 마지막 항목을 찾는다


def _sequential_search(
    norm_text: str, start_from: int, keyword_specs: list[tuple[str, str, list[str]]]
) -> dict[str, tuple[int, str]]:
    """cursor 이후 각 코드의 키워드 중 가장 이른 매치를 순서대로 찾아나간다."""
    positions: dict[str, tuple[int, str]] = {}
    cursor = start_from

    for code, label, keywords in keyword_specs:
        found_idx = -1
        found_kw = ""

        for kw in keywords:
            kw_norm = _normalize_for_match(kw)
            if not kw_norm:
                continue
            idx = norm_text.find(kw_norm, cursor)
            if idx >= 0 and (found_idx == -1 or idx < found_idx):
                found_idx = idx
                found_kw = kw_norm

        if found_idx >= 0:
            positions[code] = (found_idx, found_kw)
            cursor = found_idx + len(found_kw)

    return positions


def _skip_past_toc_listing(
    norm_text: str, keyword_specs: list[tuple[str, str, list[str]]]
) -> int:
    """
    '목차'라는 글자 자체를 찾아, 그 직후 나열되는 목차 항목들 중 마지막 코드
    (keyword_specs의 마지막 항목)가 나오는 위치를 반환한다. 못 찾으면 0.

    표지 요약표(예: 사업명·사업기간을 나열한 첫 페이지 표)는 소제목과 같은 문구가
    목차보다도 먼저, 아주 소수의 코드에서만 중복되기 때문에 아래
    _find_section_positions()의 비율 기반 재탐색(전체 코드의 절반 이상이 몰려
    있어야 함)만으로는 걸러지지 않는다 — 표지 표에 등장하는 항목이 적어서
    "몰려 있다"는 조건 자체를 못 채운다. 반면 '목차'라는 글자 자체는 표지 뒤·
    본문 앞에 항상 있고 그 직후엔 반드시 소제목이 전부 나열되므로, 목차 나열의
    마지막 항목이 나오는 지점까지 한 번에 건너뛰면 표지 표와 목차 나열을
    통째로 스킵할 수 있다.
    """
    toc_marker_idx = norm_text.find(_normalize_for_match('목차'))
    if toc_marker_idx < 0:
        return 0

    _, _, last_keywords = keyword_specs[-1]
    window_end = min(len(norm_text), toc_marker_idx + _TOC_LISTING_WINDOW)
    window = norm_text[toc_marker_idx:window_end]

    listing_last_pos = -1
    for kw in last_keywords:
        kw_norm = _normalize_for_match(kw)
        if not kw_norm:
            continue
        idx = window.find(kw_norm)
        if idx >= 0:
            listing_last_pos = max(listing_last_pos, toc_marker_idx + idx + len(kw_norm))

    return listing_last_pos if listing_last_pos >= 0 else 0


def _find_section_positions(
    text: str, keyword_specs: list[tuple[str, str, list[str]]]
) -> tuple[dict[str, tuple[int, str]], list[int]]:
    """
    원본에서 각 소제목이 실제로 등장하는 위치를 찾는다.

    문서 앞부분에 목차가 있으면(과업수행계획서·결과보고서는 대부분 그렇다) 목차에도
    소제목 문구가 본문과 똑같은 순서로 나열돼 있어서, '직전 위치 다음의 첫 매치'만으로는
    검색 커서가 목차 블록 안에서만 맴돌고 실제 본문까지 못 넘어간다. 게다가 표지에
    사업명·사업기간 같은 요약표가 목차보다도 먼저 나오는 문서도 있다. 그래서 먼저
    _skip_past_toc_listing()으로 '목차' 나열 전체(표지 요약표 포함)를 건너뛴 지점부터
    순차 탐색을 하고, 그래도 매칭된 위치의 상당수가 문서 앞부분 좁은 구간에 몰려
    있으면(= 목차 마커를 못 찾은 다른 형태의 목차로 추정) 그 구간 바로 다음부터
    다시 한번 순차 탐색해서 실제 본문 위치를 잡는다.
    (LLM/qa_agent/engine.py의 _find_section_positions와 같은 전략.)
    """
    norm_text, index_map = _build_normalized_index(text)

    initial_start = _skip_past_toc_listing(norm_text, keyword_specs)
    positions = _sequential_search(norm_text, start_from=initial_start, keyword_specs=keyword_specs)

    doc_len = len(norm_text)
    toc_window = doc_len * _TOC_WINDOW_RATIO
    clustered = [pos for pos, _ in positions.values() if pos <= toc_window]

    if positions and len(clustered) >= len(positions) * _TOC_MIN_CLUSTER_RATIO:
        toc_end = max(clustered)
        retried = _sequential_search(norm_text, start_from=toc_end, keyword_specs=keyword_specs)
        if retried:
            positions = retried

    return positions, index_map


# 두 소제목 키워드 매치 사이에 아직 추적하지 않는 상위 챕터 제목(예: "4. 사업범위")이나
# 하위 항목 기호("가.", "나.")만 있고 실제 내용은 없는 줄이 낀 경우, 그 챕터는 이 파서가
# 아는 코드 목록에 없어서 직전 섹션의 content 끝에 그대로 흡수돼버린다. 특히 직전 섹션이
# 실제로는 본문이 비어 있는 경우, 이 "고아 헤더" 텍스트 때문에 내용이 있는 것처럼 보여
# 빈 섹션 검출을 무력화한다 — 아래에서 이런 줄을 끝에서부터 제거한다.
_ORPHAN_HEADING_LINE = re.compile(r'^\s*(?:\d{1,2}\.\s*\S.{0,30}|\d{1,2}\.|[가-힣]\.\s*\S{0,20}|[가-힣]\.)\s*$')


def _strip_trailing_orphan_headings(content: str, max_lines: int = 3) -> str:
    lines = content.split('\n')
    trimmed = 0
    while lines and trimmed < max_lines and _ORPHAN_HEADING_LINE.match(lines[-1]):
        lines.pop()
        trimmed += 1
    return '\n'.join(lines).strip()


def _parse_by_keywords(text: str, keyword_specs: list[tuple[str, str, list[str]]]) -> dict:
    """키워드 스펙(code, label, keywords 목록)을 기준으로 텍스트를 소제목 단위로 자른다.

    parse_execution_plan()/parse_final_report()가 공유하는 공통 엔진이다.
    """
    positions, index_map = _find_section_positions(text, keyword_specs)

    # 정규화 텍스트 위치 -> 원본 text 위치로 변환, 위치 순으로 정렬.
    # header_start(키워드 매치 시작)는 정렬 기준으로만 쓰고, 실제 content는
    # content_start(키워드 매치가 끝난 바로 다음)부터 잘라서 소제목 자기 텍스트가
    # 본문에 그대로 끼어 들어가지 않게 한다.
    label_by_code = {code: label for code, label, _ in keyword_specs}
    anchor_positions: list[tuple[str, str, int, int]] = []

    for code, (norm_pos, kw_norm) in positions.items():
        header_start = index_map[norm_pos] if norm_pos < len(index_map) else len(text)
        norm_content_start = norm_pos + len(kw_norm)
        content_start = index_map[norm_content_start] if norm_content_start < len(index_map) else len(text)
        anchor_positions.append((code, label_by_code[code], header_start, content_start))

    anchor_positions.sort(key=lambda x: x[2])

    sections: dict[str, Any] = {}

    for i, (code, label, _header_start, content_start) in enumerate(anchor_positions):
        end = anchor_positions[i + 1][2] if i + 1 < len(anchor_positions) else len(text)
        content = text[content_start:end].strip()
        content = _strip_trailing_orphan_headings(content)

        # 여러 모듈/항목을 상세히 다루는 절(예: RPT의 "세부 개발내역")은 실제로
        # 5000자를 넘기도 한다. 너무 낮은 상한은 qa_agent가 "잘린 파싱결과"를
        # 전체 원본과 비교해 유사도가 낮게 나오는 오탐(section_content_mismatch)을
        # 유발하므로, 정상적인 섹션 하나 분량을 넉넉히 담을 수 있는 값으로 올려둔다.
        if len(content) > 8000:
            content = content[:8000] + ' [이하 생략]'

        sections[code] = {
            'label': label,
            'content': content,
            'found': True,
        }

    # 찾지 못한 섹션 채움
    found_codes = {item[0] for item in anchor_positions}
    for code, label, _ in keyword_specs:
        if code not in found_codes:
            sections[code] = {'label': label, 'content': '', 'found': False}

    return sections


def parse_execution_plan(text: str) -> dict:
    """
    과업수행계획서 텍스트를 파싱해 PEP 코드 체계 기반 JSON 반환.

    Args:
        text: extract_text()로 추출된 원문 텍스트

    Returns:
        {
            "PEP-01": {"label": "사업명", "content": "...", "found": True},
            "PEP-02": {"label": "사업기간", "content": "...", "found": True},
            ...
        }
    """
    return _parse_by_keywords(text, _PEP_SECTION_KEYWORDS)


def parse_final_report(text: str) -> dict:
    """
    사업추진결과보고서 텍스트를 파싱해 RPT 코드 체계 기반 JSON 반환.

    Args:
        text: extract_text()로 추출된 원문 텍스트

    Returns:
        {
            "RPT-01-01": {"label": "제1절 개요", "content": "...", "found": True},
            "RPT-01-02": {"label": "제2절 사업의 배경 및 목적", "content": "...", "found": True},
            ...
        }
    """
    return _parse_by_keywords(text, _RPT_SECTION_KEYWORDS)


def to_qa_agent_records(parsed: dict) -> list[dict]:
    """
    parse_execution_plan()의 출력(code -> {label, content, found})을
    LLM/qa_agent.engine.review_section_mapping()이 받는 레코드 리스트
    ([{"section_id": "pep_03_01", "section_title": ..., "content": ...}, ...])로 변환한다.

    found=False(본문에서 못 찾은) 섹션은 넣지 않는다 — 넣으면 content가 빈 문자열이라
    qa_agent가 missing_section 대신 empty_section으로 판정해버린다.
    """
    records = []
    for code, info in parsed.items():
        if not info.get('found'):
            continue
        records.append({
            'section_id': code.lower().replace('-', '_'),
            'section_title': info.get('label', ''),
            'content': info.get('content', ''),
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# RFP ↔ PEP 구조적 비교 (LLM 없음)
# ─────────────────────────────────────────────────────────────────────────────

# RFP 코드 → 대응 PEP 코드 매핑
# 'required': True → 해당 PEP 항목이 없으면 미충족으로 분류
_RFP_TO_PEP_MAPPING: list[dict] = [
    {
        'rfp_code': 'RFP-01-02',
        'rfp_subtitle': '추진배경 및 필요성',
        'target_codes': ['PEP-03-01', 'PEP-03-02'],
        'required': True,
        'description': '사업 목적 및 추진 배경',
        'criteria': ['완전성'],
    },
    {
        'rfp_code': 'RFP-01-04',
        'rfp_subtitle': '사업 범위',
        'target_codes': ['PEP-04-01', 'PEP-04-02', 'PEP-04-03'],
        'required': True,
        'description': '개발 대상 업무 및 환경',
        'criteria': ['완전성', '정확성'],
    },
    {
        'rfp_code': 'RFP-03-02',
        'rfp_subtitle': '추진 체계',
        'target_codes': ['PEP-05-01', 'PEP-05-02'],
        'required': True,
        'description': '사업 추진 체계',
        'criteria': ['완전성'],
    },
    {
        'rfp_code': 'RFP-03-03',
        'rfp_subtitle': '추진일정',
        'target_codes': ['PEP-08'],
        'required': True,
        'description': '일정 계획',
        'criteria': ['완전성'],
    },
    {
        'rfp_code': 'RFP-04-04-01',
        'rfp_subtitle': '기능 요구사항',
        'target_codes': ['PEP-04-01', 'PEP-06'],
        'required': True,
        'description': '기능 요구사항 대응 개발 계획',
        'criteria': ['완전성', '추적성'],
    },
    {
        'rfp_code': 'RFP-04-04-06',
        'rfp_subtitle': '테스트 요구사항',
        'target_codes': ['PEP-07', 'PEP-06'],
        'required': False,
        'description': '테스트 계획 및 산출물',
        'criteria': ['완전성', '추적성'],
    },
    {
        'rfp_code': 'RFP-04-04-07',
        'rfp_subtitle': '보안 요구사항',
        'target_codes': ['PEP-14'],
        'required': True,
        'description': '보안 대책',
        'criteria': ['완전성', '정확성', '검증가능성'],
    },
    {
        'rfp_code': 'RFP-04-04-08',
        'rfp_subtitle': '품질 요구사항',
        'target_codes': ['PEP-12'],
        'required': False,
        'description': '품질 관리 계획',
        'criteria': ['완전성', '정확성', '검증가능성'],
    },
    {
        'rfp_code': 'RFP-04-04-09',
        'rfp_subtitle': '제약사항',
        'target_codes': ['PEP-11-01', 'PEP-11-02', 'PEP-11-03', 'PEP-11-04'],
        'required': False,
        'description': '표준화 및 제약사항 준수 계획',
        'criteria': ['완전성', '검증가능성'],
    },
    {
        'rfp_code': 'RFP-04-04-10',
        'rfp_subtitle': '프로젝트 관리',
        'target_codes': ['PEP-09', 'PEP-10'],
        'required': False,
        'description': '인력 및 보고 계획',
        'criteria': ['완전성'],
    },
    {
        'rfp_code': 'RFP-04-04-11',
        'rfp_subtitle': '프로젝트 지원',
        'target_codes': ['PEP-15', 'PEP-13'],
        'required': False,
        'description': '교육 및 위험 관리 계획',
        'criteria': ['완전성', '정확성', '검증가능성'],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PEP(사업수행계획서) ↔ RPT(사업추진결과보고서) 구조적 비교 (LLM 없음)
#
# RPT의 2단계 비교는 RFP가 아니라 "같은 이행 건의 계획(PEP)이 실제로 이행됐는지"를
# 본다 — RFP 대비 이행 여부는 이미 PEP 쪽 2단계(compare_rfp_and_pep)에서 확인하므로,
# 여기서는 PEP에 적어둔 계획 항목이 RPT의 어느 절에서 실적으로 확인되는지 매핑한다.
# ─────────────────────────────────────────────────────────────────────────────

# PEP 코드 → 대응 RPT 코드 매핑
_PEP_TO_RPT_MAPPING: list[dict] = [
    {
        'pep_code': 'PEP-01',
        'target_codes': ['RPT-01-01'],
        'required': True,
        'description': '사업명 이행 확인',
        'criteria': ['완전성'],
    },
    {
        'pep_code': 'PEP-02',
        'target_codes': ['RPT-01-01'],
        'required': True,
        'description': '사업기간 이행 확인',
        'criteria': ['완전성', '정확성'],
    },
    {
        'pep_code': 'PEP-04-01',
        'target_codes': ['RPT-02-02'],
        'required': True,
        'description': '개발대상업무 이행 결과',
        'criteria': ['완전성', '정확성', '추적성'],
    },
    {
        'pep_code': 'PEP-04-02',
        'target_codes': ['RPT-02-03'],
        'required': True,
        'description': '개발 및 운영환경 구축 결과',
        'criteria': ['완전성', '정확성'],
    },
    {
        'pep_code': 'PEP-05-01',
        'target_codes': ['RPT-01-02'],
        'required': True,
        'description': '총괄추진체계 이행 결과',
        'criteria': ['완전성'],
    },
    {
        'pep_code': 'PEP-05-02',
        'target_codes': ['RPT-01-02'],
        'required': True,
        'description': '사업자 추진체계 이행 결과',
        'criteria': ['완전성'],
    },
    {
        'pep_code': 'PEP-06',
        'target_codes': ['RPT-01-03'],
        'required': True,
        'description': '사업추진절차 이행 결과',
        'criteria': ['완전성', '추적성'],
    },
    {
        'pep_code': 'PEP-07',
        'target_codes': ['RPT-01-04'],
        'required': True,
        'description': '산출물계획 이행 결과',
        'criteria': ['완전성', '추적성'],
    },
    {
        'pep_code': 'PEP-08',
        'target_codes': ['RPT-01-03'],
        'required': True,
        'description': '일정계획 이행 결과',
        'criteria': ['완전성', '추적성'],
    },
    {
        'pep_code': 'PEP-11-01',
        'target_codes': ['RPT-02-05'],
        'required': False,
        'description': '표준화 계획 이행 결과',
        'criteria': ['완전성', '정확성', '검증가능성'],
    },
    {
        'pep_code': 'PEP-12',
        'target_codes': ['RPT-02-08', 'RPT-02-11'],
        'required': False,
        'description': '품질관리계획 이행 결과',
        'criteria': ['완전성', '정확성', '검증가능성'],
    },
    {
        'pep_code': 'PEP-13',
        'target_codes': ['RPT-01-05'],
        'required': True,
        'description': '위험관리계획 이행 결과',
        'criteria': ['완전성', '검증가능성'],
    },
    {
        'pep_code': 'PEP-14',
        'target_codes': ['RPT-02-06'],
        'required': True,
        'description': '보안대책 이행 결과',
        'criteria': ['완전성', '정확성', '검증가능성'],
    },
    {
        'pep_code': 'PEP-15',
        'target_codes': ['RPT-04-07'],
        'required': False,
        'description': '교육계획 이행 결과',
        'criteria': ['완전성', '정확성'],
    },
]

_MIN_CONTENT_LENGTH = 30  # 이 이상이어야 "내용 있음"으로 판단

# 완전히 다른 사업의 문서가 잘못 업로드됐는지 판단하는 기준값.
# (문턱값·단어 추출 근거는 아래 _project_identity_score() 참고)
_PROJECT_IDENTITY_THRESHOLD = 0.4
_PROJECT_ANCHOR_CODES = ['RFP-01-01', 'RFP-01-02']  # 사업명·추진배경 — 그 사업만의 고유명사가 담긴 섹션

_WORD_RE = re.compile(r'[가-힣]{2,}|[A-Za-z]{2,}|\d{2,}')
_GENERIC_WORD_STOPWORDS = {
    "요구사항", "분류", "고유번호", "고유번", "명칭", "세부내용", "정의", "산출정보",
    "제공", "수행", "구성", "적용", "관리", "확인", "작성", "계획", "기능", "시스템",
    "제시", "방안", "위한", "대한", "경우", "사업", "포함", "필요", "진행", "운영",
    "사용", "가능", "이용", "통해", "해당", "관련", "그리고", "이를", "이는", "위해",
    "대해", "구축", "구축사업", "대상", "목표", "계약", "발주", "입찰",
}


def _significant_words(text: str) -> set[str]:
    """한글 2글자 이상·영문/숫자 2자 이상 토큰 중 일반 용어를 제외한 단어 집합."""
    words = _WORD_RE.findall(text or "")
    return {w for w in words if w not in _GENERIC_WORD_STOPWORDS}


def _project_identity_score(rfp_json: dict, target_sections: dict) -> float:
    """
    RFP의 사업명·추진배경(RFP-01-01/02)에 등장하는 고유명사(기관명·시스템명 등)가
    대상 문서(PEP/RPT) 전체에 얼마나 등장하는지로 "같은 사업"인지 판단한다.

    처음에는 매핑된 섹션끼리 문장 유사도(difflib/단어 overlap)를 직접 비교했는데,
    RFP는 요구사항 형식 문장이라 "단위테스트·통합테스트·UAT"처럼 어느 IT 사업에서나
    나오는 공통 용어가 많아서 완전히 다른 사업끼리도 유사도가 비슷하게 높게 나오는
    문제가 있었다(실측: 같은 사업 0.121, 다른 사업 0.117로 구분이 안 됨). 반면
    사업명·추진배경에는 그 사업만의 고유한 기관명·시스템명이 담겨 있어, 이 단어들이
    대상 문서 전체에 등장하는 비율을 보는 편이 훨씬 안정적으로 구분됐다
    (실측: 같은 사업 0.67, 다른 사업 0.24).
    """
    anchor_words: set[str] = set()
    for code in _PROJECT_ANCHOR_CODES:
        section = rfp_json.get('sections', {}).get(code, {})
        anchor_words |= _significant_words(section.get('content', ''))

    if not anchor_words:
        return 1.0  # 판단할 근거가 없으면 페널티를 주지 않는다

    target_text = ' '.join(
        s.get('content', '') for s in target_sections.values() if isinstance(s, dict)
    )
    target_words = _significant_words(target_text)
    if not target_words:
        return 0.0

    return len(anchor_words & target_words) / len(anchor_words)


def _has_content(sections: dict, code: str) -> bool:
    """섹션이 존재하고 글자 수가 충분한지("내용 있음")를 확인한다."""
    section = sections.get(code, {})
    content = section.get('content', '')
    return section.get('found', False) and len(content.strip()) >= _MIN_CONTENT_LENGTH


# ─────────────────────────────────────────────────────────────────────────────
# 코멘트에 쓸 "대제목 > 소제목" 표시명 — 화면에는 섹션 코드를 절대 노출하지 않고
# 사람이 읽는 제목만 보여준다. group(대제목)은 qa_agent/configs/*.py의
# SectionSpec.group과 반드시 같은 값을 써야 한다.
# ─────────────────────────────────────────────────────────────────────────────

_PEP_LABELS: dict[str, str] = {code: label for code, label, _ in _PEP_SECTION_KEYWORDS}
_PEP_CODE_GROUPS: dict[str, str] = {
    'PEP-03-01': '사업목적', 'PEP-03-02': '사업목적',
    'PEP-04-01': '사업범위', 'PEP-04-02': '사업범위', 'PEP-04-03': '사업범위',
    'PEP-05-01': '사업추진체계', 'PEP-05-02': '사업추진체계',
    'PEP-11-01': '표준화 계획', 'PEP-11-02': '표준화 계획',
    'PEP-11-03': '표준화 계획', 'PEP-11-04': '표준화 계획',
}

_RPT_LABELS: dict[str, str] = {code: label for code, label, _ in _RPT_SECTION_KEYWORDS}
_RPT_CODE_GROUPS: dict[str, str] = {
    'RPT-01-01': '제1장 사업개요', 'RPT-01-02': '제1장 사업개요', 'RPT-01-03': '제1장 사업개요',
    'RPT-01-04': '제1장 사업개요', 'RPT-01-05': '제1장 사업개요',
    'RPT-02-01': '제2장 사업내용(개발사업)', 'RPT-02-02': '제2장 사업내용(개발사업)',
    'RPT-02-03': '제2장 사업내용(개발사업)', 'RPT-02-04': '제2장 사업내용(개발사업)',
    'RPT-02-05': '제2장 사업내용(개발사업)', 'RPT-02-06': '제2장 사업내용(개발사업)',
    'RPT-02-07': '제2장 사업내용(개발사업)', 'RPT-02-08': '제2장 사업내용(개발사업)',
    'RPT-02-09': '제2장 사업내용(개발사업)', 'RPT-02-10': '제2장 사업내용(개발사업)',
    'RPT-02-11': '제2장 사업내용(개발사업)',
    'RPT-03-01': '제3장 운영계획 및 발전방향', 'RPT-03-02': '제3장 운영계획 및 발전방향',
    'RPT-03-03': '제3장 운영계획 및 발전방향', 'RPT-03-04': '제3장 운영계획 및 발전방향',
    'RPT-03-05': '제3장 운영계획 및 발전방향', 'RPT-03-06': '제3장 운영계획 및 발전방향',
    'RPT-04-01': '부록', 'RPT-04-02': '부록', 'RPT-04-03': '부록', 'RPT-04-04': '부록',
    'RPT-04-05': '부록', 'RPT-04-06': '부록', 'RPT-04-07': '부록', 'RPT-04-08': '부록',
}

# RFP는 장(Ⅰ~Ⅳ) 구분이 코드 앞자리(RFP-01/02/03/04)에 그대로 대응돼서 표 대신
# 접두어로 바로 유도한다.
_RFP_GROUP_BY_PREFIX: dict[str, str] = {
    'RFP-01': 'Ⅰ.사업개요',
    'RFP-02': 'Ⅱ.시스템현황',
    'RFP-03': 'Ⅲ.사업추진방안',
    'RFP-04': 'Ⅳ.제안요청내용',
}


def _rfp_display_name(rfp_code: str, rfp_subtitle: str) -> str:
    """RFP 섹션의 '대제목 > 소제목' 표시명(코드 없이)."""
    prefix = '-'.join(rfp_code.split('-')[:2])
    group = _RFP_GROUP_BY_PREFIX.get(prefix)
    return f"{group} > {rfp_subtitle}" if group else rfp_subtitle


def _target_display_name(code: str, labels: dict[str, str], groups: dict[str, str]) -> str:
    """PEP/RPT 섹션의 '대제목 > 소제목' 표시명(코드 없이)."""
    title = labels.get(code, code)
    group = groups.get(code)
    return f"{group} > {title}" if group else title


def _compare_rfp_to_target(
    rfp_json: dict,
    target_sections: dict,
    mapping_list: list[dict],
    target_label: str,
    target_titles: dict[str, str],
    target_groups: dict[str, str],
) -> dict:
    """
    RFP 파싱 결과와 대상 문서(PEP 또는 RPT) 파싱 결과를 매핑 테이블 기준으로
    비교하는 공통 엔진. RFP 대비 이행 여부는 PEP 쪽(compare_rfp_and_pep)에서만 확인한다
    — RPT(사업추진결과보고서)는 compare_pep_and_final()로 PEP(계획) 대비 비교한다.

    화면에 그대로 보여줄 자연어 문장(message)을 여기서 만들어서 넘긴다 —
    qa_agent/engine.py가 1단계 QA 코멘트를 백엔드에서 완성해 넘기는 것과 같은
    방식이다. 프론트는 이 문장을 조합하지 않고 그대로 표시하기만 하면 된다.
    코멘트에는 섹션 코드(RFP-XX/PEP-XX 등)를 절대 노출하지 않고, 항상
    "대제목 > 소제목" 형태의 사람이 읽는 제목만 담는다.

    Args:
        target_label: 비교 대상 문서명(화면 표시용). 예: "사업수행계획서", "사업추진결과보고서".
        target_titles: target_code -> 소제목 (예: _PEP_LABELS/_RPT_LABELS).
        target_groups: target_code -> 대제목 (예: _PEP_CODE_GROUPS/_RPT_CODE_GROUPS).

    Returns:
        {
            "overall_score": 85,
            "total_items": 11,
            "satisfied_count": 8,
            "partial_count": 2,
            "unsatisfied_count": 1,
            "satisfied": [...],
            "partial": [...],   # 각 항목에 message 필드 포함
            "unsatisfied": [...],  # 각 항목에 message 필드 포함
        }
    """
    rfp_sections = rfp_json.get('sections', {})

    identity_score = _project_identity_score(rfp_json, target_sections)
    is_different_project = identity_score < _PROJECT_IDENTITY_THRESHOLD

    satisfied = []
    partial = []
    unsatisfied = []

    for mapping in mapping_list:
        rfp_code = mapping['rfp_code']
        rfp_section = rfp_sections.get(rfp_code, {})

        # RFP 섹션 자체가 없으면 건너뜀
        if not rfp_section.get('found', False):
            continue

        target_codes = mapping['target_codes']
        rfp_display = _rfp_display_name(rfp_code, mapping['rfp_subtitle'])
        header = f"RFP '{rfp_display}' 대응 확인 — 관련 분류: {mapping['description']}"

        if is_different_project:
            # 사업명·추진배경부터 다른 사업으로 보이면, 대응 섹션 내용을 볼 것도 없이
            # 전부 불가 처리한다. (매핑된 섹션끼리 문장 유사도를 직접 비교하는 방식은
            # RFP·PEP/RPT의 공통 IT 용어 때문에 신뢰도가 낮아 채택하지 않았다 —
            # 자세한 이유는 _project_identity_score() 참고)
            entry = {
                'rfp_code': rfp_code,
                'rfp_subtitle': mapping['rfp_subtitle'],
                'description': mapping['description'],
                'required': mapping['required'],
                'target_codes_checked': target_codes,
                'target_codes_found': [],
                'target_codes_missing': target_codes,
                'coverage': 0,
                'message': f"{header}\nRFP와 사업명·추진배경이 일치하지 않아 다른 사업의 {target_label}로 보입니다 (충족률 0%)",
            }
            unsatisfied.append(entry)
            continue

        found = [c for c in target_codes if _has_content(target_sections, c)]
        missing = [c for c in target_codes if not _has_content(target_sections, c)]

        coverage = len(found) / len(target_codes) if target_codes else 0
        coverage_pct = round(coverage * 100)

        entry: dict[str, Any] = {
            'rfp_code': rfp_code,
            'rfp_subtitle': mapping['rfp_subtitle'],
            'description': mapping['description'],
            'required': mapping['required'],
            'target_codes_checked': target_codes,
            'target_codes_found': found,
            'target_codes_missing': missing,
            'coverage': coverage_pct,
        }

        if coverage == 1.0:
            # 모든 대응 섹션에 내용 있음
            satisfied.append(entry)
        else:
            missing_display = ', '.join(
                _target_display_name(c, target_titles, target_groups) for c in missing
            )
            if coverage > 0:
                # 일부만 있음 — 사람이 다시 봐야 하는 "검토" 상태
                entry['message'] = (
                    f"{header}\n{target_label}의 '{missing_display}' 항목이 일부 비어 있어 "
                    f"검토가 필요합니다 (충족률 {coverage_pct}%)"
                )
                partial.append(entry)
            else:
                # 대응 섹션이 하나도 없음 — "불가" 상태
                entry['message'] = (
                    f"{header}\n{target_label}의 '{missing_display}' 항목이 비어 있어 "
                    f"보완이 필요합니다 (충족률 {coverage_pct}%)"
                )
                unsatisfied.append(entry)

    if is_different_project:
        overall_score = 0
    else:
        # 점수 계산
        # required 항목 미충족 시 감점 가중치 2, 선택 항목 미충족 시 감점 가중치 1
        total_weight = 0
        earned_weight = 0

        for mapping in mapping_list:
            rfp_code = mapping['rfp_code']
            rfp_section = rfp_sections.get(rfp_code, {})
            if not rfp_section.get('found', False):
                continue

            w = 2 if mapping['required'] else 1
            total_weight += w

            target_codes = mapping['target_codes']
            found = [c for c in target_codes if _has_content(target_sections, c)]
            coverage = len(found) / len(target_codes) if target_codes else 0
            earned_weight += w * coverage

        overall_score = round((earned_weight / total_weight * 100) if total_weight else 0)

    return {
        'overall_score': overall_score,
        'project_mismatch': is_different_project,
        'project_identity_score': round(identity_score * 100),
        'total_items': len(satisfied) + len(partial) + len(unsatisfied),
        'satisfied_count': len(satisfied),
        'partial_count': len(partial),
        'unsatisfied_count': len(unsatisfied),
        'satisfied': satisfied,
        'partial': partial,
        'unsatisfied': unsatisfied,
    }


def compare_rfp_and_pep(rfp_json: dict, pep_json: dict) -> dict:
    """파싱된 RFP JSON과 PEP(사업수행계획서) JSON을 구조적으로 비교한다."""
    return _compare_rfp_to_target(
        rfp_json, pep_json, _RFP_TO_PEP_MAPPING, '사업수행계획서', _PEP_LABELS, _PEP_CODE_GROUPS
    )


def _compare_pep_to_rpt(pep_sections: dict, rpt_sections: dict, mapping_list: list[dict]) -> dict:
    """
    사업수행계획서(PEP, 계획)와 사업추진결과보고서(RPT, 실적)를 매핑 테이블 기준으로
    비교한다 — "계획한 대로 실제로 이행됐는지" 확인.

    RFP 비교(_compare_rfp_to_target)와 달리 PEP·RPT는 같은 이행(Performance) 건에
    속한 문서라서 "다른 사업 문서가 잘못 올라왔는지" 판단하는 project_identity 체크는
    필요 없다.
    """
    satisfied = []
    partial = []
    unsatisfied = []

    for mapping in mapping_list:
        pep_code = mapping['pep_code']
        pep_section = pep_sections.get(pep_code, {})
        if not pep_section.get('found', False):
            continue

        target_codes = mapping['target_codes']
        pep_display = _target_display_name(pep_code, _PEP_LABELS, _PEP_CODE_GROUPS)
        header = f"사업수행계획서 '{pep_display}' 대응 확인 — 관련 분류: {mapping['description']}"

        found = [c for c in target_codes if _has_content(rpt_sections, c)]
        missing = [c for c in target_codes if not _has_content(rpt_sections, c)]

        coverage = len(found) / len(target_codes) if target_codes else 0
        coverage_pct = round(coverage * 100)

        entry: dict[str, Any] = {
            'pep_code': pep_code,
            'description': mapping['description'],
            'required': mapping['required'],
            'target_codes_checked': target_codes,
            'target_codes_found': found,
            'target_codes_missing': missing,
            'coverage': coverage_pct,
        }

        if coverage == 1.0:
            satisfied.append(entry)
        else:
            missing_display = ', '.join(
                _target_display_name(c, _RPT_LABELS, _RPT_CODE_GROUPS) for c in missing
            )
            if coverage > 0:
                entry['message'] = (
                    f"{header}\n사업추진결과보고서의 '{missing_display}' 항목이 일부 비어 있어 "
                    f"검토가 필요합니다 (충족률 {coverage_pct}%)"
                )
                partial.append(entry)
            else:
                entry['message'] = (
                    f"{header}\n사업추진결과보고서의 '{missing_display}' 항목이 비어 있어 "
                    f"보완이 필요합니다 (충족률 {coverage_pct}%)"
                )
                unsatisfied.append(entry)

    # 점수 계산 (required 항목 감점 가중치 2, 선택 항목 가중치 1)
    total_weight = 0
    earned_weight = 0

    for mapping in mapping_list:
        pep_code = mapping['pep_code']
        pep_section = pep_sections.get(pep_code, {})
        if not pep_section.get('found', False):
            continue

        w = 2 if mapping['required'] else 1
        total_weight += w

        target_codes = mapping['target_codes']
        found = [c for c in target_codes if _has_content(rpt_sections, c)]
        coverage = len(found) / len(target_codes) if target_codes else 0
        earned_weight += w * coverage

    overall_score = round((earned_weight / total_weight * 100) if total_weight else 0)

    return {
        'overall_score': overall_score,
        'total_items': len(satisfied) + len(partial) + len(unsatisfied),
        'satisfied_count': len(satisfied),
        'partial_count': len(partial),
        'unsatisfied_count': len(unsatisfied),
        'satisfied': satisfied,
        'partial': partial,
        'unsatisfied': unsatisfied,
    }


def compare_pep_and_final(pep_json: dict, rpt_json: dict) -> dict:
    """파싱된 PEP(사업수행계획서) JSON과 RPT(사업추진결과보고서) JSON을 비교한다 — 계획 대비 이행 여부."""
    return _compare_pep_to_rpt(pep_json, rpt_json, _PEP_TO_RPT_MAPPING)

def collect_llm_compare_items_rfp_pep(rfp_json: dict, pep_json: dict) -> list[dict]:
    """_RFP_TO_PEP_MAPPING을 순회하며 LLM 판정용 원문 발췌를 모은다 (LLM 호출 없음).

    RFP 섹션 자체가 없으면 비교할 게 없으므로 건너뛴다. 대응 PEP 섹션들의 본문을
    이어붙여 pep_excerpt로 쓰고, 없으면 빈 문자열로 둬서 LLM이 "본문에 없음"으로
    정확히 판정하게 한다(억지로 값을 지어내지 않는다).
    """
    rfp_sections = rfp_json.get('sections', {})

    items = []
    for mapping in _RFP_TO_PEP_MAPPING:
        rfp_code = mapping['rfp_code']
        rfp_section = rfp_sections.get(rfp_code, {})
        if not rfp_section.get('found', False):
            continue

        pep_excerpt = '\n\n'.join(
            pep_json[c]['content'] for c in mapping['target_codes']
            if pep_json.get(c, {}).get('found', False) and pep_json[c].get('content', '').strip()
        )

        items.append({
            'rfp_code': rfp_code,
            'description': mapping['description'],
            'required': mapping['required'],
            'criteria': mapping['criteria'],
            'rfp_excerpt': rfp_section.get('content', ''),
            'pep_excerpt': pep_excerpt,
        })
    return items


def collect_llm_compare_items_pep_rpt(pep_json: dict, rpt_json: dict) -> list[dict]:
    """_PEP_TO_RPT_MAPPING을 순회하며 LLM 판정용 원문 발췌를 모은다 (LLM 호출 없음)."""
    items = []
    for mapping in _PEP_TO_RPT_MAPPING:
        pep_code = mapping['pep_code']
        pep_section = pep_json.get(pep_code, {})
        if not pep_section.get('found', False):
            continue

        rpt_excerpt = '\n\n'.join(
            rpt_json[c]['content'] for c in mapping['target_codes']
            if rpt_json.get(c, {}).get('found', False) and rpt_json[c].get('content', '').strip()
        )

        items.append({
            'pep_code': pep_code,
            'description': mapping['description'],
            'required': mapping['required'],
            'criteria': mapping['criteria'],
            'pep_excerpt': pep_section.get('content', ''),
            'rpt_excerpt': rpt_excerpt,
        })
    return items


def merge_llm_verdicts(comparison_json: dict, llm_results: dict, code_key: str) -> dict:
    """구조적 비교 결과(comparison_json)에 LLM 판정(llm_results)을 병합해 재작성한다.

    llm_results: {코드: {'description','required','label','eval'}} — collect_llm_compare_items_*로
    모은 항목에 LLM 판정을 붙인 것. label(충족/검토/불가) 그대로 satisfied/partial/unsatisfied로
    나누고, message를 LLM의 eval 근거로 재작성한다. 점수(퍼센트)는 계산하지 않는다 —
    항목별 충족/검토/불가 개수만 보여준다.

    project_mismatch(다른 사업으로 보이는 경우)라 LLM을 아예 안 돌린 경우엔 이 함수를
    호출하지 않고 기존 구조적 비교 결과를 그대로 쓴다.
    """
    satisfied, partial, unsatisfied = [], [], []

    for code, result in llm_results.items():
        label = result.get('label')
        required = result.get('required', False)
        description = result.get('description', '')
        eval_lines = result.get('eval') or []

        entry = {
            code_key: code,
            'description': description,
            'required': required,
            'llm_label': label,
            'llm_eval': eval_lines,
        }

        if label == '충족':
            satisfied.append(entry)
        elif label == '검토':
            entry['message'] = f"{description}\n" + '\n'.join(eval_lines)
            partial.append(entry)
        else:
            # '불가' 또는 파싱 실패(None)는 전부 미흡으로 — 애매하면 사람이 보게 한다
            entry['message'] = f"{description}\n" + '\n'.join(eval_lines)
            unsatisfied.append(entry)

    merged = dict(comparison_json)
    merged.pop('overall_score', None)  # 점수 안 씀 — 항목별 개수만 사용
    merged.update({
        'total_items': len(satisfied) + len(partial) + len(unsatisfied),
        'satisfied_count': len(satisfied),
        'partial_count': len(partial),
        'unsatisfied_count': len(unsatisfied),
        'satisfied': satisfied,
        'partial': partial,
        'unsatisfied': unsatisfied,
    })
    return merged
