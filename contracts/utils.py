"""
contracts/utils.py

- extract_text(): 업로드 파일에서 텍스트 추출 (PDF / DOCX / TXT)
- parse_to_workit(): RAG+sLLM 결과 → Workit AIReviewResult 형식 변환
"""

import os
import re
import tempfile
from contextlib import contextmanager

_VERDICT_RE = re.compile(r"판정\s*:\s*(\S+)")


@contextmanager
def local_copy(filefield):
    """S3/로컬 공용: FileField를 임시 로컬 파일로 내려받아 경로를 yield. 원본 확장자 유지."""
    ext = os.path.splitext(filefield.name)[1] or '.bin'
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        filefield.open('rb')
        for chunk in filefield.chunks():
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        try:
            filefield.close()
        except Exception:
            pass
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def extract_text(file_path: str) -> str:
    """업로드된 파일 경로를 받아 텍스트 문자열 반환. pdf/hwp만 지원한다.

    accounts.file_validators.ALLOWED_UPLOAD_TYPES가 업로드 시점에 pdf/hwp만
    통과시키므로, 여기 도달하는 파일은 항상 이 둘 중 하나여야 한다. 예전엔
    docx/txt/md/hwpx도 처리했지만, docx는 python-docx가 표(table) 안 내용을
    읽지 못해 표 위주 섹션이 통째로 누락되고 doc/hwpx/xlsx/xls는 애초에
    처리 분기가 없어 빈 문자열만 반환되는 등 실제로는 안전하게 동작하지
    않아 업로드 자체를 막았다.
    """
    ext = file_path.lower().rsplit('.', 1)[-1]

    if ext == 'hwp':
        # 구형 HWP는 pdfplumber로 직접 못 읽으므로
        # LibreOffice(H2Orestart 확장)로 PDF 변환 후 같은 함수에 재귀 호출해 처리한다.
        import sys
        import os
        import tempfile

        rag_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rag')
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)

        from hwp_converter import convert_hwp_to_pdf

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = convert_hwp_to_pdf(file_path, tmp_dir)
            return extract_text(pdf_path)  # 변환된 PDF를 아래 'pdf' 분기로 재처리

    if ext == 'pdf':
        import pdfplumber
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
        return '\n'.join(texts)

    raise ValueError(f"지원하지 않는 파일 형식입니다 (.{ext}). pdf/hwp만 지원합니다.")


def parse_to_workit(inference_results: list) -> dict:
    """
    inference.run_inference() 반환값 →  Workit AIReviewResult 형식으로 변환.

    inference_results 각 항목 구조:
    {
        "clause_number": "제1조",
        "clause_text":   "...",
        "risk_names":    ["손해배상 범위 일방적 제한", ...],
        "prediction":    "위험 조항입니다. 근거: ..."   ← sLLM 출력 텍스트
    }

    반환 형식 (AIReviewResult 모델에 저장):
    {
        "blanks":       [],   # RAG/sLLM 미지원 → 빈 리스트
        "typos":        [],   # RAG/sLLM 미지원 → 빈 리스트
        "legal_issues": [
            {
                "location":      "제1조",
                "original_text": "...",
                "issue":         "손해배상 범위 일방적 제한, ...",
                "legal_ref":     "... (sLLM 판정 전문)"
            },
            ...
        ]
    }

    필터링 기준:
    - 판정이 "일치"/"충족"(정상)인 항목 제외
    - 동일 조항 번호 중복 제거 (첫 번째 위반/누락 항목만 사용)
    """
    legal_issues = []
    seen_locations = set()

    for item in inference_results:
        prediction = (item.get('prediction') or '').strip()
        risk_names = item.get('risk_names') or []
        location = item.get('clause_number', '')

        # sLLM 판정 결과 없는 항목 제외
        if not prediction:
            continue

        # 정상 판정 제외 (계약서='일치', 산출물='충족')
        # 문자열 통짜 매칭 대신 "판정: X" 값을 직접 파싱해서 비교한다 — sLLM 출력의
        # 공백/줄바꿈 변형(예: "판정:일치", 뒤에 다른 말이 붙는 경우)에도 안정적으로 걸러진다.
        verdict_match = _VERDICT_RE.search(prediction)
        verdict = verdict_match.group(1).strip() if verdict_match else ''
        if verdict in ('일치', '충족'):
            continue

        # 동일 조항 번호 중복 제거
        if location in seen_locations:
            continue
        seen_locations.add(location)

        legal_issues.append({
            'location':      location,
            'original_text': item.get('clause_text', '')[:300],
            'issue':         ', '.join(risk_names) if risk_names else '위험 조항 감지',
            'legal_ref':     prediction,
            'page':          item.get('page'),
            'bbox':          item.get('bbox'),
            'fragments':     item.get('fragments'),
        })

    return {
        'blanks':       [],
        'typos':        [],
        'legal_issues': legal_issues,
    }

import re
from datetime import date

def extract_contract_period(text: str):
    """
    ex.) "2026년 6월 1일부터 2026년 7월 31일까지" 또는
         "2025년 03월 01일 ~ 2025년 12월 31일" 패턴 추출
    """
    pattern = (
        r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*"
        r"(?:부터|[~\-])\s*"
        r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(?:까지)?"
    )
    match = re.search(pattern, text)
    if match:
        y1, m1, d1, y2, m2, d2 = map(int, match.groups())
        return date(y1, m1, d1), date(y2, m2, d2)
    return None, None