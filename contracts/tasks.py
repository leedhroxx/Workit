import os
import sys
from celery import shared_task

MAX_CLAUSES = 3  # 시연용 조항 수 제한 (CPU 환경)


@shared_task(bind=True)
def analyze_document_task(self, doc_id):
    """AI 분석 비동기 태스크"""
    from contracts.models import ContractDocument, AIReviewResult
    from contracts.utils import extract_text, parse_to_workit

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [os.path.join(BASE_DIR, 'rag'), os.path.join(BASE_DIR, 'data')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    doc = ContractDocument.objects.get(pk=doc_id)

    # 1. 계약서 1~2페이지 요약 표(계약 조건 + 계약당사자) 빈칸 검사
    # 규칙 기반이라 LLM과 완전히 무관하게 항상 먼저 실행한다. 
    # 아래 LLM 파이프라인이 (모델 미연결 등으로) 실패하더라도 이 결과는 그대로 저장·반환되어야 한다.
    blanks = []
    try:
        from contracts.contract_field_checker import check_contract_fields
        blanks = check_contract_fields(doc.file.path)
    except Exception:
        import traceback
        print(f'[analyze_document_task] 계약서 빈칸 검사 실패 (doc_id={doc_id}):\n{traceback.format_exc()}')

    typos: list = []
    legal_issues: list = []

    # 2. RAG + sLLM 법률 조항 검토 (실패해도 위 빈칸 결과는 유지한 채 계속 진행)
    try:
        file_text = extract_text(doc.file.path)
        if not file_text.strip():
            raise ValueError('텍스트 추출 실패')

        clause_positions = {}
        file_path_lower = doc.file.path.lower()

        try:
            from clause_locator import extract_clause_positions

            if file_path_lower.endswith('.pdf'):
                clause_positions = extract_clause_positions(doc.file.path)
            elif file_path_lower.endswith('.hwp'):
                from hwp_converter import convert_hwp_to_pdf
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    converted_pdf = convert_hwp_to_pdf(doc.file.path, tmp_dir)
                    clause_positions = extract_clause_positions(converted_pdf)
        except Exception:
            clause_positions = {}

        from qdrant_client import QdrantClient
        from yoonha_law_rag import (
            load_embed_model,
            load_laws_ref,
            review_contract_jo as review_contract,
            results_to_json,
        )

        embed_model = load_embed_model()
        qdrant_client = QdrantClient(url="http://localhost:6333")
        laws_ref = load_laws_ref()

        clause_results = review_contract(
            contract_text=file_text,
            client=qdrant_client,
            model=embed_model,
            laws_ref=laws_ref,
        )
        rag_results = results_to_json(clause_results)

        del embed_model
        del qdrant_client
        import gc
        gc.collect()

        import re as _re

        def _fallback_keys(num: str) -> list[str]:
            keys = [num]
            m_hang = _re.match(r"(제\d+조(?:의\d+)?제\d+항)", num or "")
            if m_hang and m_hang.group(1) != num:
                keys.append(m_hang.group(1))
            m_jo = _re.match(r"(제\d+조(?:의\d+)?)", num or "")
            if m_jo and m_jo.group(1) not in keys:
                keys.append(m_jo.group(1))
            return keys

        for item in rag_results:
            clause_number = item.get('clause_number')
            pos = None
            for key in _fallback_keys(clause_number):
                pos = clause_positions.get(key)
                if pos:
                    break

            if pos and pos.get('fragments'):
                item['fragments'] = pos['fragments']
                first = pos['fragments'][0]
                item['page'] = first['page']
                item['bbox'] = {
                    'x': first['x'], 'y': first['y'],
                    'width': first['width'], 'height': first['height'],
                }
            else:
                item['fragments'] = None
                item['page'] = None
                item['bbox'] = None

        from jihye_inference import load_model as load_llm_model, predict

        llm_model, tokenizer = load_llm_model()

        filtered = [r for r in rag_results if r.get('law_refs')][:MAX_CLAUSES]
        total = len(filtered)
        done = 0

        inference_results = []
        for item in filtered:
            prediction = predict(
                clause_text=item['clause_text'],
                law_refs=item['law_refs'],
                model=llm_model,
                tokenizer=tokenizer,
            )
            inference_results.append({
                'clause_number': item['clause_number'],
                'clause_text': item['clause_text'],
                'risk_names': item.get('categories', []),
                'page': item.get('page'),
                'bbox': item.get('bbox'),
                'fragments': item.get('fragments'),
                'prediction': prediction,
            })
            done += 1
            self.update_state(
                state='PROGRESS',
                meta={'current': done, 'total': total}
            )

        parsed = parse_to_workit(inference_results)
        typos = parsed['typos']
        legal_issues = parsed['legal_issues']

    except Exception:
        import traceback
        print(
            f'[analyze_document_task] LLM 법률 검토 실패 (doc_id={doc_id}) - '
            f'규칙 기반 빈칸 검사 결과는 그대로 저장/반환합니다:\n{traceback.format_exc()}'
        )

    # LLM 단계가 실패해도 규칙 기반 빈칸 검사 결과(blanks)는 항상 저장·반환한다
    AIReviewResult.objects.update_or_create(
        document=doc,
        defaults={
            'blanks': blanks,
            'typos': typos,
            'legal_issues': legal_issues,
        }
    )

    return {
        'status': 'ok',
        'total': len(legal_issues),
        'blanks': blanks,
        'typos': typos,
        'legal_issues': legal_issues,
    }


# RFP 파싱 태스크 (규칙 기반, LLM 없음)
# 실행 시점 : document_complete_review (이행관리 이관) 직후 비동기
# 파서 : contracts.parsers.parse_rfp (키워드·정규식 기반)
# 결과 : contracts.models.RFPParsedData.parsed_json (RDS)

@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def parse_rfp_task(self, rfp_doc_id: int):
    """
    RFP 문서를 규칙 기반으로 파싱해 RFPParsedData에 저장한다.

    contracts.parsers.parse_rfp() 를 사용하므로 LLM 호출이 없고,
    텍스트 추출 후 즉시 완료된다(보통 1~2초).
    """
    from django.utils import timezone
    from contracts.models import ContractDocument, RFPParsedData
    from contracts.utils import extract_text
    from contracts.parsers import parse_rfp  # ← 규칙 기반 파서

    rfp_doc = ContractDocument.objects.select_related('contract').get(pk=rfp_doc_id)

    parsed, _ = RFPParsedData.objects.get_or_create(document=rfp_doc)
    parsed.parse_status = 'processing'
    parsed.error_message = ''
    parsed.save(update_fields=['parse_status', 'error_message'])

    try:
        text = extract_text(rfp_doc.file.path)
        if not text.strip():
            raise ValueError('RFP 텍스트 추출 실패 — 파일을 확인하세요.')

        result_json = parse_rfp(text)

        found_count = sum(1 for s in result_json.get('sections', {}).values() if s.get('found'))
        total_count = len(result_json.get('sections', {}))

        parsed.parsed_json = result_json
        parsed.parse_status = 'done'
        parsed.parsed_at = timezone.now()
        parsed.save(update_fields=['parsed_json', 'parse_status', 'parsed_at'])

        print(f'[parse_rfp_task] 완료 — doc_id={rfp_doc_id}, 섹션 {found_count}/{total_count} 발견')
        return {'status': 'ok', 'doc_id': rfp_doc_id, 'found': found_count, 'total': total_count}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        parsed.parse_status = 'failed'
        parsed.error_message = err[:2000]
        parsed.save(update_fields=['parse_status', 'error_message'])
        print(f'[parse_rfp_task] 실패 — doc_id={rfp_doc_id}\n{err}')
        raise self.retry(exc=exc)