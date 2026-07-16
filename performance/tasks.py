import logging

from celery import shared_task
from django.utils import timezone
from datetime import timedelta
import os

SITE_URL = os.environ.get('SITE_URL', 'http://localhost:8000')

# RunPod 원격 GPU 추론 서버 URL (contracts.tasks와 동일 패턴). 비어있으면 로컬 CPU로 폴백.
LLM_SERVER_URL = os.environ.get('LLM_SERVER_URL', '').strip() or None


def _rag_path():
    """rag/ 폴더를 import 경로에 추가한다 (로컬 CPU 폴백 시 inference를 쓰기 위해)."""
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_dir = os.path.join(base, 'rag')
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)


def _get_pep_predictor():
    """PEP(RFP↔사업수행계획서) 판정 함수를 하나 만들어 반환한다.

    RunPod가 있으면 매번 원격 호출, 없으면 로컬 모델을 딱 한 번만 로드해서 재사용한다
    (항목마다 매번 8B 모델을 새로 로드하면 너무 느려진다).
    """
    if LLM_SERVER_URL:
        from remote_inference_client import remote_compare_pep
        return lambda item: remote_compare_pep(item, LLM_SERVER_URL)
    _rag_path()
    from inference import load_model, predict_pep
    model, tokenizer = load_model()
    return lambda item: predict_pep(item, model, tokenizer)


def _get_rpt_predictor():
    """RPT(PEP↔사업추진결과보고서) 판정 함수를 하나 만들어 반환한다. _get_pep_predictor와 동일 패턴."""
    if LLM_SERVER_URL:
        from remote_inference_client import remote_compare_rpt
        return lambda item: remote_compare_rpt(item, LLM_SERVER_URL)
    _rag_path()
    from inference import load_model, predict_rpt
    model, tokenizer = load_model()
    return lambda item: predict_rpt(item, model, tokenizer)


logger = logging.getLogger(__name__)

TARGET_TYPES = ['tech_apply', 'final']

@shared_task
def check_deadlines():
    from performance.models import Deliverable, Notification

    today = timezone.now().date()

    for days in [7, 3, 1]:
        target_date = today + timedelta(days=days)

        deliverables = Deliverable.objects.filter(
            due_date=target_date,
            status='pending',
            deliverable_type__in=TARGET_TYPES,
        ).select_related('performance__contract__created_by')

        for d in deliverables:
            contract = d.performance.contract
            user = contract.created_by
            label = d.get_deliverable_type_display()
            project = contract.project_name

            if not user.notification_enabled:
                continue

            message = f"[{project}] {label} 제출 마감 {days}일 전입니다."

            # 사이트 내 알림
            Notification.objects.create(
                user=user,
                message=message,
                url='/performance/',
            )

            # 이메일 발송
            if user.email:
                try:
                    from django.core.mail import send_mail
                    send_mail(
                        subject=f'[Workit] {label} 마감 {days}일 전 알림',
                        message=(
                            f"안녕하세요, {user.korean_name()}님.\n\n"
                            f"'{project}'의 {label} 제출 마감일이 "
                            f"{days}일 후({target_date.strftime('%Y년 %m월 %d일')})입니다.\n\n"
                            f"Workit에 접속하여 산출물을 등록해 주세요.\n"
                            f"{SITE_URL}/performance/"
                        ),
                        from_email=None,              # ← DEFAULT_FROM_EMAIL 사용
                        recipient_list=[user.email],  # ← 각 사용자 이메일로
                        fail_silently=True,
                    )
                except Exception as e:
                    print(f"[알림] 이메일 발송 실패: {e}")


# 과업수행계획서 파싱 태스크 (규칙 기반, LLM 없음)

# 실행 시점 : 과업수행계획서 파일 업로드 직후 비동기
# 파서 : performance.parsers.parse_execution_plan (키워드·정규식 기반)
# 결과 : performance.models.ExecutionPlanParsedData.parsed_json (RDS)

@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def parse_execution_plan_task(self, deliverable_id: int):
    """
    과업수행계획서 파일을 규칙 기반으로 파싱해 ExecutionPlanParsedData에 저장한다.

    performance.parsers.parse_execution_plan() 를 사용하므로 LLM 없이 동작한다.
    호출 시점: 과업수행계획서 파일 업로드 직후.
    """
    import os
    import sys

    from performance.models import Deliverable, ExecutionPlanParsedData
    from contracts.utils import extract_text, local_copy
    from performance.parsers import parse_execution_plan, to_qa_agent_records

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llm_dir = os.path.join(BASE_DIR, 'LLM')
    if llm_dir not in sys.path:
        sys.path.insert(0, llm_dir)

    deliverable = Deliverable.objects.select_related('performance__contract').get(pk=deliverable_id)

    parsed, _ = ExecutionPlanParsedData.objects.get_or_create(deliverable=deliverable)
    parsed.parse_status = 'processing'
    parsed.error_message = ''
    parsed.save(update_fields=['parse_status', 'error_message'])

    try:
        if not deliverable.file:
            raise ValueError('과업수행계획서 파일이 없습니다.')

        with local_copy(deliverable.file) as _local:
            text = extract_text(_local)

            # 표 안 개별 셀 완전성 검사. qa_agent는 소제목 블록 전체를 하나의 텍스트로 보기 때문에, 
            # 표 안 특정 칸 하나가 비어 있어도 같은 소제목에 다른 내용이 많으면 통과시켜버린다 — 
            # 그 사각지대를 표 단위로 보완한다. PDF가 아닌 원본(HWP 등)은 건너뛴다.
            sparse_cells = []
            if _local.lower().endswith('.pdf'):
                try:
                    from performance.table_completeness_checker import find_empty_required_cells
                    sparse_cells = find_empty_required_cells(_local)
                except Exception:
                    import traceback
                    print(f'[parse_execution_plan_task] 표 빈칸 검사 실패 — deliverable_id={deliverable_id}\n{traceback.format_exc()}')

            # 소제목 클릭 시 좌측 뷰어에 하이라이트하기 위한 위치 정보.
            # PDF만 지원 — HWP는 위 표 검사와 동일하게 건너뛴다(하이라이트 없이 텍스트만 표시됨).
            section_bboxes = {}
            if _local.lower().endswith('.pdf'):
                try:
                    from qa_agent.bbox_locator import locate_section_bboxes
                    section_bboxes = locate_section_bboxes(_local, 'pep')
                except Exception:
                    import traceback
                    print(f'[parse_execution_plan_task] 소제목 위치 탐색 실패 — deliverable_id={deliverable_id}\n{traceback.format_exc()}')

        if not text.strip():
            raise ValueError('과업수행계획서 텍스트 추출 실패 — 파일을 확인하세요.')

        result_json = parse_execution_plan(text)

        found_count = sum(1 for s in result_json.values() if s.get('found'))
        total_count = len(result_json)

        # 소제목 매핑 QA 검수 (LLM/qa_agent). 
        # 검수 자체가 실패해도 파싱 성공은 그대로 살려야 하므로 별도 try/except로 감싸고, 실패 시 리포트만 비워둔다.
        qa_report = {}
        try:
            from qa_agent.engine import review_section_mapping

            qa_report = review_section_mapping(
                original_text=text,
                parsed_sections=to_qa_agent_records(result_json),
                document_type='pep',
            )
        except Exception:
            import traceback
            print(f'[parse_execution_plan_task] QA 검수 실패 — deliverable_id={deliverable_id}\n{traceback.format_exc()}')

        if qa_report and section_bboxes:
            qa_report['section_bboxes'] = section_bboxes

        if qa_report and sparse_cells:
            for i, cell in enumerate(sparse_cells):
                code = f'TABLE-{i + 1}'
                location = f"{cell['table']} 표 ({cell['page']}페이지)"
                qa_report.setdefault('issues', []).append({
                    'issue_type': 'empty_table_cell', 'code': code, 'title': cell['table'],
                    'location': location, 'message': cell['message'], 'sample': '',
                    'severity': 'review', 'related_code': '', 'related_title': '',
                    'parsed_sample': '', 'expected_missing': [], 'similarity_score': 0.0,
                })
                qa_report.setdefault('comments', []).append({
                    'code': code, 'title': cell['table'], 'location': location,
                    'message': cell['message'],
                })
            qa_report['review_status'] = 'FAIL'
            qa_report['passed'] = False
            qa_report['can_auto_proceed'] = False

        qa_issues = qa_report.get('issues', [])
        from performance.models import AIAnalysisLog
        if qa_issues:
            logger.info(
                '[QA] 사업수행계획서(deliverable_id=%s) 1단계 QA 이슈 %d건 발견 (review_status=%s): %s',
                deliverable_id, len(qa_issues), qa_report.get('review_status'),
                [issue.get('issue_type') for issue in qa_issues],
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_issue', issue_count=len(qa_issues),
                detail={'review_status': qa_report.get('review_status'),
                        'issue_types': [issue.get('issue_type') for issue in qa_issues]},
            )
        else:
            logger.info(
                '[QA] 사업수행계획서(deliverable_id=%s) 1단계 QA 이슈 없음 (review_status=%s)',
                deliverable_id, qa_report.get('review_status'),
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_ok',
                detail={'review_status': qa_report.get('review_status')},
            )

        parsed.parsed_json = result_json
        parsed.qa_report = qa_report
        parsed.parse_status = 'done'
        parsed.parsed_at = timezone.now()
        parsed.save(update_fields=['parsed_json', 'qa_report', 'parse_status', 'parsed_at'])

        print(
            f'[parse_execution_plan_task] 완료 — deliverable_id={deliverable_id}, '
            f'섹션 {found_count}/{total_count} 발견, QA={qa_report.get("review_status", "N/A")}'
        )
        return {'status': 'ok', 'deliverable_id': deliverable_id, 'found': found_count, 'total': total_count,
                'qa_status': qa_report.get('review_status')}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        parsed.parse_status = 'failed'
        parsed.error_message = err[:2000]
        parsed.save(update_fields=['parse_status', 'error_message'])
        print(f'[parse_execution_plan_task] 실패 — deliverable_id={deliverable_id}\n{err}')
        raise self.retry(exc=exc)


# RFP ↔ 과업수행계획서 비교 태스크 (구조적 비교, LLM 없음)

# 실행 시점 : 비교 버튼 클릭 → rfp_compare view
# 비교 로직  : performance.parsers.compare_rfp_and_pep (코드 매핑 기반)
# 결과 : performance.models.RFPComparisonResult.comparison_json (RDS)

@shared_task(bind=True, max_retries=1, default_retry_delay=10)
def compare_rfp_execution_plan_task(self, performance_id: int):
    """
    파싱된 RFP와 과업수행계획서를 비교해 RFPComparisonResult에 저장한다.

    1) performance.parsers.compare_rfp_and_pep()로 구조적 비교(대응 섹션 존재 여부,
       다른 사업인지 등)를 먼저 하고,
    2) project_mismatch가 아니면 각 매핑 항목을 kanana LLM(PEP 태스크)에 실제로
       판정시켜 충족/검토/불가로 재분류한다 (performance.parsers.merge_llm_verdicts).

    LLM_SERVER_URL이 설정돼 있으면 RunPod 원격 추론, 없으면 로컬 CPU로 폴백한다.
    """
    from performance.models import Performance, ExecutionPlanParsedData, RFPComparisonResult, AIPerfLog
    from contracts.models import RFPParsedData
    from performance.parsers import compare_rfp_and_pep, collect_llm_compare_items_rfp_pep, merge_llm_verdicts

    performance = Performance.objects.select_related('contract').get(pk=performance_id)
    contract = performance.contract

    # 전제 조건 확인

    rfp_doc = contract.documents.filter(doc_type='rfp').first()
    if not rfp_doc:
        return {'status': 'error', 'message': 'RFP 문서가 없습니다.'}

    try:
        rfp_parsed = rfp_doc.rfp_parsed
    except RFPParsedData.DoesNotExist:
        return {'status': 'error', 'message': 'RFP가 아직 파싱되지 않았습니다.'}

    if rfp_parsed.parse_status != 'done':
        return {'status': 'error', 'message': f'RFP 파싱 상태: {rfp_parsed.parse_status}'}

    execution_plan = performance.deliverables.filter(deliverable_type='kickoff').first()
    if not execution_plan:
        return {'status': 'error', 'message': '과업수행계획서 산출물이 없습니다.'}

    try:
        pep_parsed = execution_plan.parsed_data
    except ExecutionPlanParsedData.DoesNotExist:
        return {'status': 'error', 'message': '과업수행계획서가 아직 파싱되지 않았습니다.'}

    if pep_parsed.parse_status != 'done':
        return {'status': 'error', 'message': f'과업수행계획서 파싱 상태: {pep_parsed.parse_status}'}

    # 구조적 비교 + LLM 판정

    try:
        with AIPerfLog.track('kickoff_analyze', performance_id=performance_id) as ctx:
            comparison_json = compare_rfp_and_pep(rfp_parsed.parsed_json, pep_parsed.parsed_json)

            if not comparison_json.get('project_mismatch'):
                items = collect_llm_compare_items_rfp_pep(rfp_parsed.parsed_json, pep_parsed.parsed_json)
                total = len(items)
                llm_results = {}
                predict_fn = _get_pep_predictor()

                for i, item in enumerate(items):
                    result = predict_fn(item)
                    llm_results[item['rfp_code']] = {
                        'description': item['description'],
                        'required': item['required'],
                        'target_codes': item.get('target_codes', []),
                        'label': result.get('label'),
                        'eval': result.get('eval'),
                    }
                    self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': total})

                comparison_json = merge_llm_verdicts(comparison_json, llm_results, 'rfp_code')

            ctx['item_count'] = len(comparison_json.get('satisfied', [])) + len(comparison_json.get('partial', [])) + len(comparison_json.get('unsatisfied', []))

            RFPComparisonResult.objects.update_or_create(
                performance=performance,
                defaults={
                    'rfp_parsed': rfp_parsed,
                    'execution_plan_parsed': pep_parsed,
                    'comparison_json': comparison_json,
                    'status': 'done',
                },
            )

            print(
                f'[compare_rfp_execution_plan_task] 완료 — performance_id={performance_id}, '
                f'충족={comparison_json.get("satisfied_count")} 검토={comparison_json.get("partial_count")} '
                f'불가={comparison_json.get("unsatisfied_count")}'
            )

        # 화면을 나가 있어도 분석은 celery worker에서 계속 진행되므로, 완료 시점에 알림을 띄워 사용자가 다시 들어와서 결과를 확인할 수 있게 한다.
        try:
            from performance.models import Notification

            owner = contract.created_by
            if owner.notification_enabled:
                Notification.objects.create(
                    user=owner,
                    message=f'대응 비교 분석이 완료되었습니다: {contract.project_name} 과업수행계획서',
                    url=f'/performance/deliverable/{execution_plan.id}/analyze/',
                )
        except Exception:
            import traceback
            print(f'[compare_rfp_execution_plan_task] 완료 알림 생성 실패 (performance_id={performance_id}):\n{traceback.format_exc()}')

        return {'status': 'ok', 'performance_id': performance_id}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        print(f'[compare_rfp_execution_plan_task] 실패 — performance_id={performance_id}\n{err}')
        # status가 'processing'에 그대로 머무르면 deliverable_compare_rfp의 중복 실행 방지 로직이
        # COMPARE_STALE_MINUTES(60분) 동안 이 실패한 task_id를 그대로 재사용해버려서,
        # 사용자가 재시도해도 같은 에러가 계속 뜨는 것처럼 보인다 — 즉시 재시도 가능하도록 실패로 표시한다.
        try:
            RFPComparisonResult.objects.filter(performance_id=performance_id).update(status='failed')
        except Exception:
            pass
        raise self.retry(exc=exc)



# 사업추진결과보고서 파싱 태스크 (규칙 기반, LLM 없음)

# 실행 시점 : 산출물 분석 화면에서 "분석 시작" 클릭
# 파서 : performance.parsers.parse_final_report (키워드·정규식 기반)
# 결과 : performance.models.FinalReportParsedData.parsed_json (RDS)


@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def parse_final_report_task(self, deliverable_id: int):
    """
    사업추진결과보고서 파일을 규칙 기반으로 파싱해 FinalReportParsedData에 저장한다.

    performance.parsers.parse_final_report() 를 사용하므로 LLM 없이 동작한다.
    parse_execution_plan_task와 동일한 구조 — RPT 코드 체계로 파싱 후 qa_agent로
    소제목 매핑 QA 검수까지 함께 수행한다.
    """
    import os
    import sys

    from performance.models import Deliverable, FinalReportParsedData
    from contracts.utils import extract_text, local_copy
    from performance.parsers import parse_final_report, to_qa_agent_records

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llm_dir = os.path.join(BASE_DIR, 'LLM')
    if llm_dir not in sys.path:
        sys.path.insert(0, llm_dir)

    deliverable = Deliverable.objects.select_related('performance__contract').get(pk=deliverable_id)

    parsed, _ = FinalReportParsedData.objects.get_or_create(deliverable=deliverable)
    parsed.parse_status = 'processing'
    parsed.error_message = ''
    parsed.save(update_fields=['parse_status', 'error_message'])

    try:
        if not deliverable.file:
            raise ValueError('사업추진결과보고서 파일이 없습니다.')

        with local_copy(deliverable.file) as _local:
            text = extract_text(_local)

            # 소제목 클릭 시 좌측 뷰어에 하이라이트하기 위한 위치 정보.
            # PDF만 지원 — HWP는 하이라이트 없이 텍스트만 표시된다.
            section_bboxes = {}
            if _local.lower().endswith('.pdf'):
                try:
                    from qa_agent.bbox_locator import locate_section_bboxes
                    section_bboxes = locate_section_bboxes(_local, 'rpt')
                except Exception:
                    import traceback
                    print(f'[parse_final_report_task] 소제목 위치 탐색 실패 — deliverable_id={deliverable_id}\n{traceback.format_exc()}')

        if not text.strip():
            raise ValueError('사업추진결과보고서 텍스트 추출 실패 — 파일을 확인하세요.')

        result_json = parse_final_report(text)

        found_count = sum(1 for s in result_json.values() if s.get('found'))
        total_count = len(result_json)

        # 소제목 매핑 QA 검수 (LLM/qa_agent). 
        # 검수 자체가 실패해도 파싱 성공은 그대로 살려야 하므로 별도 try/except로 감싸고, 실패 시 리포트만 비워둔다.
        qa_report = {}
        try:
            from qa_agent.engine import review_section_mapping

            qa_report = review_section_mapping(
                original_text=text,
                parsed_sections=to_qa_agent_records(result_json),
                document_type='rpt',
            )
        except Exception:
            import traceback
            print(f'[parse_final_report_task] QA 검수 실패 — deliverable_id={deliverable_id}\n{traceback.format_exc()}')

        if qa_report and section_bboxes:
            qa_report['section_bboxes'] = section_bboxes

        qa_issues = qa_report.get('issues', [])
        from performance.models import AIAnalysisLog
        if qa_issues:
            logger.info(
                '[QA] 사업추진결과보고서(deliverable_id=%s) 1단계 QA 이슈 %d건 발견 (review_status=%s): %s',
                deliverable_id, len(qa_issues), qa_report.get('review_status'),
                [issue.get('issue_type') for issue in qa_issues],
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_issue', issue_count=len(qa_issues),
                detail={'review_status': qa_report.get('review_status'),
                        'issue_types': [issue.get('issue_type') for issue in qa_issues]},
            )
        else:
            logger.info(
                '[QA] 사업추진결과보고서(deliverable_id=%s) 1단계 QA 이슈 없음 (review_status=%s)',
                deliverable_id, qa_report.get('review_status'),
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_ok',
                detail={'review_status': qa_report.get('review_status')},
            )

        parsed.parsed_json = result_json
        parsed.qa_report = qa_report
        parsed.parse_status = 'done'
        parsed.parsed_at = timezone.now()
        parsed.save(update_fields=['parsed_json', 'qa_report', 'parse_status', 'parsed_at'])

        print(
            f'[parse_final_report_task] 완료 — deliverable_id={deliverable_id}, '
            f'섹션 {found_count}/{total_count} 발견, QA={qa_report.get("review_status", "N/A")}'
        )
        return {'status': 'ok', 'deliverable_id': deliverable_id, 'found': found_count, 'total': total_count,
                'qa_status': qa_report.get('review_status')}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        parsed.parse_status = 'failed'
        parsed.error_message = err[:2000]
        parsed.save(update_fields=['parse_status', 'error_message'])
        print(f'[parse_final_report_task] 실패 — deliverable_id={deliverable_id}\n{err}')
        raise self.retry(exc=exc)


# PEP(사업수행계획서) ↔ 사업추진결과보고서 비교 태스크 (구조적 비교, LLM 없음)

# RFP 대비 이행 여부는 PEP 쪽(compare_rfp_execution_plan_task)에서 이미 확인하므로,
# 여기서는 "계획(PEP)한 대로 실제로 이행됐는지"를 PEP 대비로 비교한다.

# 실행 시점 : QA 검수 결과 확인 후 "그대로 진행" 버튼 클릭
# 비교 로직 : performance.parsers.compare_pep_and_final (코드 매핑 기반)
# 결과 : performance.models.PEPFinalComparisonResult.comparison_json (RDS)

@shared_task(bind=True, max_retries=1, default_retry_delay=10)
def compare_pep_final_report_task(self, performance_id: int):
    """
    파싱된 사업수행계획서(PEP)와 사업추진결과보고서(RPT)를 비교해 PEPFinalComparisonResult에 저장한다.

    compare_rfp_execution_plan_task와 동일한 2단계 구조: 구조적 비교 → LLM(RPT 태스크) 판정.
    PEP·RPT는 같은 이행 건에 속한 문서라 project_mismatch 체크는 필요 없다.
    """
    from performance.models import Performance, ExecutionPlanParsedData, FinalReportParsedData, PEPFinalComparisonResult, AIPerfLog
    from performance.parsers import compare_pep_and_final, collect_llm_compare_items_pep_rpt, merge_llm_verdicts

    performance = Performance.objects.select_related('contract').get(pk=performance_id)

    # 전제 조건 확인 

    kickoff_doc = performance.deliverables.filter(deliverable_type='kickoff').first()
    if not kickoff_doc:
        return {'status': 'error', 'message': '과업수행계획서 산출물이 없습니다.'}

    try:
        pep_parsed = kickoff_doc.parsed_data
    except ExecutionPlanParsedData.DoesNotExist:
        return {'status': 'error', 'message': '과업수행계획서가 아직 파싱되지 않았습니다.'}

    if pep_parsed.parse_status != 'done':
        if pep_parsed.parse_status == 'pending':
            message = '과업수행계획서가 파일 변경 후 아직 재분석되지 않았습니다. 이행관리에서 과업수행계획서를 먼저 분석해주세요.'
        elif pep_parsed.parse_status == 'processing':
            message = '과업수행계획서를 아직 분석 중입니다. 잠시 후 다시 시도해주세요.'
        else:
            message = f'과업수행계획서 파싱 상태: {pep_parsed.parse_status}'
        return {'status': 'error', 'message': message}

    final_doc = performance.deliverables.filter(deliverable_type='final').first()
    if not final_doc:
        return {'status': 'error', 'message': '사업추진결과보고서 산출물이 없습니다.'}

    try:
        final_parsed = final_doc.final_parsed_data
    except FinalReportParsedData.DoesNotExist:
        return {'status': 'error', 'message': '사업추진결과보고서가 아직 파싱되지 않았습니다.'}

    if final_parsed.parse_status != 'done':
        return {'status': 'error', 'message': f'사업추진결과보고서 파싱 상태: {final_parsed.parse_status}'}

    # ── 구조적 비교 + LLM 판정 ─────────────────────────────────────────────

    try:
        with AIPerfLog.track('final_analyze', performance_id=performance_id) as ctx:
            comparison_json = compare_pep_and_final(pep_parsed.parsed_json, final_parsed.parsed_json)

            items = collect_llm_compare_items_pep_rpt(pep_parsed.parsed_json, final_parsed.parsed_json)
            total = len(items)
            llm_results = {}
            predict_fn = _get_rpt_predictor()

            for i, item in enumerate(items):
                result = predict_fn(item)
                llm_results[item['pep_code']] = {
                    'description': item['description'],
                    'required': item['required'],
                    'target_codes': item.get('target_codes', []),
                    'label': result.get('label'),
                    'eval': result.get('eval'),
                }
                self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': total})

            comparison_json = merge_llm_verdicts(comparison_json, llm_results, 'pep_code')
            ctx['item_count'] = total

            PEPFinalComparisonResult.objects.update_or_create(
                performance=performance,
                defaults={
                    'execution_plan_parsed': pep_parsed,
                    'final_report_parsed': final_parsed,
                    'comparison_json': comparison_json,
                    'status': 'done',
                },
            )

            print(
                f'[compare_pep_final_report_task] 완료 — performance_id={performance_id}, '
                f'충족={comparison_json.get("satisfied_count")} 검토={comparison_json.get("partial_count")} '
                f'불가={comparison_json.get("unsatisfied_count")}'
            )

        # 화면을 나가 있어도 분석은 celery worker에서 계속 진행되므로, 완료 시점에
        # 알림을 띄워 사용자가 다시 들어와서 결과를 확인할 수 있게 한다.
        try:
            from performance.models import Notification

            owner = performance.contract.created_by
            if owner.notification_enabled:
                Notification.objects.create(
                    user=owner,
                    message=f'대응 비교 분석이 완료되었습니다: {performance.contract.project_name} 사업추진결과보고서',
                    url=f'/performance/deliverable/{final_doc.id}/analyze/',
                )
        except Exception:
            import traceback
            print(f'[compare_pep_final_report_task] 완료 알림 생성 실패 (performance_id={performance_id}):\n{traceback.format_exc()}')

        return {'status': 'ok', 'performance_id': performance_id}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        print(f'[compare_pep_final_report_task] 실패 — performance_id={performance_id}\n{err}')
        # status가 'processing'에 그대로 머무르면 deliverable_compare_rfp의 중복 실행 방지 로직이
        # COMPARE_STALE_MINUTES(60분) 동안 이 실패한 task_id를 그대로 재사용해버려서,
        # 사용자가 재시도해도 같은 에러가 계속 뜨는 것처럼 보인다 — 즉시 재시도 가능하도록 실패로 표시한다.
        try:
            PEPFinalComparisonResult.objects.filter(performance_id=performance_id).update(status='failed')
        except Exception:
            pass
        raise self.retry(exc=exc)



# 사업수행계획서 → 산출물 일정 자동 반영 태스크 (규칙 기반, LLM 없음)
# 실행 시점 : 사업수행계획서(kickoff) 파일 업로드 직후 비동기
# 파서 : performance.deliverable_date_extractor.parse_output_plan (표 파싱)
# 결과 : 같은 Performance의 tech_apply/final Deliverable.due_date 자동 채움
#             (이미 수동 입력된 값은 덮어쓰지 않음)

@shared_task(bind=True, max_retries=1, default_retry_delay=10)
def sync_deliverable_dates_from_kickoff_task(self, deliverable_id: int):
    """
    사업수행계획서(kickoff) Deliverable을 파싱해 그 안의 '산출물계획' 표에서
    기술적용결과표(tech_apply)/사업추진결과보고서(final)의 제출일자를 찾아
    같은 Performance의 Deliverable.due_date에 자동 반영한다.
    """
    from performance.models import Deliverable
    from performance.deliverable_date_sync import sync_deliverable_dates_from_kickoff

    deliverable = Deliverable.objects.select_related('performance__contract').get(pk=deliverable_id)

    try:
        result = sync_deliverable_dates_from_kickoff(deliverable)
        print(
            f'[sync_deliverable_dates_from_kickoff_task] 완료 — '
            f'deliverable_id={deliverable_id}, {result}'
        )
        return {'status': 'ok', 'deliverable_id': deliverable_id, **result}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        print(f'[sync_deliverable_dates_from_kickoff_task] 실패 — deliverable_id={deliverable_id}\n{err}')
        raise self.retry(exc=exc)
