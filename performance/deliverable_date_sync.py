# -*- coding: utf-8 -*-
"""
performance/deliverable_date_sync.py

사업수행계획서(kickoff Deliverable) PDF가 업로드되면, 그 안의
"산출물계획" 표를 파싱해 tech_apply(기술적용결과표)/final(사업추진결과보고서)
Deliverable의 due_date를 자동으로 채워준다.

전제:
- 기존 Deliverable 모델(kickoff/tech_apply/final, due_date 필드)을 그대로 사용
  → 새 모델 추가 없음, 마이그레이션 불필요
- 이미 사용자가 수동으로 입력한 due_date는 덮어쓰지 않음
  (deliverable_update_due_date 로 수정한 값을 자동 동기화가 되돌리지 않도록)
"""

import logging

logger = logging.getLogger(__name__)

from performance.deliverable_date_extractor import parse_output_plan

TARGET_TYPES = ("tech_apply", "final")


def sync_deliverable_dates_from_kickoff(kickoff_deliverable, overwrite: bool = False) -> dict:
    """
    kickoff(사업수행계획서) Deliverable 객체를 받아 파일을 파싱하고,
    같은 Performance 밑의 tech_apply/final Deliverable.due_date를 채운다.

    Args:
        kickoff_deliverable: Deliverable 인스턴스 (deliverable_type='kickoff', file 존재)
        overwrite: True면 이미 due_date가 있어도 PDF 값으로 덮어씀 (기본 False)

    Returns:
        {"updated": [...], "skipped": [...], "not_found_in_pdf": [...]}
    """
    from performance.models import Deliverable  # 순환 import 방지용 지연 import

    result = {"updated": [], "skipped": [], "not_found_in_pdf": []}

    if not kickoff_deliverable.file:
        logger.warning("kickoff Deliverable(id=%s)에 파일이 없어 파싱을 건너뜁니다.",
                        kickoff_deliverable.id)
        return result

    file_path = kickoff_deliverable.file.path
    if not file_path.lower().endswith('.pdf'):
        logger.warning(
            "kickoff Deliverable(id=%s)이 PDF가 아니라 자동 일정 반영을 건너뜁니다. (%s)",
            kickoff_deliverable.id, file_path,
        )
        return result

    try:
        parsed_items = parse_output_plan(file_path)
    except Exception:
        logger.exception("사업수행계획서 파싱 실패 (deliverable_id=%s)", kickoff_deliverable.id)
        raise

    # 파싱 결과를 deliverable_type 기준으로 정리 (같은 타입이 여러 번 매칭되면 첫 번째만 사용)
    matched_by_type = {}
    for item in parsed_items:
        if item.matched_type in TARGET_TYPES and item.matched_type not in matched_by_type:
            matched_by_type[item.matched_type] = item

    performance = kickoff_deliverable.performance

    for d_type in TARGET_TYPES:
        item = matched_by_type.get(d_type)
        if item is None:
            result["not_found_in_pdf"].append(d_type)
            continue

        target, _ = Deliverable.objects.get_or_create(
            performance=performance,
            deliverable_type=d_type,
        )

        if target.due_date and not overwrite:
            # 이미 값이 있고 덮어쓰기 옵션이 아니면 건너뜀 (수동 입력값 보호)
            result["skipped"].append({
                "type": d_type,
                "existing_due_date": target.due_date.isoformat(),
            })
            continue

        target.due_date = item.due_date
        target.save(update_fields=["due_date"])
        result["updated"].append({
            "type": d_type,
            "due_date": item.due_date,
            "raw_name": item.raw_name,
        })

    logger.info(
        "산출물 일정 자동 반영 완료 (performance_id=%s): %s",
        performance.id, result,
    )
    return result
