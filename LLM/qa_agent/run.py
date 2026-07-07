"""
로컬 테스트용 실행 스크립트.

백엔드/프론트 연결 전, 원본 텍스트 파일과 파싱된 JSON 파일을 직접 넣어서
QA 에이전트 결과를 확인할 때 사용한다.

사용법:
    python run.py --original original.txt --parsed parsed.json --doc-type rpt

--doc-type 은 rfp / pep / rpt 중 하나.

parsed.json 형식 예시 (소제목 텍스트를 key로, 본문을 value로):
    {
        "제1절 개요": "본 사업은 ...",
        "제2절 사업의 배경 및 목적": "본 사업의 배경은 ..."
    }
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from qa_agent.engine import review_section_mapping


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def display_label(item: dict) -> str:
    return (
        item.get("location")
        or item.get("title")
        or item.get("code")
        or "-"
    )


ISSUE_TYPE_LABELS = {
    "section_title_mismatch": "소제목 검토",
    "content_misplaced": "내용 위치 검토",
    "section_content_mismatch": "소제목 및 내용 검토",
    "section_title_content_mismatch": "소제목 및 내용 검토",
    "missing_section": "소제목 누락",
    "empty_section": "본문 누락",
    "unrecognized_section": "미등록 소제목",
    "section_order_mismatch": "소제목 순서 검토",
    "paragraph_missing": "문단 누락",
    "paragraph_misplaced": "문단 위치 검토",
}


def display_issue_type(issue_type: str) -> str:
    return ISSUE_TYPE_LABELS.get(issue_type, issue_type)


def clean_message_codes(message: str) -> str:
    return re.sub(r"\(([A-Z]{3}-\d+(?:-\d+)*)\)", "", message or "").strip()


def normalize_comment_focus(message: str) -> str:
    """
    평가셋 코멘트의 확인 대상을 부드럽게 통일한다.
    예: 원문의 제안요청 개요 소제목을 확인하세요.
        -> 원문의 제안요청 개요 부분을 확인하세요.
    """
    message = message or ""

    # 원문의 OO 소제목/내용/본문/섹션/문단 확인 -> 원문의 OO 부분 확인
    message = re.sub(
        r"원문의 ([^'\n]+?) (?:소제목과 내용을 함께|소제목|내용|본문|섹션|문단 위치|문단)을 확인하세요",
        r"원문의 \1 부분을 확인하세요",
        message,
    )

    # 위 문장 끝이 '확인하세요.'처럼 마침표를 포함하는 경우도 자연스럽게 유지
    return message.strip()


def print_report(report: dict) -> None:
    print()
    print("========== 소제목 매핑 QA 검수 ==========")
    print(f"[문서 유형]        {report['document_type']}")
    review_status = report.get("review_status") or "FAIL"
    print(f"[최종 판정]        {review_status}")
    print(f"[자동 진행 가능]   {format_auto_proceed(report)}")
    print(f"[전체 유사도]      {report['content_similarity']}")
    print(f"[기대 소제목 수]   {report['expected_section_count']}")
    print(f"[매칭된 소제목 수] {report['matched_section_count']}")

    issues = report["issues"]
    print()
    print(f"[발견된 이슈: {len(issues)}건]")
    if not issues:
        print("- 없음")
    for issue in issues:
        label = display_label(issue)
        message = clean_message_codes(issue.get("message") or "")
        print(f"- [{display_issue_type(issue['issue_type'])}] {label}: {message}")
        if issue.get("sample"):
            print(f"    예시: {issue['sample']}")

    comments = report.get("comments") or []
    if comments:
        print()
        print(f"[평가셋 코멘트: {len(comments)}건]")
        for comment in comments:
            label = display_label(comment)
            message = clean_message_codes(comment.get("message") or "")
            message = normalize_comment_focus(message)
            print(f"- {label}: {message}")


def format_auto_proceed(report: dict) -> str:
    if report.get("can_auto_proceed"):
        return "예"
    return "아니오 (반려 코멘트 확인 필요)"


def main() -> None:
    parser = argparse.ArgumentParser(description="소제목 매핑 QA 검수 에이전트 실행")
    parser.add_argument("--original", required=True, help="원본 텍스트 파일 경로 (.txt)")
    parser.add_argument("--parsed", required=True, help="파싱 결과 JSON 파일 경로 (.json)")
    parser.add_argument("--doc-type", required=True, choices=["rfp", "pep", "rpt"], help="문서 유형")
    args = parser.parse_args()

    original_text = load_text(args.original)
    parsed_sections = load_json(args.parsed)

    report = review_section_mapping(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=args.doc_type,
    )

    print_report(report)

    out_path = Path(args.parsed).with_suffix(".review.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[리포트 저장] {out_path}")


if __name__ == "__main__":
    main()
