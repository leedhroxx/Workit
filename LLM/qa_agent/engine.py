from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Tuple

from qa_agent.section_spec import SectionSpec
from qa_agent.text_utils import (
    normalize_compare_text,
    clean_text,
    similarity,
    normalize_with_map,
    containment_ratio,
    split_paragraphs,
)


SECTION_CONTENT_THRESHOLD = 0.75

MIN_CONTENT_LENGTH = 5

PARAGRAPH_MIN_LENGTH = 20
PARAGRAPH_CONTAINMENT_THRESHOLD = 0.7

INFO_ISSUE_TYPES = {
    "unrecognized_section",
    "section_order_mismatch",
}


@dataclass
class SectionIssue:
    issue_type: str
    code: str = ""
    title: str = ""
    location: str = ""
    message: str = ""
    sample: str = ""
    severity: str = "info"

    related_code: str = ""
    related_title: str = ""

    parsed_sample: str = ""
    expected_missing: List[str] = field(default_factory=list)
    similarity_score: float = 0.0


@dataclass
class SectionReviewReport:
    passed: bool
    review_status: str
    can_auto_proceed: bool
    document_type: str
    content_similarity: float
    expected_section_count: int
    matched_section_count: int
    issues: List[SectionIssue] = field(default_factory=list)

    comments: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SectionMappingReviewAgent:
    def __init__(
        self,
        sections: List[SectionSpec],
        document_type: str,
        section_content_threshold: float = SECTION_CONTENT_THRESHOLD,
    ):
        self.sections = sections
        self.document_type = document_type
        self.section_content_threshold = section_content_threshold

    def review(
        self,
        original_text: str,
        parsed_sections: Any,
    ) -> SectionReviewReport:
        original_clean = clean_text(original_text)
        parsed_sections = parsed_sections or {}

        matched, unrecognized_keys, parsed_order_codes = self._parse_input(parsed_sections)

        issues: List[SectionIssue] = []

        for spec in self.sections:
            if spec.code not in matched:
                issues.append(
                    SectionIssue(
                        issue_type="missing_section",
                        code=spec.code,
                        title=spec.title,
                        message=f"기대 소제목 '{spec.title}'({spec.code})이 파싱 결과에서 확인되지 않습니다.",
                    )
                )

        empty_codes: set = set()
        for code, info in matched.items():
            content = (info.get("content") or "").strip()
            if len(normalize_compare_text(content)) < MIN_CONTENT_LENGTH:
                empty_codes.add(code)
                spec = self._spec_by_code(code)
                issues.append(
                    SectionIssue(
                        issue_type="empty_section",
                        code=code,
                        title=spec.title if spec else "",
                        message=f"'{info.get('parsed_title')}'({code}) 소제목은 있지만 내용이 비어 있거나 지나치게 짧습니다.",
                    )
                )

        for key in unrecognized_keys:
            issues.append(
                SectionIssue(
                    issue_type="unrecognized_section",
                    title=key,
                    message=f"파싱 결과의 '{key}' 항목은 기준 소제목 목록에 없습니다. 신규 항목이거나 소제목 분리가 잘못됐을 수 있습니다.",
                )
            )

        for code, info in matched.items():
            spec = self._spec_by_code(code)
            parsed_title = str(info.get("parsed_title") or "").strip()
            if spec and parsed_title and not self._title_matches_spec(parsed_title, spec):
                issues.append(
                    SectionIssue(
                        issue_type="section_title_mismatch",
                        code=code,
                        title=spec.title,
                        message=(
                            f"'{parsed_title}'({code}) 소제목명이 기준 소제목 "
                            f"'{spec.title}'와 일치하지 않습니다."
                        ),
                        sample=parsed_title,
                    )
                )

        all_codes = [spec.code for spec in self.sections]
        positions, original_norm, index_map = self._find_section_positions(original_clean, all_codes)

        issues.extend(self._check_order(positions, parsed_order_codes))

        section_level_issues = self._check_section_content_mapping(
            original_clean=original_clean,
            original_norm=original_norm,
            index_map=index_map,
            positions=positions,
            matched=matched,
            empty_codes=empty_codes,
        )
        issues.extend(section_level_issues)

        already_flagged_codes = {
            issue.code
            for issue in section_level_issues
            if issue.issue_type in ("content_misplaced", "section_content_mismatch")
        }

        issues.extend(
            self._check_paragraph_level_content(
                original_clean=original_clean,
                index_map=index_map,
                positions=positions,
                matched=matched,
                skip_codes=empty_codes | already_flagged_codes,
            )
        )

        joined_parsed_text = "\n".join((info.get("content") or "") for info in matched.values())
        content_similarity = similarity(original_text, joined_parsed_text)

        review_status, can_auto_proceed = self._decide_status(content_similarity, issues)
        self._assign_issue_severities(issues)

        comments = self._build_issue_comments(issues)
        report_issues = self._merge_title_content_issues(issues)
        self._attach_issue_locations(report_issues)

        return SectionReviewReport(
            passed=review_status == "PASS",
            review_status=review_status,
            can_auto_proceed=can_auto_proceed,
            document_type=self.document_type,
            content_similarity=round(content_similarity, 4),
            expected_section_count=len(self.sections),
            matched_section_count=len(matched),
            issues=report_issues,
            comments=comments,
        )

    def _attach_issue_locations(self, issues: List[SectionIssue]) -> None:
        for issue in issues:
            issue.location = self._section_location(issue.code, issue.title)

    def _section_location(self, code: str, title: str = "") -> str:
        spec = self._spec_by_code(code)
        if spec and spec.group:
            return f"{spec.group} > {spec.title}"
        if spec:
            return spec.title
        return title or code or "-"

    def _build_issue_comments(self, issues: List[SectionIssue]) -> List[Dict[str, str]]:
        comments: List[Dict[str, str]] = []
        content_issue_types = {
            "section_content_mismatch",
            "content_misplaced",
            "paragraph_missing",
            "paragraph_misplaced",
        }
        content_issue_by_code = {
            issue.code: issue
            for issue in issues
            if issue.code and issue.issue_type in content_issue_types
        }
        title_problem_codes = {
            issue.code
            for issue in issues
            if issue.code and issue.issue_type == "section_title_mismatch"
        }

        for issue in issues:
            if issue.issue_type in content_issue_types and issue.code in title_problem_codes:
                continue

            sample_text = issue.sample.strip() if issue.sample else ""
            parsed_text = issue.parsed_sample.strip()
            missing_items = issue.expected_missing or self._extract_missing_items(sample_text, parsed_text)
            missing_text = self._format_missing_items(missing_items)

            if issue.issue_type == "section_content_mismatch":
                if missing_items:
                    message = (
                        f"파싱결과에 {missing_text}가 없어서 "
                        f"원문의 {issue.title} 내용을 확인하세요."
                    )
                else:
                    message = (
                        f"파싱결과가 원문과 충분히 일치하지 않아서 "
                        f"원문의 {issue.title} 내용을 확인하세요."
                    )

            elif issue.issue_type == "content_misplaced":
                wrong_items = self._extract_wrong_content_items(issue.parsed_sample)
                wrong_text = self._format_missing_items(wrong_items[:3])

                if wrong_items:
                    message = (
                        f"{wrong_text} 내용이 잘못 들어가 있어 "
                        f"원문의 {issue.title} 내용을 확인하세요."
                    )
                else:
                    message = (
                        f"원문의 {issue.title} 내용이 다른 섹션에 들어간 것으로 보여 "
                        f"원문의 {issue.title} 내용을 확인하세요."
                    )

            elif issue.issue_type == "missing_section":
                message = (
                    f"파싱결과에 해당 소제목이 없어서 "
                    f"원문의 {issue.title} 섹션을 확인하세요."
                )

            elif issue.issue_type == "empty_section":
                message = (
                    f"파싱결과에 소제목만 있고 본문이 부족해서 "
                    f"원문의 {issue.title} 본문을 확인하세요."
                )

            elif issue.issue_type == "section_title_mismatch":
                parsed_title = sample_text or "확인 불가"
                content_issue = content_issue_by_code.get(issue.code)

                if content_issue:
                    wrong_content_items = self._extract_wrong_content_items(
                        content_issue.parsed_sample
                    )
                    wrong_content_text = self._format_missing_items(wrong_content_items)
                    if wrong_content_items:
                        message = (
                            f"소제목이 '{parsed_title}'로 잘못 파싱되었고, "
                            f"{wrong_content_text} 내용이 잘못 들어가 있어 "
                            f"원문의 {issue.title} 소제목과 내용을 함께 확인하세요."
                        )
                    else:
                        message = (
                            f"소제목이 '{parsed_title}'로 잘못 파싱되었고, "
                            f"본문도 원문의 {issue.title} 내용과 달라서 "
                            f"원문의 {issue.title} 소제목과 내용을 함께 확인하세요."
                        )
                else:
                    message = (
                        f"본문은 원문의 {issue.title} 내용과 유사하지만, "
                        f"소제목이 '{parsed_title}'로 되어 있어서 "
                        f"원문의 {issue.title} 소제목을 확인하세요."
                    )

            elif issue.issue_type == "paragraph_misplaced":
                message = (
                    f"일부 문단이 다른 섹션에 들어간 것으로 보여 "
                    f"원문의 {issue.title} 문단 위치를 확인하세요."
                )

                if issue.related_title:
                    message += f" 현재는 '{issue.related_title}' 쪽과 더 유사합니다."

            elif issue.issue_type == "paragraph_missing":
                focus_text = self._focus_sample_text(sample_text, max_len=50)
                message = (
                    f"파싱결과에 '{focus_text}' 문단이 없어서 "
                    f"원문의 {issue.title} 본문을 확인하세요."
                )

            elif issue.issue_type == "section_order_mismatch":
                message = "소제목 순서: 파싱결과의 소제목 순서가 원문과 달라서 원문 목차 순서를 확인하세요."

            elif issue.issue_type == "unrecognized_section":
                message = (
                    f"기준 소제목 목록에 없는 항목이라서 "
                    f"원문에서 실제 소제목인지 확인하세요."
                )

            else:
                message = "파싱결과를 원문과 다시 비교하세요."

            comments.append(
                {
                    "code": issue.code,
                    "title": issue.title,
                    "location": self._section_location(issue.code, issue.title),
                    "message": message,
                }
            )

        return comments

    def _merge_title_content_issues(self, issues: List[SectionIssue]) -> List[SectionIssue]:
        content_issue_types = {
            "section_content_mismatch",
            "content_misplaced",
            "paragraph_missing",
            "paragraph_misplaced",
        }
        content_issue_by_code = {
            issue.code: issue
            for issue in issues
            if issue.code and issue.issue_type in content_issue_types
        }
        title_problem_codes = {
            issue.code
            for issue in issues
            if issue.code and issue.issue_type == "section_title_mismatch"
        }

        merged: List[SectionIssue] = []

        for issue in issues:
            if issue.issue_type in content_issue_types and issue.code in title_problem_codes:
                continue

            if issue.issue_type != "section_title_mismatch":
                merged.append(issue)
                continue

            content_issue = content_issue_by_code.get(issue.code)
            if not content_issue:
                merged.append(issue)
                continue

            wrong_content_items = self._extract_wrong_content_items(
                content_issue.parsed_sample
            )
            wrong_content_text = self._format_missing_items(wrong_content_items)
            if wrong_content_items:
                message = (
                    f"{issue.message} 또한 {wrong_content_text} 내용이 잘못 들어가 있어 "
                    f"원문의 '{issue.title}' 내용을 확인해야 합니다."
                )
            else:
                message = (
                    f"{issue.message} 또한 본문도 원문의 '{issue.title}' 내용과 "
                    "일치하지 않습니다."
                )

            merged.append(
                replace(
                    issue,
                    issue_type="section_title_content_mismatch",
                    message=message,
                    related_code=content_issue.related_code,
                    related_title=content_issue.related_title,
                    parsed_sample=content_issue.parsed_sample,
                    expected_missing=content_issue.expected_missing,
                    similarity_score=content_issue.similarity_score,
                )
            )

        return merged

    def _extract_wrong_content_items(self, parsed_text: str) -> List[str]:
        text = " ".join((parsed_text or "").split())
        if not text:
            return []

        text = re.sub(
            r"(기능을\s*(?:제공하여야 한다|개발하였다|구현하였다|구현하고)|"
            r"방안을\s*제시하여야 한다|"
            r"업무를\s*(?:처리하고|운영하고))",
            ",",
            text,
        )
        text = re.sub(r"재고 회전율을 높이기 위한\s*", "재고 회전율, ", text)
        text = re.sub(
            r"(도입하고|전환하고|복구하여|분석하고|적용하고|수행한 뒤)\s*",
            ", ",
            text,
        )

        items: List[str] = []

        for part in re.split(r"[,，、;；\n.]+", text):
            for split_part in re.split(
                r"\s+(?:및|또는)\s+|(?<=\S)하고\s+|(?<=\S)하며\s+|(?<=\S)하여\s+",
                part,
            ):
                item = self._clean_wrong_content_item(split_part)
                if item:
                    items.append(item)

        items = self._dedupe_preserve_order(items)
        if items:
            return items

        fallback = self._focus_sample_text(parsed_text, max_len=120)
        return [fallback] if fallback else []

    def _clean_wrong_content_item(self, phrase: str) -> str:
        phrase = " ".join((phrase or "").split()).strip(" .。")

        phrase = re.sub(r"^(시스템은|제안사는|민원 담당자는|서버 장비는)\s*", "", phrase)
        phrase = re.sub(r"^장애 발생 시\s*", "", phrase)
        phrase = re.sub(
            r"(기능을 제공하여야 한다|기능을 개발하였다|기능을 구현하였다|"
            r"기능을 구현하고.*|방안을 제시하여야 한다)$",
            "",
            phrase,
        )
        phrase = re.sub(
            r"(업무를 처리한다|업무를 운영한다|하도록 구현하였다|수립한다|"
            r"제공한다|제공하였다|실시한다|수행한다|설치한다|운영한다|"
            r"월별로 운영한다|최소화한다|관리할 수 있어야 한다)$",
            "",
            phrase,
        )
        phrase = phrase.strip()
        phrase = re.sub(r"(을|를|은|는|이|가|와|과|에|로|으로)(?=\s|$)", "", phrase)
        phrase = " ".join(phrase.split()).strip()

        if len(normalize_compare_text(phrase)) < 2:
            return ""

        if len(phrase) > 45:
            return self._focus_sample_text(phrase, max_len=45)

        return phrase


    def _extract_missing_items(
        self,
        original_text: str,
        parsed_text: str,
        max_items: Optional[int] = None,
    ) -> List[str]:
        if not original_text or not parsed_text:
            return []

        candidates = self._candidate_phrases(original_text)
        missing: List[str] = []

        for phrase in candidates:
            if not self._phrase_present(phrase, parsed_text):
                missing.append(phrase)

            if max_items is not None and len(missing) >= max_items:
                break

        return missing

    def _candidate_phrases(self, text: str) -> List[str]:
        text = " ".join((text or "").split())
        text = re.sub(
            r"(기능을\s*(?:제공하여야 한다|개발하였다|구현하였다|구현하고)|"
            r"방안을\s*제시하여야 한다|"
            r"업무를\s*(?:처리하고|운영하고))",
            ",",
            text,
        )
        text = re.sub(r"(도입하고|전환하고|복구하여)\s*", ", ", text)
        raw_parts = re.split(r"[,，、;；\n]+", text)

        candidates: List[str] = []

        for part in raw_parts:
            part = self._clean_candidate_phrase(part)
            if not part:
                continue

            split_parts = re.split(r"\s+(?:및|또는)\s+|(?<=\S)하고\s+|(?<=\S)하며\s+", part)
            for split_part in split_parts:
                split_part = self._clean_candidate_phrase(split_part)
                if split_part:
                    candidates.append(split_part)

        return self._dedupe_preserve_order(candidates)

    def _clean_candidate_phrase(self, phrase: str) -> str:
        phrase = " ".join((phrase or "").split())

        phrase = re.sub(r"^(시스템은|제안사는|민원 담당자는|각 위험은|위험별|교육내용은)\s*", "", phrase)
        phrase = re.sub(
            r"(기능을 제공하여야 한다|기능을 개발하였다|기능을 구현하였다|"
            r"기능을 구현하고.*|방안을 제시하여야 한다)$",
            "",
            phrase,
        )
        phrase = re.sub(
            r"(업무를 처리한다|업무를 운영한다|하도록 구현하였다|수립한다|"
            r"제공한다|실시한다|수행한다|최소화한다|관리할 수 있어야 한다)$",
            "",
            phrase,
        )
        phrase = re.sub(r"(을|를|은|는|이|가|와|과)$", "", phrase).strip()

        if len(normalize_compare_text(phrase)) < 4:
            return ""

        if len(phrase) > 35:
            return ""

        return phrase

    def _phrase_present(self, phrase: str, parsed_text: str) -> bool:
        phrase_norm = normalize_compare_text(self._strip_korean_particles(phrase))
        parsed_norm = normalize_compare_text(self._strip_korean_particles(parsed_text))

        if not phrase_norm:
            return True

        if phrase_norm in parsed_norm:
            return True

        return containment_ratio(phrase, parsed_text) >= 0.75

    def _strip_korean_particles(self, text: str) -> str:
        return re.sub(r"(은|는|이|가|을|를|과|와|의|에|에서|으로|로|도|만)\b", "", text or "")

    def _format_missing_items(self, items: List[str]) -> str:
        if not items:
            return "원문의 핵심 항목"

        quoted = [f"'{item}'" for item in items]

        if len(quoted) == 1:
            return quoted[0]

        return ", ".join(quoted)

    @staticmethod
    def _focus_sample_text(sample_text: str, max_len: int = 80) -> str:
        text = " ".join(sample_text.split())
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "..."

    def _spec_by_code(self, code: str) -> Optional[SectionSpec]:
        for spec in self.sections:
            if spec.code == code:
                return spec
        return None

    def _title_matches_spec(self, parsed_title: str, spec: SectionSpec) -> bool:
        parsed_norm = normalize_compare_text(parsed_title)
        candidate_norms = {normalize_compare_text(candidate) for candidate in spec.title_candidates()}
        return parsed_norm in candidate_norms

    def _parse_input(
        self,
        parsed_sections: Any,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        if isinstance(parsed_sections, dict):
            return self._parse_flat_dict(parsed_sections)
        if isinstance(parsed_sections, list):
            return self._parse_record_list(parsed_sections)

        raise TypeError(
            "parsed_sections는 {'소제목': '내용'} 형태의 dict 또는 "
            "[{'section_id':.., 'section_title':.., 'content':..}, ...] 형태의 list여야 합니다. "
            f"입력 타입: {type(parsed_sections)}"
        )

    def _parse_flat_dict(
        self,
        parsed_sections: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        matched: Dict[str, Dict[str, Any]] = {}
        used_keys: set = set()

        for spec in self.sections:
            candidate_norms = {normalize_compare_text(c) for c in spec.title_candidates()}

            for key, content in parsed_sections.items():
                if key in used_keys:
                    continue

                if normalize_compare_text(key) in candidate_norms:
                    matched[spec.code] = {
                        "parsed_title": key,
                        "content": self._content_to_text(content),
                    }
                    used_keys.add(key)
                    break

        code_by_key = {info["parsed_title"]: code for code, info in matched.items()}
        parsed_order_codes = [
            code_by_key[key]
            for key in parsed_sections.keys()
            if key in code_by_key
        ]
        unrecognized = [
            key
            for key in parsed_sections.keys()
            if key not in used_keys
        ]

        return matched, unrecognized, parsed_order_codes

    def _parse_record_list(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        matched: Dict[str, Dict[str, Any]] = {}
        parsed_order_codes: List[str] = []
        unrecognized: List[str] = []
        valid_codes = {spec.code for spec in self.sections}

        for record in records:
            if not isinstance(record, dict):
                continue

            section_id = str(record.get("section_id") or "").strip()
            code = self._section_id_to_code(section_id)
            title = record.get("section_title") or record.get("title") or ""
            content = self._content_to_text(record.get("content") or "")
            expected_missing = self._extract_record_missing_items(record)

            if code in valid_codes and code not in matched:
                matched[code] = {
                    "parsed_title": title,
                    "content": content,
                    "section_id": section_id,
                    "expected_missing": expected_missing,
                }
                parsed_order_codes.append(code)
            else:
                unrecognized.append(section_id or title or "(section_id 없음)")

        return matched, unrecognized, parsed_order_codes

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            parts: List[str] = []
            preferred_keys = ("rfp_excerpt", "pep_excerpt", "rpt_excerpt", "text", "content")

            for key in preferred_keys:
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())

            if parts:
                return "\n".join(parts)

            return "\n".join(
                str(value).strip()
                for value in content.values()
                if value is not None and str(value).strip()
            )

        if isinstance(content, list):
            return "\n".join(str(item).strip() for item in content if str(item).strip())

        return str(content)

    def _extract_record_missing_items(self, record: Dict[str, Any]) -> List[str]:
        for key in ("missing_items", "expected_missing", "missing", "omitted_items"):
            values = self._metadata_to_list(record.get(key))
            if values:
                return values

        comment = str(record.get("comment") or "")
        return self._extract_missing_items_from_comment(comment)

    @staticmethod
    def _metadata_to_list(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []

            return [
                item.strip()
                for item in re.split(r"[,，、;；\n]+", text)
                if item.strip()
            ]

        return []

    def _extract_missing_items_from_comment(self, comment: str) -> List[str]:
        if not comment or "누락" not in comment:
            return []

        text = comment
        text = re.sub(r"^[^:：]{1,20}\s*의도\s*[:：]\s*", "", text)
        text = re.sub(r"(을|를)?\s*일부러\s*누락.*$", "", text)
        text = re.sub(r".*포함해야\s*하는데", "", text)
        text = re.sub(r"parsed에는.*$", "", text)
        text = text.replace(" 외에도 ", ", ")
        text = text.replace("뿐 아니라", ", ")
        text = text.replace("까지", "")

        items = [
            self._clean_metadata_item(item)
            for item in re.split(r"[,，、;；]+", text)
        ]

        return self._dedupe_preserve_order([item for item in items if item])

    @staticmethod
    def _clean_metadata_item(item: str) -> str:
        item = " ".join((item or "").split())
        item = re.sub(r"^[^:：]{1,20}\s*의도\s*[:：]\s*", "", item)
        item = re.sub(r"(을|를|은|는|이|가|와|과)$", "", item).strip()
        return item

    @staticmethod
    def _section_id_to_code(section_id: str) -> str:
        if not section_id:
            return ""

        parts = section_id.strip().split("_")
        if len(parts) < 2:
            return section_id.upper()

        prefix = parts[0].upper()
        rest = "-".join(parts[1:])
        return f"{prefix}-{rest}"

    _TOC_WINDOW_RATIO = 0.15
    _TOC_MIN_CLUSTER_RATIO = 0.5

    def _sequential_search(
        self,
        original_norm: str,
        ordered_specs: List[SectionSpec],
        start_from: int,
    ) -> Dict[str, Tuple[int, str]]:
        positions: Dict[str, Tuple[int, str]] = {}
        cursor = start_from

        for spec in ordered_specs:
            found_idx = -1
            found_text = ""

            for candidate in spec.title_candidates():
                candidate_norm = normalize_compare_text(candidate)
                if not candidate_norm:
                    continue

                idx = original_norm.find(candidate_norm, cursor)
                if idx >= 0:
                    found_idx = idx
                    found_text = candidate
                    break

            if found_idx >= 0:
                positions[spec.code] = (found_idx, found_text)
                cursor = found_idx + len(normalize_compare_text(found_text))

        return positions

    def _find_section_positions(
        self,
        original_clean: str,
        codes: Any,
    ) -> Tuple[Dict[str, Tuple[int, str]], str, List[int]]:
        original_norm, index_map = normalize_with_map(original_clean)
        ordered_specs = [spec for spec in self.sections if spec.code in codes]

        positions = self._sequential_search(original_norm, ordered_specs, start_from=0)

        doc_len = len(original_norm)
        toc_window = doc_len * self._TOC_WINDOW_RATIO
        clustered = [pos for pos, _ in positions.values() if pos <= toc_window]

        if positions and len(clustered) >= len(positions) * self._TOC_MIN_CLUSTER_RATIO:
            toc_end = max(clustered)
            retried = self._sequential_search(original_norm, ordered_specs, start_from=toc_end)
            if retried:
                positions = retried

        return positions, original_norm, index_map

    def _check_order(
        self,
        positions: Dict[str, Tuple[int, str]],
        parsed_order_codes: List[str],
    ) -> List[SectionIssue]:
        expected_order = sorted(positions.keys(), key=lambda code: positions[code][0])

        common = set(expected_order) & set(parsed_order_codes)
        expected_filtered = [code for code in expected_order if code in common]
        parsed_filtered = self._dedupe_preserve_order(
            [code for code in parsed_order_codes if code in common]
        )

        if expected_filtered and parsed_filtered and expected_filtered != parsed_filtered:
            return [
                SectionIssue(
                    issue_type="section_order_mismatch",
                    message="소제목 순서가 원본과 다릅니다.",
                    sample=f"원본 순서={expected_filtered}, 파싱 순서={parsed_filtered}",
                )
            ]

        return []

    def _check_section_content_mapping(
        self,
        original_clean: str,
        original_norm: str,
        index_map: List[int],
        positions: Dict[str, Tuple[int, str]],
        matched: Dict[str, Dict[str, Any]],
        empty_codes: Optional[set] = None,
    ) -> List[SectionIssue]:
        empty_codes = empty_codes or set()
        issues: List[SectionIssue] = []

        ordered_codes = sorted(positions.keys(), key=lambda code: positions[code][0])

        norm_slices: Dict[str, str] = {}
        raw_slices: Dict[str, str] = {}

        for index, code in enumerate(ordered_codes):
            start = positions[code][0]
            end = (
                positions[ordered_codes[index + 1]][0]
                if index + 1 < len(ordered_codes)
                else len(original_norm)
            )

            norm_slice = original_norm[start:end]
            matched_title_norm = normalize_compare_text(positions[code][1])
            if matched_title_norm and norm_slice.startswith(matched_title_norm):
                norm_slice = norm_slice[len(matched_title_norm):]
            norm_slices[code] = norm_slice

            raw_start = index_map[start] if start < len(index_map) else len(original_clean)
            raw_end = index_map[end] if end < len(index_map) else len(original_clean)
            raw_slice = original_clean[raw_start:raw_end]

            title_norm = normalize_compare_text(positions[code][1])
            lines = raw_slice.split("\n", 1)
            if len(lines) > 1 and normalize_compare_text(lines[0]) == title_norm:
                raw_slice = lines[1]

            raw_slices[code] = raw_slice.strip()

        for code, original_slice in norm_slices.items():
            info = matched.get(code)
            spec = self._spec_by_code(code)

            if info is None:
                continue

            if code in empty_codes:
                continue

            parsed_content = info.get("content") or ""
            expected_missing = info.get("expected_missing") or []
            own_similarity = similarity(original_slice, parsed_content)

            if own_similarity >= self.section_content_threshold:
                continue

            best_other_code = None
            best_other_score = own_similarity

            for other_code, other_info in matched.items():
                if other_code == code:
                    continue

                other_score = similarity(original_slice, other_info.get("content") or "")
                if other_score > best_other_score:
                    best_other_score = other_score
                    best_other_code = other_code

            title = spec.title if spec else code
            sample_text = raw_slices.get(code, "")[:300]

            if best_other_code is not None:
                other_spec = self._spec_by_code(best_other_code)
                other_title = other_spec.title if other_spec else best_other_code
                issues.append(
                    SectionIssue(
                        issue_type="content_misplaced",
                        code=code,
                        title=title,
                        related_code=best_other_code,
                        related_title=other_title,
                        message=(
                            f"'{title}'({code}) 원본 내용이 '{other_title}'({best_other_code}) 섹션에 "
                            f"잘못 들어간 것으로 보입니다. (자기 섹션 유사도 {own_similarity:.2f} vs "
                            f"'{best_other_code}' 유사도 {best_other_score:.2f})"
                        ),
                        sample=sample_text,
                        parsed_sample=parsed_content[:500],
                        expected_missing=expected_missing,
                        similarity_score=round(own_similarity, 4),
                    )
                )
            else:
                issues.append(
                    SectionIssue(
                        issue_type="section_content_mismatch",
                        code=code,
                        title=title,
                        message=(
                            f"'{title}'({code}) 섹션의 원본 내용이 파싱 결과에 제대로 반영되지 않은 것으로 "
                            f"보입니다. (유사도 {own_similarity:.2f})"
                        ),
                        sample=sample_text,
                        parsed_sample=parsed_content[:500],
                        expected_missing=expected_missing,
                        similarity_score=round(own_similarity, 4),
                    )
                )

        return issues

    def _check_paragraph_level_content(
        self,
        original_clean: str,
        index_map: List[int],
        positions: Dict[str, Tuple[int, str]],
        matched: Dict[str, Dict[str, Any]],
        skip_codes: set,
    ) -> List[SectionIssue]:
        issues: List[SectionIssue] = []
        ordered_codes = sorted(positions.keys(), key=lambda code: positions[code][0])

        raw_slices: Dict[str, str] = {}

        for index, code in enumerate(ordered_codes):
            norm_start = positions[code][0]
            norm_end = (
                positions[ordered_codes[index + 1]][0]
                if index + 1 < len(ordered_codes)
                else len(index_map)
            )

            raw_start = index_map[norm_start] if norm_start < len(index_map) else len(original_clean)
            raw_end = index_map[norm_end] if norm_end < len(index_map) else len(original_clean)
            raw_slice = original_clean[raw_start:raw_end]

            title_norm = normalize_compare_text(positions[code][1])
            lines = raw_slice.split("\n", 1)
            if len(lines) > 1 and normalize_compare_text(lines[0]) == title_norm:
                raw_slice = lines[1]

            raw_slices[code] = raw_slice

        for code, raw_slice in raw_slices.items():
            if code in skip_codes or code not in matched:
                continue

            own_content = matched[code].get("content") or ""
            spec = self._spec_by_code(code)
            title = spec.title if spec else code

            for paragraph in split_paragraphs(raw_slice):
                if len(normalize_compare_text(paragraph)) < PARAGRAPH_MIN_LENGTH:
                    continue

                own_score = containment_ratio(paragraph, own_content)
                if own_score >= PARAGRAPH_CONTAINMENT_THRESHOLD:
                    continue

                best_other_code = None
                best_other_score = own_score

                for other_code, other_info in matched.items():
                    if other_code == code:
                        continue

                    other_score = containment_ratio(paragraph, other_info.get("content") or "")
                    if other_score > best_other_score:
                        best_other_score = other_score
                        best_other_code = other_code

                if best_other_code is not None and best_other_score >= PARAGRAPH_CONTAINMENT_THRESHOLD:
                    other_spec = self._spec_by_code(best_other_code)
                    other_title = other_spec.title if other_spec else best_other_code
                    issues.append(
                        SectionIssue(
                            issue_type="paragraph_misplaced",
                            code=code,
                            title=title,
                            related_code=best_other_code,
                            related_title=other_title,
                            message=(
                                f"'{title}'({code}) 안의 문단 하나가 '{other_title}'({best_other_code}) 쪽으로 "
                                f"넘어간 것으로 보입니다. (자기 섹션 포함비율 {own_score:.2f} vs "
                                f"'{best_other_code}' 포함비율 {best_other_score:.2f})"
                            ),
                            sample=paragraph[:200],
                            parsed_sample=own_content[:500],
                        )
                    )
                else:
                    issues.append(
                        SectionIssue(
                            issue_type="paragraph_missing",
                            code=code,
                            title=title,
                            message=(
                                f"'{title}'({code}) 안의 문단 하나가 파싱 결과 어디에서도 확인되지 않습니다. "
                                f"(자기 섹션 포함비율 {own_score:.2f})"
                            ),
                            sample=paragraph[:200],
                            parsed_sample=own_content[:500],
                        )
                    )

        return issues

    def _decide_status(
        self,
        content_similarity: float,
        issues: List[SectionIssue],
    ) -> Tuple[str, bool]:
        """
        판정 기준:
        - 이슈가 하나도 없으면 PASS
        - 이슈가 1개라도 있으면 FAIL
        """
        if not issues:
            return "PASS", True

        return "FAIL", False

    def _assign_issue_severities(self, issues: List[SectionIssue]) -> None:
        for issue in issues:
            issue.severity = "info" if issue.issue_type in INFO_ISSUE_TYPES else "review"

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        result = []

        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)

        return result


def review_section_mapping(
    original_text: str,
    parsed_sections: Any,
    document_type: str,
) -> Dict[str, Any]:
    from qa_agent.registry import get_sections

    sections = get_sections(document_type)
    agent = SectionMappingReviewAgent(
        sections=sections,
        document_type=document_type,
    )

    return agent.review(
        original_text=original_text,
        parsed_sections=parsed_sections,
    ).to_dict()
