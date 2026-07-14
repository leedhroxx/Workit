# -*- coding: utf-8 -*-
"""
업로드 파일 확장자·MIME 타입·용량 검증.

계약서·산출물 등 사용자가 올리는 파일은 여기 정의한 화이트리스트를 통과해야
저장된다. 확장자와 브라우저가 보내는 Content-Type을 함께 확인해서, 확장자만
바꿔치기한 파일을 1차로 걸러낸다.

⚠ 한계: Content-Type은 클라이언트(브라우저)가 보내는 값이라 완전히 신뢰할 수는
없다. 더 엄격하게 하려면 파일 시그니처(매직 바이트) 검사가 필요하지만,
이번 적용 범위에서는 확장자+MIME 화이트리스트까지만 다룬다.
"""

from django.core.exceptions import ValidationError

# 확장자 → 허용되는 Content-Type 목록 (브라우저/OS별로 다르게 보낼 수 있어 여러 개 허용)
#
# pdf/hwp만 허용한다 — contracts/utils.py의 extract_text()가 이 두 형식만 정확하게
# 텍스트를 뽑아낸다. docx는 python-docx가 표(table) 안 내용을 못 읽어서 표 위주
# 섹션이 누락되고, doc/hwpx/xlsx/xls는 extract_text()에 처리 분기 자체가 없어
# 빈 문자열이 반환되어 "텍스트 추출 실패"로 곧바로 실패한다.
ALLOWED_UPLOAD_TYPES = {
    "pdf": {"application/pdf"},
    "hwp": {"application/x-hwp", "application/haansofthwp", "application/octet-stream"},
}

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50MB — 프런트 업로드 안내 문구와 동일하게 맞춤


def validate_uploaded_file(f):
    """
    허용되지 않는 확장자/형식/용량이면 django.core.exceptions.ValidationError를 던진다.
    통과하면 아무것도 반환하지 않는다 (호출부에서 f를 그대로 저장하면 됨).
    """
    if f is None:
        return

    name = (f.name or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext not in ALLOWED_UPLOAD_TYPES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_TYPES))
        raise ValidationError(f"허용되지 않는 파일 형식입니다 (.{ext or '확장자없음'}). 허용: {allowed}")

    if f.size and f.size > MAX_UPLOAD_SIZE_BYTES:
        raise ValidationError(f"파일 용량이 너무 큽니다 (최대 {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB).")

    content_type = (getattr(f, "content_type", "") or "").lower()
    allowed_mimes = ALLOWED_UPLOAD_TYPES[ext]
    # content_type이 브라우저/OS마다 들쭉날쭉해서(특히 hwp/hwpx), 화이트리스트에
    # 없어도 범용 fallback(application/octet-stream)이면 확장자 검증만으로 통과시킨다.
    if content_type and content_type not in allowed_mimes and content_type != "application/octet-stream":
        raise ValidationError(f"파일 형식이 일치하지 않습니다 (.{ext} 파일인데 감지된 형식: {content_type}).")
