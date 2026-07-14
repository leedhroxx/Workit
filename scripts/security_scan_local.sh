#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Workit 정적 보안 점검 스크립트
# 대응 체크리스트 항목: s7 (CSAP 3.3.2 / 13.2.1) — pip-audit · Bandit · Trivy
# 사용 환경: Windows + Git Bash(MINGW64) / Miniconda(web_service_env) 가정
#
# 사용법:
#   chmod +x security_scan_local.sh
#   ./security_scan_local.sh [Django 프로젝트 루트 경로]
#
# 기본값: 스크립트를 프로젝트 루트에 두고 인자 없이 실행하면 현재 디렉터리를 스캔합니다.
# ─────────────────────────────────────────────────────────────
set -uo pipefail

PROJECT_ROOT="${1:-.}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="./scripts/security_reports/${TS}"
mkdir -p "${REPORT_DIR}"

# Django 앱 디렉터리만 대상으로 (venv, node_modules, migrations 등 제외)
SCAN_TARGETS="contracts performance accounts config"

echo "================================================================"
echo " Workit 정적 보안 점검 시작  (${TS})"
echo " 리포트 저장 경로: ${REPORT_DIR}"
echo "================================================================"

# ── 1. Bandit: Python 코드 취약점 정적 분석 (CSAP 13.2.1) ──
echo ""
echo "[1/3] Bandit 실행 중 — Django 앱 소스코드 취약점 스캔..."
if ! command -v bandit &>/dev/null; then
  echo "  bandit 미설치 → pip install bandit --break-system-packages 로 설치해주세요 (건너뜀)"
else
  EXISTING_TARGETS=""
  for d in $SCAN_TARGETS; do
    [ -d "${PROJECT_ROOT}/${d}" ] && EXISTING_TARGETS="${EXISTING_TARGETS} ${PROJECT_ROOT}/${d}"
  done
  if [ -z "$EXISTING_TARGETS" ]; then
    EXISTING_TARGETS="${PROJECT_ROOT}"
  fi
  bandit -r ${EXISTING_TARGETS} \
    -f json -o "${REPORT_DIR}/bandit_report.json" \
    -x "*/migrations/*,*/tests/*,*/venv/*,*/.venv/*" \
    2>"${REPORT_DIR}/bandit_stderr.log"
  bandit -r ${EXISTING_TARGETS} \
    -f txt -o "${REPORT_DIR}/bandit_report.txt" \
    -x "*/migrations/*,*/tests/*,*/venv/*,*/.venv/*" \
    2>>"${REPORT_DIR}/bandit_stderr.log"
  HIGH=$(grep -o '"issue_severity": "HIGH"' "${REPORT_DIR}/bandit_report.json" 2>/dev/null | wc -l)
  MED=$(grep -o '"issue_severity": "MEDIUM"' "${REPORT_DIR}/bandit_report.json" 2>/dev/null | wc -l)
  echo "  완료 → HIGH: ${HIGH}건, MEDIUM: ${MED}건 (상세: bandit_report.txt)"
fi

# ── 2. pip-audit: 의존성 패키지 CVE 점검 (CSAP 3.3.2) ──
echo ""
echo "[2/3] pip-audit 실행 중 — 의존성 패키지 취약점(CVE) 스캔..."
if ! command -v pip-audit &>/dev/null; then
  echo "  pip-audit 미설치 → pip install pip-audit --break-system-packages 로 설치해주세요 (건너뜀)"
else
  REQ_FILE=""
  for f in "${PROJECT_ROOT}/requirements.txt" "${PROJECT_ROOT}/requirements/base.txt"; do
    [ -f "$f" ] && REQ_FILE="$f" && break
  done
  if [ -n "$REQ_FILE" ]; then
    pip-audit -r "$REQ_FILE" -f json -o "${REPORT_DIR}/pip_audit_report.json" 2>"${REPORT_DIR}/pip_audit_stderr.log"
    pip-audit -r "$REQ_FILE" 2>&1 | tee "${REPORT_DIR}/pip_audit_report.txt" >/dev/null
  else
    echo "  requirements.txt를 찾지 못해 현재 활성 환경 기준으로 스캔합니다."
    pip-audit -f json -o "${REPORT_DIR}/pip_audit_report.json" 2>"${REPORT_DIR}/pip_audit_stderr.log"
    pip-audit 2>&1 | tee "${REPORT_DIR}/pip_audit_report.txt" >/dev/null
  fi
  VULN_COUNT=$(grep -o '"id":' "${REPORT_DIR}/pip_audit_report.json" 2>/dev/null | wc -l)
  echo "  완료 → 발견된 취약 패키지 CVE: ${VULN_COUNT}건 (상세: pip_audit_report.txt)"
fi

# ── 3. Trivy: 파일시스템 / 컨테이너 이미지 취약점 스캔 (CSAP 3.3.2) ──
echo ""
echo "[3/3] Trivy 실행 중 — 파일시스템(의존성+설정) 취약점 스캔..."
if ! command -v trivy &>/dev/null; then
  echo "  trivy 미설치 → https://github.com/aquasecurity/trivy/releases 에서 설치해주세요 (건너뜀)"
else
  trivy fs "${PROJECT_ROOT}" \
    --severity HIGH,CRITICAL \
    --skip-dirs "node_modules,venv,.venv,migrations,security_reports" \
    --format json --output "${REPORT_DIR}/trivy_fs_report.json" \
    2>"${REPORT_DIR}/trivy_stderr.log"
  trivy fs "${PROJECT_ROOT}" \
    --severity HIGH,CRITICAL \
    --skip-dirs "node_modules,venv,.venv,migrations,security_reports" \
    2>>"${REPORT_DIR}/trivy_stderr.log" | tee "${REPORT_DIR}/trivy_fs_report.txt" >/dev/null

  # Dockerfile이 있으면 이미지 빌드 후 이미지 스캔도 옵션 제공
  if [ -f "${PROJECT_ROOT}/Dockerfile" ]; then
    echo "  Dockerfile 발견 — 이미지 스캔은 별도로 실행하세요:"
    echo "    docker build -t workit:scan-${TS} ${PROJECT_ROOT}"
    echo "    trivy image --severity HIGH,CRITICAL workit:scan-${TS}"
  fi
  CRIT=$(grep -o '"Severity":"CRITICAL"' "${REPORT_DIR}/trivy_fs_report.json" 2>/dev/null | wc -l)
  echo "  완료 → CRITICAL: ${CRIT}건 (상세: trivy_fs_report.txt)"
fi

echo ""
echo "================================================================"
echo " 정적 점검 완료. 리포트: ${REPORT_DIR}"
echo " 체크리스트 s7 항목(CSAP 3.3.2 / 13.2.1) 증적으로 활용 가능"
echo "================================================================"
