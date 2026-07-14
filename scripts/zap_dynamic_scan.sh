#!/usr/bin/env bash

# Workit OWASP ZAP 동적 보안 점검 스크립트
# 대응 체크리스트 항목: s7 (CSAP 3.3.2 / 13.2.1) — OWASP ZAP 동적 점검
#
# ⚠️ 반드시 본인이 소유/관리하는 서버(예: 스테이징 환경)에만 실행하세요.
#    타인이 운영하는 시스템에 대한 무단 스캔은 불법입니다.
#
# 사용법:
#   chmod +x zap_dynamic_scan.sh
#   ./zap_dynamic_scan.sh https://staging.workit.example.com [baseline|full]
#
#   baseline (기본값): 수동적 스캔 위주, 빠름 (수 분), 운영 서버에도 비교적 안전
#   full : 능동적 침투 시도 포함, 느림, 반드시 스테이징/테스트 서버에서만 실행
#
# 요구사항: Docker 설치 필요

set -euo pipefail

TARGET_URL="${1:-}"
SCAN_TYPE="${2:-baseline}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="./scripts/security_reports/zap_${TS}"

if [ -z "$TARGET_URL" ]; then
  echo "사용법: $0 <target_url> [baseline|full]"
  echo "예시:   $0 https://staging.workit.example.com baseline"
  exit 1
fi

if ! command -v docker &>/dev/null; then
  echo "Docker가 필요합니다. https://www.docker.com/ 에서 설치 후 다시 실행해주세요."
  exit 1
fi

mkdir -p "${REPORT_DIR}"
ABS_REPORT_DIR="$(cd "${REPORT_DIR}" && pwd)"

echo "================================================================"
echo " OWASP ZAP ${SCAN_TYPE} 스캔 시작"
echo " 대상: ${TARGET_URL}"
echo " 리포트 저장 경로: ${REPORT_DIR}"
echo "================================================================"
echo ""
echo "⚠️  대상이 본인 소유/관리 서버(스테이징 등)인지 다시 한 번 확인하세요."
read -r -p "계속하시겠습니까? (y/N) " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

if [ "$SCAN_TYPE" == "full" ]; then
  ZAP_SCRIPT="zap-full-scan.py"
  echo ""
  echo "※ full 스캔은 실제 침투 시도를 포함합니다. 운영 서버가 아닌지 다시 확인하세요."
else
  ZAP_SCRIPT="zap-baseline.py"
fi

MSYS_NO_PATHCONV=1 docker run --rm \
  --platform linux/amd64 \
  -v "${ABS_REPORT_DIR}:/zap/wrk/:rw" \
  -t ghcr.io/zaproxy/zaproxy:stable \
  ${ZAP_SCRIPT} \
  -t "${TARGET_URL}" \
  -r zap_report.html \
  -J zap_report.json \
  -w zap_report.md \
  -I

echo ""
echo "================================================================"
echo " ZAP 스캔 완료 (일부 경고가 있어도 정상 — HIGH 심각도 확인 필요)"
echo " HTML 리포트: ${REPORT_DIR}/zap_report.html"
echo " 체크리스트 s7 항목(CSAP 3.3.2 / 13.2.1) 증적으로 활용 가능"
echo "================================================================"
