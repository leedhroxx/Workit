<div align="center">

# 📑 Workit

**교육청 SI 계약·산출물 검토 보조 플랫폼**

담당자의 계약서 검토 부담을 줄이고, 검토 기준을 정형화하고, 검토 이력을 축적합니다.

[![Django](https://img.shields.io/badge/Django-6.0.5-092E20?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-DC244C)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Dev Period](https://img.shields.io/badge/개발기간-2026.05.22%20~%202026.07.16-lightgrey)](#)

**🔗 배포/데모: _(링크 추가 예정)_**

</div>

---

## 목차

- [1. 프로젝트 소개](#1-프로젝트-소개)
- [2. 핵심 기능](#2-핵심-기능)
- [3. 사용자 시나리오](#3-사용자-시나리오)
- [4. 시스템 구조](#4-시스템-구조)
- [5. 데이터 처리 과정](#5-데이터-처리-과정)
- [6. 기술 스택](#6-기술-스택)
- [7. Quick Start](#7-quick-start)
- [8. 디렉터리 구조](#8-디렉터리-구조)
- [9. 트러블슈팅](#9-트러블슈팅)
- [10. 향후 개선 방향](#10-향후-개선-방향)
- [11. 팀 소개](#11-팀-소개)

---

## 1. 프로젝트 소개

교육청 담당자는 IT·계약 전문가가 아님에도 SI 계약서와 산출물을 직접 검토해야 합니다. 순환보직 때문에 검토 노하우가 담당자가 바뀔 때마다 사라지고, 산출물 형식·내용도 사업마다 제각각이라 검토 기준 자체가 명확하지 않습니다.

Workit은 담당자를 완전히 대체하는 게 아니라, **검토 부담 완화 + 기준 정형화 + 이력 축적**에 집중하는 검토 보조 플랫폼입니다. 핵심 기능은 다음 다섯 가지입니다.

| # | 기능 |
|---|---|
| 1 | 계약서 검토 |
| 2 | 산출물 관리 |
| 3 | 정형화된 평가 기준 적용 |
| 4 | sLLM 기반 검토 코멘트 제공 |
| 5 | 검토 결과 보고서화 |

## 2. 핵심 기능

### ① 계약서 검토
조항 파싱 → 법령 지식베이스(KB) 검색 → 위험/누락/위반 판정 → 근거 조항 제공

### ② 산출물 관리
착수계획서(PEP)·제안요청서(RFP)·결과보고서(RPT) 등 등록, 납기/상태 관리, 프로젝트별 이력 관리

### ③ 산출물 정형화 평가
산출물별 평가 항목 정의, 완전성·정확성·검증가능성 등 기준 적용, 프롬프트 에이전트로 평가 기준 일관화

### ④ sLLM 검토
Kanana 1.5 8B 기반 자체 모델(QLoRA 파인튜닝)로 계약/산출물 검토 태스크를 분리하여 판정 + 근거 + 코멘트 생성

## 3. 사용자 시나리오

```
계약서·RFP·요구사항 업로드
        ↓
   AI 계약서 분석
        ↓
    산출물 등록
        ↓
정형화된 평가 프롬프트/에이전트 실행
        ↓
   sLLM 검토 결과 확인
        ↓
     보고서 다운로드
```

## 4. 시스템 구조

```
Frontend
   ↓
Django Backend (accounts / contracts / performance)
   ↓
문서 파싱 (PDF·DOCX·HWP)
   ↓
Qdrant RAG (법령 조문 검색)
   ↓
FastAPI sLLM 서버 (Kanana 1.5 8B + QLoRA)
   ↓
결과 저장 / 보고서 생성
```

<!-- 📸 스크린샷/데모 이미지 자리 (추후 채우기) -->
<!-- ![architecture](./static/images/architecture.png) -->

## 5. 데이터 처리 과정

**법령 RAG 파이프라인**

```
법령 수집 → 조문(조/호) 단위 청킹 → BGE-M3 임베딩 → Qdrant 저장
```

- 조 단위(JoRAG) 청킹을 기준 아키텍처로 확정
- BGE-M3 임베딩 + bge-reranker-v2-m3 리랭커, 하이브리드 검색(RRF) 적용

**sLLM 학습 파이프라인**

```
산출물 평가 기준 데이터 구성 → SFT 데이터 제작 → 모델 학습·검증
```

- Base 모델: Kanana 1.5 8B
- 파인튜닝: QLoRA

## 6. 기술 스택

| 영역 | 기술 |
|---|---|
| 백엔드 | Django, Celery, Redis |
| 프론트엔드 | HTML/CSS/JavaScript, Django Template |
| 데이터베이스 | PostgreSQL (AWS RDS) |
| 벡터DB | Qdrant |
| AI/RAG | Python, OpenAI API, Kanana 1.5 8B(sLLM, QLoRA), BGE-M3(FlagEmbedding, Hugging Face), sentence-transformers, RRF, runpod |
| 문서 처리 | pdfplumber, PyMuPDF, python-docx, pdf2image, poppler, LibreOffice, ReportLab |
| 클라우드/배포 | AWS(EC2, S3, VPC), GitHub Actions, Docker |
| 협업/관리 | GitHub, Jira |

<!-- 📸 스크린샷/데모 이미지 자리 (추후 채우기) -->
<!-- ![demo](./static/images/demo.gif) -->

## 7. Quick Start

### 사전 요구사항
- Python 3.11+
- Docker / Docker Compose
- (sLLM 학습·추론 시) CUDA GPU 환경

### 1) 저장소 클론 및 가상환경

```bash
git clone https://github.com/SKN26-FLOW/Workit.git
cd Workit

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2) 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 SECRET_KEY 등 값을 채워주세요.
```

### 3) 인프라 실행 (PostgreSQL, Qdrant)

```bash
docker compose up -d
```

### 4) 모델 호환성 패치 (최초 1회)

```bash
python setup_env.py
```

> `transformers==4.49.0` 환경에서 로컬 sLLM·FlagEmbedding 로딩 시 필요한 호환성 패치를 적용합니다.

### 5) DB 마이그레이션 및 서버 실행

```bash
python manage.py migrate
python manage.py runserver
```

### 6) Celery 워커 실행 (비동기 문서 분석용, 별도 터미널)

```bash
celery -A config worker -l info
```

## 8. 디렉터리 구조

```
Workit/
├── config/          # Django 프로젝트 설정 (settings, celery, urls)
├── accounts/        # 계정/조직/권한 관리
├── contracts/       # 계약서·RFP·요구사항 등록 및 AI 분석
├── performance/      # 산출물 등록·평가·알림
├── rag/              # 법령 RAG 파이프라인 (청킹, 임베딩, 검색, 평가)
├── LLM/              # sLLM 학습·평가·검토 에이전트 (Kanana 1.5 8B + QLoRA)
├── templates/         # Django 템플릿
├── static/            # CSS/JS/이미지
├── docker/            # DB 초기화 스크립트, Qdrant 볼륨
├── docker-compose.yml
├── requirements.txt
└── setup_env.py       # 팀원 환경 세팅 스크립트
```

## 9. 트러블슈팅

> 개발 중 겪었던 주요 이슈와 해결 과정을 정리합니다. _(추후 채우기)_

| 이슈 | 원인 | 해결 |
|---|---|---|
| *(예: FlagEmbedding·transformers 버전 충돌)* | | |
| | | |
| | | |

## 10. 향후 개선 방향

- 실제 교육청 문서 데이터 확보
- 대용량 문서 처리 테스트
- 전문가 검수 기반 평가셋 확대
- 산출물 유형 확장
- 사용자 커스터마이징 기능 추가

## 11. 팀 소개

| 이름 | GitHub | 담당 역할 |
|---|---|---|
| 김민하 | [@leedhroxx](https://github.com/leedhroxx) | 데이터 수집 및 전처리, 백엔드 개발, 프론트엔드 개발, 인프라 배포 및 운영, 테스트 및 품질 개선 |
| 김용욱 | [@yonguk12077-beep](https://github.com/yonguk12077-beep) | 데이터 수집 및 전처리, AI 모델 개발, 테스트 및 품질 개선, 문서화 및 산출물 관리 |
| 배재현 | [@rshyun24](https://github.com/rshyun24) | 프로젝트 기획 및 관리, 데이터 수집 및 전처리, 데이터베이스 및 시스템 설계, AI 모델 개발, 문서화 및 산출물 관리 |
| 윤지혜 | [@jjhhyy0926](https://github.com/jjhhyy0926) | 데이터 수집 및 전처리, 데이터베이스 및 시스템 설계, AI 모델 개발, 테스트 및 품질 개선, 문서화 및 산출물 관리 |
| 전윤하 | [@yoonha315](https://github.com/yoonha315) | 데이터 수집 및 전처리, AI 모델 개발(법령 RAG 임베딩·검색 파이프라인 포함), 테스트 및 품질 개선, 문서화 및 산출물 관리 |
| 정다솔 | [@soll07](https://github.com/soll07) | 데이터 수집 및 전처리, 프로젝트 기획 및 관리, AI 모델 개발, 데이터베이스 및 시스템 설계, 인프라 배포 및 운영 |
| 홍진서 | [@Hong-Jin-seo](https://github.com/Hong-Jin-seo) | 데이터 수집 및 전처리, 백엔드 개발, 프론트엔드 개발, 테스트 및 품질 개선, 문서화 및 산출물 관리 |