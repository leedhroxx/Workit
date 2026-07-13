from django.db import models
from contracts.models import Contract
from django.contrib.auth import get_user_model


class Performance(models.Model):
    contract = models.OneToOneField(Contract, on_delete=models.CASCADE, related_name='performance')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = '이행 관리'
        verbose_name_plural = '이행 관리 목록'

    def __str__(self):
        return f"{self.contract.project_name} 이행관리"

    def progress_count(self):
        valid_types = [t for t, _ in Deliverable.TYPE_CHOICES]
        return self.deliverables.filter(
            status='submitted',
            deliverable_type__in=valid_types,
        ).count()

    def total_count(self):
        return 3  # 사업수행계획서, 기술적용결과표, 사업추진결과보고서

    def next_deliverable_label(self):
        existing = {d.deliverable_type: d for d in self.deliverables.all()}
        for t, label in Deliverable.TYPE_CHOICES:
            d = existing.get(t)
            if not d or d.status == 'pending':
                return label
        return ''


class Deliverable(models.Model):
    TYPE_CHOICES = [
        ('kickoff', '사업수행계획서'),
        ('tech_apply',  '기술적용결과표'),
        ('final', '사업추진결과보고서'),
    ]
    STATUS_CHOICES = [
        ('pending', '미등록'),
        ('submitted', '제출완료'),
    ]
    TYPE_ORDER = ['kickoff', 'tech_apply', 'final']

    performance = models.ForeignKey(Performance, on_delete=models.CASCADE, related_name='deliverables')
    deliverable_type = models.CharField('산출물 유형', max_length=20, choices=TYPE_CHOICES)
    file = models.FileField('파일', upload_to='performance/deliverables/', blank=True, null=True)
    original_filename = models.CharField('원본 파일명', max_length=255, blank=True)
    due_date = models.DateField('제출 예정일', null=True, blank=True)
    submitted_date = models.DateField('실제 제출일', null=True, blank=True)
    status = models.CharField('상태', max_length=20, choices=STATUS_CHOICES, default='pending')

    class Meta:
        verbose_name = '산출물'
        verbose_name_plural = '산출물 목록'
        ordering = ['deliverable_type']

    def __str__(self):
        return f"{self.performance.contract.project_name} - {self.get_deliverable_type_display()}"

    def filename(self):
        return self.original_filename or (self.file.name.split('/')[-1] if self.file else '')

    def type_order(self):
        return self.TYPE_ORDER.index(self.deliverable_type) if self.deliverable_type in self.TYPE_ORDER else 99

User = get_user_model()

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.CharField('메시지', max_length=255)
    url = models.CharField('이동 경로', max_length=255, blank=True, default='/performance/')
    is_read = models.BooleanField('읽음 여부', default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'notifications'
        ordering = ['-created_at']
        verbose_name = '알림'
        verbose_name_plural = '알림 목록'

    def __str__(self):
        return f"[{self.user}] {self.message}"

class ExecutionPlanParsedData(models.Model):
    """
    과업수행계획서(Deliverable) 파싱한 정형화 데이터.
 
    - 파싱 시점: 과업수행계획서 파일 업로드 직후 비동기
    - parse_status 흐름: pending → processing → done | failed
    """
 
    deliverable = models.OneToOneField(
        'Deliverable',
        on_delete=models.CASCADE,
        related_name='parsed_data',
        # deliverable_type == 'execution_plan' 인 레코드에 연결됨
    )
    # PEP 코드 체계(PEP-01 ~ PEP-16) 기반 정형화 JSON
    parsed_json = models.JSONField(default=dict, blank=True)

    # 소제목 매핑 QA 검수 리포트 (LLM/qa_agent.review_section_mapping 결과)
    qa_report = models.JSONField('QA 검수 리포트', default=dict, blank=True)

    parse_status = models.CharField(
        max_length=20,
        choices=[
            ('pending',    '대기'),
            ('processing', '처리중'),
            ('done',       '완료'),
            ('failed',     '실패'),
        ],
        default='pending',
    )
    parsed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
 
    class Meta:
        verbose_name = '과업수행계획서 파싱 결과'
        verbose_name_plural = '과업수행계획서 파싱 결과 목록'
 
    def __str__(self):
        return f'PEP 파싱 — {self.deliverable} [{self.parse_status}]'
 
 
class RFPComparisonResult(models.Model):
    """
    RFP ↔ 과업수행계획서 AI 비교 분석 결과.
 
    - 비교 시점: 프론트의 "비교 분석" 버튼 클릭
    - 새 비교를 실행할 때마다 이전 결과는 삭제하고 최신 1건만 유지
    """

    STATUS_CHOICES = [
        ('idle',       '대기'),
        ('processing', '분석중'),
        ('done',       '완료'),
        ('failed',     '실패'),
    ]

    performance = models.ForeignKey(
        'Performance',
        on_delete=models.CASCADE,
        related_name='rfp_comparisons',
    )
    rfp_parsed = models.ForeignKey(
        # contracts 앱 모델 참조 — 앱이 분리돼 있으면 문자열로 지정
        'contracts.RFPParsedData',
        on_delete=models.SET_NULL,
        null=True,
        related_name='+',
    )
    execution_plan_parsed = models.ForeignKey(
        ExecutionPlanParsedData,
        on_delete=models.SET_NULL,
        null=True,
        related_name='+',
    )
    # 비교 분석 JSON
    # {"overall_score": 85, "summary": "...", "satisfied": [...], "partial": [...], "unsatisfied": [...]}
    comparison_json = models.JSONField(default=dict)
    status = models.CharField('분석 상태', max_length=20, choices=STATUS_CHOICES, default='idle')
    task_id = models.CharField('Celery task id', max_length=255, blank=True, default='')
    started_at = models.DateTimeField('분석 시작 시각', null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
 
    class Meta:
        verbose_name = 'RFP 비교 결과'
        verbose_name_plural = 'RFP 비교 결과 목록'
        ordering = ['-created_at']
 
    def __str__(self):
        return f'비교 결과 — {self.performance} ({self.created_at:%Y-%m-%d})'


class TechApplyCheckResult(models.Model):
    """
    기술적용결과표(tech_apply) 산출물의 체크박스 정합성 검증 결과.

    - 검증 시점: 산출물 분석 화면에서 "AI 분석 시작" 클릭
    - 파일이 새로 업로드되면 이 결과는 폐기되고, 재분석 전까지는
      분석 화면이 "분석 시작" 초기 상태부터 다시 노출된다.
    """

    deliverable = models.OneToOneField(
        'Deliverable',
        on_delete=models.CASCADE,
        related_name='tech_apply_result',
    )
    # check_tech_apply() 반환값 그대로: {"total", "error_count", "items"}
    result_json = models.JSONField(default=dict, blank=True)
    checked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = '기술적용결과표 검증 결과'
        verbose_name_plural = '기술적용결과표 검증 결과 목록'

    def __str__(self):
        return f'기술적용결과표 검증 — {self.deliverable}'


class FinalReportParsedData(models.Model):
    """
    사업추진결과보고서(final Deliverable) 파싱한 정형화 데이터.

    - 파싱 시점: 산출물 분석 화면에서 "분석 시작" 클릭
    - parse_status 흐름: pending → processing → done | failed
    - ExecutionPlanParsedData(PEP)와 완전히 같은 역할을 RPT 문서에 대해 수행한다.
    """

    deliverable = models.OneToOneField(
        'Deliverable',
        on_delete=models.CASCADE,
        related_name='final_parsed_data',
    )
    # RPT 코드 체계(RPT-01-01 ~ RPT-03-02) 기반 정형화 JSON
    parsed_json = models.JSONField(default=dict, blank=True)

    # 소제목 매핑 QA 검수 리포트 (LLM/qa_agent.review_section_mapping 결과)
    qa_report = models.JSONField('QA 검수 리포트', default=dict, blank=True)

    parse_status = models.CharField(
        max_length=20,
        choices=[
            ('pending',    '대기'),
            ('processing', '처리중'),
            ('done',       '완료'),
            ('failed',     '실패'),
        ],
        default='pending',
    )
    parsed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        verbose_name = '사업추진결과보고서 파싱 결과'
        verbose_name_plural = '사업추진결과보고서 파싱 결과 목록'

    def __str__(self):
        return f'RPT 파싱 — {self.deliverable} [{self.parse_status}]'


class PEPFinalComparisonResult(models.Model):
    """
    사업수행계획서(PEP, 계획) ↔ 사업추진결과보고서(RPT, 실적) 구조적 비교 분석 결과.

    RFP 대비 이행 여부는 이미 RFPComparisonResult(PEP ↔ RFP)에서 확인하므로,
    여기서는 "계획한 대로 실제로 이행됐는지"를 PEP 대비로 비교한다.

    - 비교 시점: QA 검수(1단계) 완료 후 "그대로 진행" 클릭
    - 새 비교를 실행할 때마다 이전 결과는 삭제하고 최신 1건만 유지
    """
    
    STATUS_CHOICES = [
        ('idle',       '대기'),
        ('processing', '분석중'),
        ('done',       '완료'),
        ('failed',     '실패'),
    ]


    performance = models.ForeignKey(
        'Performance',
        on_delete=models.CASCADE,
        related_name='pep_final_comparisons',
    )
    execution_plan_parsed = models.ForeignKey(
        ExecutionPlanParsedData,
        on_delete=models.SET_NULL,
        null=True,
        related_name='+',
    )
    final_report_parsed = models.ForeignKey(
        FinalReportParsedData,
        on_delete=models.SET_NULL,
        null=True,
        related_name='+',
    )
    # 비교 분석 JSON (RFPComparisonResult.comparison_json과 같은 형식)
    comparison_json = models.JSONField(default=dict)
    status = models.CharField('분석 상태', max_length=20, choices=STATUS_CHOICES, default='idle')
    task_id = models.CharField('Celery task id', max_length=255, blank=True, default='')
    started_at = models.DateTimeField('분석 시작 시각', null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = '사업수행계획서 ↔ 결과보고서 비교 결과'
        verbose_name_plural = '사업수행계획서 ↔ 결과보고서 비교 결과 목록'
        ordering = ['-created_at']

    def __str__(self):
        return f'PEP-결과보고서 비교 — {self.performance} ({self.created_at:%Y-%m-%d})'


class AIAnalysisLog(models.Model):
    """
    산출물 AI 분석(1단계 QA 검수·2단계 비교) 과정에서 발생하는 이벤트를 DB에 남긴다.

    - 파싱/QA 완료 시: 이슈 유무(analysis_ok / analysis_issue)
    - "비교분석 진행" 클릭 시: 이슈가 있었는데도 반려 없이 진행했는지 여부
      (proceed_no_issue / proceed_with_issue)
    - "반려(다시 업로드)" 클릭 시: reject

    콘솔 로그(logging)는 서버 재시작 시 사라지므로, 나중에 "어떤 산출물이 몇 번
    반려됐는지" 등을 조회·집계할 수 있도록 별도 테이블로 남긴다.
    """

    EVENT_CHOICES = [
        ('analysis_ok', '분석 완료 - 이슈 없음'),
        ('analysis_issue', '분석 완료 - 이슈 발견'),
        ('proceed_no_issue', '이슈 없이 다음 단계 진행'),
        ('proceed_with_issue', '이슈 있었지만 반려 없이 진행'),
        ('reject', '반려 (다시 업로드)'),
    ]

    deliverable = models.ForeignKey(
        'Deliverable', on_delete=models.CASCADE, related_name='ai_analysis_logs',
    )
    event_type = models.CharField('이벤트 유형', max_length=30, choices=EVENT_CHOICES)
    issue_count = models.IntegerField('이슈 건수', default=0)
    # 이슈 유형 목록(issue_type들), review_status 등 부가 정보
    detail = models.JSONField('상세', default=dict, blank=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField('발생 시각', auto_now_add=True)

    class Meta:
        verbose_name = 'AI 분석 로그'
        verbose_name_plural = 'AI 분석 로그 목록'
        ordering = ['-created_at']

    def __str__(self):
        return f'[{self.get_event_type_display()}] {self.deliverable} ({self.created_at:%Y-%m-%d %H:%M})'

    @classmethod
    def log(cls, deliverable, event_type, issue_count=0, detail=None, user=None):
        """어디서 호출하든(celery task는 user 없이, view는 request.user와 함께) 같은 형태로 기록한다."""
        return cls.objects.create(
            deliverable=deliverable,
            event_type=event_type,
            issue_count=issue_count,
            detail=detail or {},
            user=user,
        )