from django.contrib import admin
from .models import Performance, Deliverable, AIAnalysisLog, AIPerfLog

@admin.register(Performance)
class PerformanceAdmin(admin.ModelAdmin):
    list_display = ['contract', 'created_at']

admin.site.register(Deliverable)

@admin.register(AIAnalysisLog)
class AIAnalysisLogAdmin(admin.ModelAdmin):
    list_display = ['deliverable', 'event_type', 'issue_count', 'user', 'created_at']
    list_filter = ['event_type', 'deliverable__deliverable_type']
    readonly_fields = ['deliverable', 'event_type', 'issue_count', 'detail', 'user', 'created_at']

@admin.register(AIPerfLog)
class AIPerfLogAdmin(admin.ModelAdmin):
    list_display = ['feature', 'duration_seconds', 'success', 'contract_document', 'performance', 'deliverable', 'created_at']
    list_filter = ['feature', 'success']
    date_hierarchy = 'created_at'
    readonly_fields = [
        'feature', 'started_at', 'finished_at', 'duration_seconds', 'success',
        'contract_document', 'performance', 'deliverable', 'context', 'created_at',
    ]
