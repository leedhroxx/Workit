from django.contrib import admin
from .models import Performance, Deliverable, AIAnalysisLog

@admin.register(Performance)
class PerformanceAdmin(admin.ModelAdmin):
    list_display = ['contract', 'created_at']

admin.site.register(Deliverable)

@admin.register(AIAnalysisLog)
class AIAnalysisLogAdmin(admin.ModelAdmin):
    list_display = ['deliverable', 'event_type', 'issue_count', 'user', 'created_at']
    list_filter = ['event_type', 'deliverable__deliverable_type']
    readonly_fields = ['deliverable', 'event_type', 'issue_count', 'detail', 'user', 'created_at']
