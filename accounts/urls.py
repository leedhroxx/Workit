from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('password/change/', views.change_password_view, name='change_password'),
    path('mypage/', views.mypage_view, name='mypage'),
    path('mypage/update/', views.mypage_update, name='mypage_update'),
    path('help/', views.help_page, name='help'),
    path('notification/toggle/', views.toggle_notification, name='toggle_notification'),

    # ── 관리자(is_superuser) 전용 계정/부서 관리 ──
    path('manage/accounts/', views.account_list_view, name='account_list'),
    path('manage/accounts/create/', views.account_create_view, name='account_create'),
    path('manage/accounts/<int:user_id>/lock-toggle/', views.account_lock_toggle_view, name='account_lock_toggle'),
    path('manage/organizations/', views.organization_list_view, name='organization_list'),
]