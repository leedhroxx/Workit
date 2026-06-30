from functools import wraps

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.sessions.models import Session
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from contracts.models import Contract
from performance.models import Performance

from .forms import AdminUserCreationForm, OrganizationCreateForm, WorkitPasswordChangeForm
from .models import Organization, User

from django.views.decorators.http import require_POST


def _delete_session(session_key):
    if session_key:
        Session.objects.filter(session_key=session_key).delete()


def _check_single_session_conflict(request, user):
    existing_session_key = user.current_session_key
    if not existing_session_key or existing_session_key == request.session.session_key:
        return False

    # 기존: .exists() → 만료된 세션도 True 반환해서 오판
    # 수정: expire_date__gt=now → 실제로 살아있는 세션만 True
    session_alive = Session.objects.filter(
        session_key=existing_session_key,
        expire_date__gt=timezone.now()
    ).exists()

    if not session_alive:
        user.current_session_key = None
        user.save(update_fields=["current_session_key"])
        return False

    return True


def _finalize_login(request, user):
    """기존 세션 종료(필요 시) 후 실제 로그인 처리."""
    existing_session_key = user.current_session_key
    if existing_session_key and existing_session_key != request.session.session_key:
        _delete_session(existing_session_key)

    login(request, user)
    if not request.session.session_key:
        request.session.save()
    user.register_login_success(request.session.session_key)

    if user.must_change_password:
        messages.info(request, '최초 로그인 또는 관리자 초기화 계정입니다. 비밀번호를 먼저 변경하세요.')
        return redirect('change_password')

    return redirect('home')


def login_view(request):
    if request.user.is_authenticated:
        if request.user.must_change_password:
            return redirect('change_password')
        return redirect('home')

    if request.method == 'POST':
        # ── 확인창에서 "계속" / "취소"를 누른 2차 제출 처리 ──
        if request.POST.get('confirm_force_login') == '1':
            pending_user_id = request.session.pop('pending_force_login_user_id', None)
            user = User.objects.filter(pk=pending_user_id).first() if pending_user_id else None
            if not user:
                messages.error(request, '로그인 확인 시간이 초과되었습니다. 다시 로그인해 주세요.')
                return redirect('login')
            return _finalize_login(request, user)

        if request.POST.get('cancel_force_login') == '1':
            request.session.pop('pending_force_login_user_id', None)
            return redirect('login')

        # ── 일반적인 1차 로그인 제출 ──
        request.session.pop('pending_force_login_user_id', None)  # 이전에 남아있던 대기 상태 정리

        username = (request.POST.get('username') or '').strip()
        password = request.POST.get('password') or ''
        candidate = User.objects.filter(username=username).first()

        if candidate and candidate.is_locked:
            messages.error(request, '잠긴 계정입니다. 관리자에게 잠금해제를 요청하세요.')
            return render(request, 'accounts/login.html', {'username': username})

        if candidate and not candidate.is_active:
            messages.error(request, '비활성화된 계정입니다. 관리자에게 문의하세요.')
            return render(request, 'accounts/login.html', {'username': username})

        user = authenticate(request, username=username, password=password)

        if not user:
            if candidate and candidate.is_active and not candidate.is_locked:
                candidate.register_login_failure()
                max_attempts = getattr(settings, 'ACCOUNTS_MAX_FAILED_LOGIN_ATTEMPTS', 5)
                remaining = max_attempts - candidate.failed_login_attempts
                if candidate.is_locked:
                    messages.error(request, '로그인 5회 연속 실패로 계정이 잠겼습니다. 관리자에게 잠금해제를 요청하세요.')
                else:
                    messages.error(request, f'아이디 또는 비밀번호가 올바르지 않습니다. 남은 시도 횟수: {remaining}회')
            else:
                messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')
            return render(request, 'accounts/login.html', {'username': username})

        if user.is_password_expired:
            user.lock(User.LOCK_REASON_PASSWORD_EXPIRED)
            messages.error(request, '비밀번호 사용기간 90일이 만료되어 계정이 잠겼습니다. 관리자에게 잠금해제를 요청하세요.')
            return render(request, 'accounts/login.html', {'username': username})

        suppress_single_session_message = request.COOKIES.get('workit_password_changed_recently') == '1'
        policy = getattr(settings, 'ACCOUNTS_SINGLE_SESSION_POLICY', 'KILL_OLD')
        conflict = _check_single_session_conflict(request, user)

        if conflict and policy == 'BLOCK_NEW':
            if not suppress_single_session_message:
                messages.error(request, '이미 동일 계정으로 접속 중입니다. 기존 세션을 종료한 뒤 다시 로그인하세요.')
            response = render(request, 'accounts/login.html', {'username': username})
            if suppress_single_session_message:
                response.delete_cookie('workit_password_changed_recently')
            return response

        if conflict:  # policy == KILL_OLD → 확인창 띄우기
            request.session['pending_force_login_user_id'] = user.pk
            request.session.save()
            return render(request, 'accounts/login.html', {
                'username': username,
                'need_session_confirm': True,
            })

        return _finalize_login(request, user)

    return render(request, 'accounts/login.html')


def logout_view(request):
    if request.user.is_authenticated and request.user.current_session_key == request.session.session_key:
        request.user.current_session_key = None
        request.user.save(update_fields=['current_session_key'])
    logout(request)
    return redirect('login')


@login_required
def change_password_view(request):
    if request.method == 'POST':
        form = WorkitPasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # update_session_auth_hash()가 세션 키를 갱신할 수 있으므로,
            # 갱신 이후의 세션 키를 사용자 current_session_key에 저장해야
            # 비밀번호 변경 직후 동일계정 동시접속으로 오인되지 않는다.
            update_session_auth_hash(request, user)
            if not request.session.session_key:
                request.session.save()
            user.mark_password_changed(request.session.session_key)
            messages.success(request, '비밀번호가 변경되었습니다. 마이페이지에서 계정 정보를 확인하세요.')
            response = redirect('mypage')
            response.set_cookie(
                'workit_password_changed_recently',
                '1',
                max_age=120,
                httponly=True,
                samesite='Lax',
            )
            return response
        messages.error(request, '비밀번호 변경 내용을 확인하세요.')
    else:
        form = WorkitPasswordChangeForm(request.user)

    return render(request, 'accounts/change_password.html', {'form': form})


@login_required
def home_view(request):
    contracts = Contract.objects.filter(created_by=request.user).order_by('-created_at')
    performances = Performance.objects.filter(contract__created_by=request.user).order_by('-created_at')

    total = contracts.count()
    in_review = contracts.filter(status='reviewing').count()
    in_progress = contracts.filter(status='in_progress').count()
    completed = contracts.filter(status='completed').count()

    context = {
        'total': total,
        'in_review': in_review,
        'in_progress': in_progress,
        'completed': completed,
        'recent_contracts': contracts[:5],
        'recent_performances': performances[:5],
    }
    return render(request, 'home.html', context)


@login_required
def mypage_view(request):
    return render(request, 'mypage/mypage.html', {'user': request.user})


@login_required
def mypage_update(request):
    if request.method == 'POST':
        required_fields = {
            'last_name': '성',
            'first_name': '이름',
            'email': '이메일',
            'phone': '전화번호',
            'department': '부서',
            'position': '직급',
        }
        missing = [label for field, label in required_fields.items() if not (request.POST.get(field) or '').strip()]
        if missing:
            return JsonResponse({'status': 'error', 'message': f"필수 항목을 입력하세요: {', '.join(missing)}"}, status=400)

        user = request.user
        user.first_name = request.POST.get('first_name').strip()
        user.last_name = request.POST.get('last_name').strip()
        user.email = request.POST.get('email').strip()
        user.phone = request.POST.get('phone').strip()
        user.department = request.POST.get('department').strip()
        user.position = request.POST.get('position').strip()
        user.save(update_fields=['first_name', 'last_name', 'email', 'phone', 'department', 'position'])
        return JsonResponse({'status': 'ok', 'message': '정보가 수정되었습니다.'})

    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다.'}, status=400)


@login_required
def help_page(request):
    return render(request, 'help/help.html')


@login_required
@require_POST
def toggle_notification(request):
    user = request.user
    user.notification_enabled = not user.notification_enabled
    user.save(update_fields=['notification_enabled'])
    return JsonResponse({
        'status': 'ok',
        'notification_enabled': user.notification_enabled,
    })


# ──────────────────────────────────────────────
# 관리자(is_superuser) 전용 계정/부서 관리
# ──────────────────────────────────────────────

def system_admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_system_admin:
            messages.error(request, '접근 권한이 없습니다.')
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return wrapper


@system_admin_required
def account_list_view(request):
    accounts = (
        User.objects.select_related('organization')
        .exclude(is_superuser=True)
        .order_by('organization__name', 'username')
    )
    return render(request, 'accounts/account_list.html', {'accounts': accounts})


@system_admin_required
def account_create_view(request):
    if request.method == 'POST':
        form = AdminUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(
                request,
                f'{user.korean_name()}님의 계정이 생성되었습니다. 초기 비밀번호로 로그인 후 변경이 필요합니다.',
            )
            return redirect('account_list')
        messages.error(request, '입력 내용을 확인하세요.')
    else:
        form = AdminUserCreationForm()

    return render(request, 'accounts/account_create.html', {'form': form})


@system_admin_required
@require_POST
def account_lock_toggle_view(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if target.is_system_admin:
        messages.error(request, '관리자 계정은 잠금 처리할 수 없습니다.')
        return redirect('account_list')

    if target.is_locked:
        target.unlock()
        messages.success(request, f'{target.korean_name()}님의 계정 잠금이 해제되었습니다.')
    else:
        target.lock(User.LOCK_REASON_ADMIN)
        messages.success(request, f'{target.korean_name()}님의 계정을 잠금 처리했습니다.')
    return redirect('account_list')


@system_admin_required
def organization_list_view(request):
    if request.method == 'POST':
        form = OrganizationCreateForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, '부서가 추가되었습니다.')
            return redirect('organization_list')
        messages.error(request, '입력 내용을 확인하세요.')
    else:
        form = OrganizationCreateForm()

    organizations = Organization.objects.all()
    return render(request, 'accounts/organization_list.html', {
        'organizations': organizations,
        'form': form,
    })