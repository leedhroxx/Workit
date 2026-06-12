# setup_env.py
# 팀원 환경 세팅 스크립트 - 프로젝트 루트에서 python setup_env.py 실행

import os
import subprocess
import sys

def patch_exaone_cache():
    """EXAONE 캐시 파일 패치 (transformers 4.49.0 호환)"""
    base = os.path.expanduser(
        r'~/.cache/huggingface/modules/transformers_modules/'
        r'LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct/'
        r'ccce25bd39c141fe053e0bc75818a8f5fe962802'
    )

    # configuration_exaone.py 패치
    config_path = os.path.join(base, 'configuration_exaone.py')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'from transformers.modeling_rope_utils import RopeParameters' in content and 'try:' not in content:
            content = content.replace(
                'from transformers.modeling_rope_utils import RopeParameters',
                'try:\n    from transformers.modeling_rope_utils import RopeParameters\nexcept ImportError:\n    class RopeParameters: pass'
            )
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print('✅ configuration_exaone.py 패치 완료')
        else:
            print('✅ configuration_exaone.py 이미 패치됨')
    else:
        print('⚠️  configuration_exaone.py 없음 - 모델 첫 실행 시 자동 다운로드 후 다시 실행하세요')

    # modeling_exaone.py 패치
    model_path = os.path.join(base, 'modeling_exaone.py')
    if os.path.exists(model_path):
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        target = 'from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func'
        if target in content and 'try:' not in content.split(target)[0].split('\n')[-2]:
            content = content.replace(
                target,
                'try:\n    from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func\nexcept ImportError:\n    def use_kernel_forward_from_hub(*a, **kw): return lambda f: f\n    def use_kernel_func_from_hub(*a, **kw): return lambda f: f\n    def use_kernelized_func(*a, **kw): return lambda f: f'
            )
            with open(model_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print('✅ modeling_exaone.py 패치 완료')
        else:
            print('✅ modeling_exaone.py 이미 패치됨')
    else:
        print('⚠️  modeling_exaone.py 없음 - 모델 첫 실행 시 자동 다운로드 후 다시 실행하세요')


def check_redis():
    """Redis 서버 확인"""
    redis_path = r'C:\Program Files\Redis\redis-cli.exe'
    if os.path.exists(redis_path):
        result = subprocess.run([redis_path, 'ping'], capture_output=True, text=True)
        if 'PONG' in result.stdout:
            print('✅ Redis 실행 중')
        else:
            print('⚠️  Redis 설치됨, 서버 미실행 - redis-server.exe 실행 필요')
    else:
        print('❌ Redis 미설치 - https://github.com/tporadowski/redis/releases 에서 Redis-x64-5.0.14.1.msi 설치')


def check_poppler():
    """poppler 확인"""
    poppler_path = r'C:\poppler-24.08.0\Library\bin\pdftoppm.exe'
    if os.path.exists(poppler_path):
        print('✅ poppler 설치됨')
    else:
        print('❌ poppler 미설치 - https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0 에서')
        print('   Release-24.08.0-0.zip 받아서 C:\\poppler-24.08.0\\ 에 압축 풀기')


def check_model():
    """모델 파일 확인"""
    model_path = os.path.join(os.path.dirname(__file__), 'data', 'jihye_sft', 'model_output', 'adapter_config.json')
    if os.path.exists(model_path):
        print('✅ 모델 파일 존재')
    else:
        print('❌ 모델 파일 없음 - 구글 드라이브에서 model_output 폴더를 data/jihye_sft/model_output/ 에 넣기')


def check_qdrant():
    """Qdrant 벡터스토어 확인"""
    qdrant_path = os.path.join(os.path.dirname(__file__), 'vectorstore', 'qdrant_storage', 'collection')
    if os.path.exists(qdrant_path):
        print('✅ Qdrant 벡터스토어 존재')
    else:
        print('❌ Qdrant 없음 - 구글 드라이브에서 qdrant_storage 폴더를 vectorstore/ 에 넣기')


if __name__ == '__main__':
    print('=' * 50)
    print('Workit 환경 세팅 스크립트')
    print('=' * 50)

    print('\n[1] EXAONE 캐시 패치')
    patch_exaone_cache()

    print('\n[2] Redis 확인')
    check_redis()

    print('\n[3] poppler 확인')
    check_poppler()

    print('\n[4] 모델 파일 확인')
    check_model()

    print('\n[5] Qdrant 벡터스토어 확인')
    check_qdrant()

    print('\n' + '=' * 50)
    print('❌ 항목이 있으면 해당 안내에 따라 설치 후 다시 실행하세요')
    print('✅ 모두 완료되면 아래 순서로 서버 실행:')
    print('  1. redis-server.exe 실행')
    print('  2. celery -A config worker --loglevel=info --pool=solo')
    print('  3. python manage.py runserver')
    print('=' * 50)