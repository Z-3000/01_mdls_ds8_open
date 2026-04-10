# helper/apikeys.py — API 키 로드 (로컬 우선, Colab 폴백)
# ============================================================================
# 사용법:
#   from helper import apikeys
#
#   # 단일 키 로드 (탐색 순서: .env 파일 → 환경변수 → Colab 보안 비밀)
#   key = apikeys.get("OPENAI_API_KEY")
#
#   # 로드 + 환경변수에 자동 등록 (라이브러리가 os.environ에서 읽는 경우)
#   key = apikeys.get("OPENAI_API_KEY", set_env=True)
#
#   # 자주 쓰는 키 한 번에 로드
#   keys = apikeys.load_all()
#   keys = apikeys.load_all(set_env=True)
#   # → keys["OPENAI_API_KEY"], keys["HF_TOKEN"] 등
#
#   # 커스텀 .env 파일 경로
#   key = apikeys.get("MY_KEY", env_path="/home/user/secrets/.env")
#
#   # 현재 등록된 키 확인 (값은 마스킹)
#   apikeys.show()
#
#   # HuggingFace 로그인 헬퍼
#   apikeys.login_hf()
# ============================================================================

import os


# ── 기본 키 목록 (필요에 따라 여기서 추가/수정) ──
DEFAULT_KEYS = [
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "LANGCHAIN_API_KEY",
    "WANDB_API_KEY",
]


# ============================================================================
# 환경 감지
# ============================================================================
def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================
# .env 파일 파싱 (python-dotenv 없이도 동작)
# ============================================================================
def _parse_env_file(env_path: str) -> dict:
    """
    .env 파일을 읽어서 {KEY: VALUE} 딕셔너리로 반환합니다.
    python-dotenv가 설치되어 있으면 사용하고, 없으면 직접 파싱합니다.

    지원 형식:
        KEY=value
        KEY="value"
        KEY='value'
        export KEY=value
        # 주석
    """
    if not os.path.isfile(env_path):
        return {}

    # python-dotenv 사용 시도
    try:
        from dotenv import dotenv_values
        return dict(dotenv_values(env_path))
    except ImportError:
        pass

    # 직접 파싱
    result = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # export 접두어 제거
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 따옴표 제거
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            result[key] = value

    return result


# ============================================================================
# .env 파일 탐색
# ============================================================================
def _find_env_file(env_path: str = "auto") -> str:
    """
    .env 파일 위치를 탐색합니다.
    탐색 순서: 지정 경로 → 현재 디렉토리 → 상위 디렉토리 (2단계까지)
    """
    if env_path != "auto":
        return env_path

    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), "..", ".env"),
        os.path.join(os.getcwd(), "..", "..", ".env"),
        os.path.expanduser("~/.env"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    return os.path.join(os.getcwd(), ".env")  # 기본값 (없어도 에러 안 남)


# ============================================================================
# 핵심 함수: 단일 키 로드
# ============================================================================
def get(
    key_name: str,
    env_path: str = "auto",
    set_env: bool = False,
    required: bool = False,
    verbose: bool = False,
) -> str | None:
    """
    API 키를 로드합니다.

    탐색 순서:
        1. .env 파일
        2. 환경변수 (os.environ)
        3. Colab 보안 비밀 (google.colab.userdata)

    Parameters
    ----------
    key_name : str
        키 이름 (예: "OPENAI_API_KEY")
    env_path : str
        .env 파일 경로 (기본 "auto" → 자동 탐색)
    set_env : bool
        True면 로드한 키를 os.environ에 등록 (기본 False)
    required : bool
        True면 키를 못 찾았을 때 에러 발생 (기본 False)
    verbose : bool
        출처 출력 (기본 False)

    Returns
    -------
    str | None : 키 값 (못 찾으면 None, required=True면 에러)
    """
    value = None
    source = None

    # 1) .env 파일
    resolved_path = _find_env_file(env_path)
    env_data = _parse_env_file(resolved_path)
    if key_name in env_data:
        value = env_data[key_name]
        source = f".env ({resolved_path})"

    # 2) 환경변수
    if value is None and key_name in os.environ:
        value = os.environ[key_name]
        source = "환경변수"

    # 3) Colab 보안 비밀
    if value is None and _is_colab():
        try:
            from google.colab import userdata  # type: ignore
            value = userdata.get(key_name)
            source = "Colab 보안 비밀"
        except Exception:
            pass

    # 결과 처리
    if value is None:
        if required:
            raise KeyError(
                f"[apikeys] '{key_name}'을(를) 찾을 수 없습니다.\n"
                f"  확인한 위치:\n"
                f"    1. .env 파일: {resolved_path}\n"
                f"    2. 환경변수 (os.environ)\n"
                f"    3. Colab 보안 비밀\n"
                f"  팁: .env 파일에 {key_name}=your_key_here 를 추가하세요."
            )
        if verbose:
            print(f"[apikeys] ✗ {key_name} — 찾을 수 없음")
        return None

    # 환경변수에 등록
    if set_env:
        os.environ[key_name] = value

    if verbose:
        print(f"[apikeys] ✓ {key_name} — 출처: {source}")

    return value


# ============================================================================
# 여러 키 한 번에 로드
# ============================================================================
def load_all(
    keys: list | None = None,
    env_path: str = "auto",
    set_env: bool = False,
    verbose: bool = True,
) -> dict:
    """
    여러 API 키를 한 번에 로드합니다.

    Parameters
    ----------
    keys : list[str] | None
        로드할 키 이름 리스트 (기본 None → DEFAULT_KEYS 사용)
    set_env : bool
        True면 찾은 키를 모두 os.environ에 등록
    verbose : bool
        결과 요약 출력 (기본 True)

    Returns
    -------
    dict : {키이름: 값} (못 찾은 키는 포함하지 않음)
    """
    if keys is None:
        keys = DEFAULT_KEYS

    result = {}
    found = []
    missing = []

    for key_name in keys:
        value = get(key_name, env_path=env_path, set_env=set_env, verbose=False)
        if value is not None:
            result[key_name] = value
            found.append(key_name)
        else:
            missing.append(key_name)

    if verbose:
        print(f"[apikeys] 로드 완료: {len(found)}/{len(keys)}")
        for k in found:
            print(f"  ✓ {k}")
        for k in missing:
            print(f"  ✗ {k}")

    return result


# ============================================================================
# 상태 확인
# ============================================================================
def show(keys: list | None = None, env_path: str = "auto"):
    """
    현재 로드 가능한 키를 확인합니다 (값은 마스킹).

    사용법:
        apikeys.show()
        apikeys.show(keys=["OPENAI_API_KEY", "MY_CUSTOM_KEY"])
    """
    if keys is None:
        keys = DEFAULT_KEYS

    resolved_path = _find_env_file(env_path)
    print(f"[apikeys] .env 경로: {resolved_path} "
          f"({'존재' if os.path.isfile(resolved_path) else '없음'})")
    print(f"[apikeys] 환경: {'Colab' if _is_colab() else '로컬'}")
    print("-" * 50)

    for key_name in keys:
        value = get(key_name, env_path=env_path, verbose=False)
        if value:
            masked = value[:4] + "…" + value[-4:] if len(value) > 12 else "****"
            print(f"  ✓ {key_name} = {masked}")
        else:
            print(f"  ✗ {key_name}")


# ============================================================================
# HuggingFace 로그인 헬퍼
# ============================================================================
def login_hf(token: str | None = None, env_path: str = "auto"):
    """
    HuggingFace Hub에 로그인합니다.
    token이 없으면 HF_TOKEN을 자동으로 찾아서 사용합니다.

    사용법:
        apikeys.login_hf()                    # HF_TOKEN 자동 탐색
        apikeys.login_hf(token="hf_xxxxx")    # 직접 지정
    """
    if token is None:
        token = get("HF_TOKEN", env_path=env_path, required=True)

    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("[apikeys] ✓ HuggingFace 로그인 완료")
