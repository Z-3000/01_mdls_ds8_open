# helper/envcheck.py — 환경 버전 확인 & 비교
# ============================================================================
# 사용법:
#   from helper import envcheck
#
#   envcheck.show_versions()                    # 핵심 패키지 버전 출력
#   envcheck.export_requirements()              # requirements_helper.txt 생성
#   envcheck.compare_requirements("req.txt")    # 현재 환경 vs 파일 비교
# ============================================================================

from __future__ import annotations      # 파이썬 하위 버전에서도 최신 타입 힌트(예: | ) 사용 가능케 함

import sys                              # 파이썬 인터프리터 제어 및 실행 환경 정보 조회
import os                               # 운영체제 상호작용 (경로, 환경 변수 등) 처리
import platform                         # 하드웨어 및 OS의 상세 플랫폼 정보(커널, 비트 등) 확인
from importlib.metadata import version as pkg_version, PackageNotFoundError  # 설치된 패키지의 메타데이터(버전) 조회 도구
from datetime import datetime                                                # 점검 시점 기록을 위한 날짜 및 시간 라이브러리


# ── 데이터 분석/ML 핵심 패키지 (도메인별 그룹핑 관리) ──
# 이 딕셔너리는 프로젝트에서 사용 중인 기술 스택의 '화이트리스트' 역할을 수행함
CORE_PACKAGES: dict[str, list[str]] = {
    "데이터 기본": [
        "pandas", "numpy", "scipy",                             
    ],
    "시각화": [
        "matplotlib", "seaborn", "plotly",                      
    ],
    "ML / DL": [
        "scikit-learn", "torch", "torchvision", "torchaudio",    # 머신러닝 및 PyTorch 기반 딥러닝 프레임워크
        "tensorflow", "xgboost", "lightgbm",                     # TF 및 고성능 부스팅 알고리즘 패키지
    ],
    "NLP / LLM": [
        "transformers", "tokenizers", "datasets",                # Hugging Face 기반 자연어 처리 생태계
        "langchain", "langchain-core", "langchain-community",    # LLM 오케스트레이션 및 프롬프트 관리 도구
        "openai", "chromadb", "sentence-transformers",           # API 기반 모델 및 벡터 데이터베이스 연동
    ],
    "실험 관리": [
        "mlflow", "wandb", "optuna",                             # 모델 이력 추적 및 하이퍼파라미터 최적화 도구
    ],
    "노트북 / 유틸": [
        "jupyter", "jupyterlab", "notebook", "ipykernel",        # 대화형 분석 환경 관련 핵심 패키지
        "streamlit", "python-dotenv",                            # 앱 프로토타이핑 및 환경변수(.env) 로드 도구
    ],
}


def _get_version(pkg: str) -> str | None:
    """패키지 버전 반환. 미설치 시 None."""
    try:
        return pkg_version(pkg)           # importlib을 통해 해당 패키지의 설치 버전 문자열 획득
    except PackageNotFoundError:          # 라이브러리가 가상환경 내에 존재하지 않을 때 예외 발생
        return None                       # 에러로 중단시키지 않고 '미설치' 상태임을 알리기 위해 None 반환


def _system_info() -> dict[str, str]:
    """시스템 정보 수집."""
    info = {
        "Platform": platform.platform(),               # 운영체제의 상세 이름, 버전, 빌드 정보 획득
        "Python": sys.version.split()[0],              # 현재 실행 중인 파이썬 인터프리터의 버전 번호만 추출
        "Python path": sys.executable,                 # 가상환경(venv/conda) 문제를 진단하기 위한 파이썬 실행 경로
        "Working dir": os.getcwd(),                    # 스크립트가 실행되고 있는 현재 작업 디렉토리 경로
        "Encoding": sys.stdout.encoding or "unknown",  # 터미널 출력 인코딩(보통 UTF-8) 확인, 없을 시 unknown 표시
    }

    # CUDA 및 GPU 정보 수집 (PyTorch 설치 여부에 따라 유동적 대응)
    try:
        import torch                                              # PyTorch 라이브러리 호출 시도
        if torch.cuda.is_available():                             # NVIDIA GPU 및 CUDA 드라이버 사용 가능 여부 확인
            props = torch.cuda.get_device_properties(0)           # 0번(메인) GPU 장치의 하드웨어 속성 객체 획득
            info["PyTorch"] = torch.__version__                   # 설치된 PyTorch 버전 저장
            info["CUDA (torch)"] = torch.version.cuda or "N/A"    # PyTorch와 연동된 CUDA 툴킷 버전 확인
            info["GPU"] = props.name                              # GPU 모델명(예: NVIDIA GeForce RTX 4090) 추출
        else:                                                     # PyTorch는 있으나 GPU 가속을 쓸 수 없는 환경인 경우
            info["PyTorch"] = torch.__version__                   # PyTorch 버전만 기록
            info["CUDA"] = "not available"                        # CUDA 사용 불가능 상태 명시
    except ImportError:                                           # 환경에 torch 패키지 자체가 설치되어 있지 않은 경우
        info["PyTorch"] = "not installed"                         # 라이브러리 미설치 상태 기록하여 충돌 방지

    return info                                                   # 수집된 시스템 및 하드웨어 정보를 딕셔너리로 반환


def show_versions(
    only_installed: bool = True,
    extra_packages: list[str] | None = None,
) -> dict[str, str | None]:
    """
    핵심 DS/ML 패키지 버전을 그룹별로 출력.

    Parameters
    ----------
    only_installed : bool
        True면 설치된 패키지만 출력, False면 미설치도 표시 (기본 True)
    extra_packages : list[str] | None
        기본 목록 외에 추가로 확인할 패키지명 리스트

    Returns
    -------
    dict : {패키지명: 버전} (미설치는 None)
    """
    print("=" * 55)
    print("  시스템 정보")
    print("=" * 55)

    sys_info = _system_info()
    for k, v in sys_info.items():
        print(f"  {k:<16}: {v}")

    print()
    print("=" * 55)
    print("  패키지 버전")
    print("=" * 55)

    all_versions = {}

    for group_name, packages in CORE_PACKAGES.items():
        group_lines = []
        for pkg in packages:
            ver = _get_version(pkg)
            all_versions[pkg] = ver
            if ver:
                group_lines.append(f"  {pkg:<28} {ver}")
            elif not only_installed:
                group_lines.append(f"  {pkg:<28} (미설치)")

        if group_lines:
            print(f"\n  [{group_name}]")
            for line in group_lines:
                print(line)

    # 추가 패키지
    if extra_packages:
        extra_lines = []
        for pkg in extra_packages:
            ver = _get_version(pkg)
            all_versions[pkg] = ver
            if ver:
                extra_lines.append(f"  {pkg:<28} {ver}")
            elif not only_installed:
                extra_lines.append(f"  {pkg:<28} (미설치)")

        if extra_lines:
            print(f"\n  [추가 지정]")
            for line in extra_lines:
                print(line)

    installed = sum(1 for v in all_versions.values() if v)
    total = len(all_versions)
    print(f"\n  설치됨: {installed}/{total}개")
    print("=" * 55)

    return all_versions


def export_requirements(
    path: str = "requirements_helper.txt",
    only_installed: bool = True,
    extra_packages: list[str] | None = None,
    pin_versions: bool = True,
) -> str:
    """
    핵심 패키지 목록을 requirements.txt 형식으로 저장.

    pip freeze 전체 덤프가 아닌, DS/ML 실무 패키지만 추출합니다.
    전체 환경 복제가 필요하면 `pip freeze > requirements_full.txt`를 사용하세요.

    Parameters
    ----------
    path : str
        저장 경로 (기본 'requirements_helper.txt')
    only_installed : bool
        True면 설치된 패키지만 기록 (기본 True)
    extra_packages : list[str] | None
        추가 패키지
    pin_versions : bool
        True면 버전 고정 (pandas==2.2.0), False면 이름만 (pandas)

    Returns
    -------
    str : 저장된 파일 경로
    """
    lines = []
    lines.append(f"# helper/envcheck.py 자동 생성")
    lines.append(f"# 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"# Python: {sys.version.split()[0]}")
    lines.append(f"# Platform: {platform.platform()}")

    # CUDA 정보 추가
    try:
        import torch
        lines.append(f"# PyTorch CUDA: {torch.version.cuda or 'N/A'}")
    except ImportError:
        pass

    lines.append("")

    all_pkgs = []
    for packages in CORE_PACKAGES.values():
        all_pkgs.extend(packages)
    if extra_packages:
        all_pkgs.extend(extra_packages)

    for pkg in all_pkgs:
        ver = _get_version(pkg)
        if ver:
            lines.append(f"{pkg}=={ver}" if pin_versions else pkg)
        elif not only_installed:
            lines.append(f"# {pkg}  (미설치)")

    content = "\n".join(lines) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    installed = sum(1 for p in all_pkgs if _get_version(p))
    print(f"[envcheck] {path} 저장 완료 ({installed}개 패키지)")

    return os.path.abspath(path)


def compare_requirements(
    path: str,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    requirements.txt와 현재 환경을 비교하여 불일치 항목을 출력.

    Parameters
    ----------
    path : str
        비교할 requirements.txt 파일 경로
    verbose : bool
        비교 결과 출력 (기본 True)

    Returns
    -------
    dict : {패키지명: {"required": 요구버전, "installed": 설치버전, "status": 상태}}
          status: "match" | "mismatch" | "missing" | "extra"
    """
    if not os.path.exists(path):
        print(f"[envcheck] 파일 없음: {path}")
        return {}

    # requirements.txt 파싱
    required = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 형식: package==1.2.3 또는 package>=1.2.3 또는 package
            for sep in ["==", ">=", "<=", "~=", "!="]:
                if sep in line:
                    pkg, ver = line.split(sep, 1)
                    required[pkg.strip()] = ver.strip()
                    break
            else:
                required[line.strip()] = None  # 버전 미지정

    # 비교
    results = {}
    match_count = 0
    mismatch_items = []
    missing_items = []

    for pkg, req_ver in required.items():
        cur_ver = _get_version(pkg)

        if cur_ver is None:
            results[pkg] = {"required": req_ver, "installed": None, "status": "missing"}
            missing_items.append((pkg, req_ver))
        elif req_ver is None or cur_ver == req_ver:
            results[pkg] = {"required": req_ver, "installed": cur_ver, "status": "match"}
            match_count += 1
        else:
            results[pkg] = {"required": req_ver, "installed": cur_ver, "status": "mismatch"}
            mismatch_items.append((pkg, req_ver, cur_ver))

    if verbose:
        total = len(required)
        print("=" * 55)
        print(f"  환경 비교: {path}")
        print(f"  일치 {match_count}/{total}개", end="")
        if mismatch_items:
            print(f" | 버전 불일치 {len(mismatch_items)}개", end="")
        if missing_items:
            print(f" | 미설치 {len(missing_items)}개", end="")
        print()
        print("=" * 55)

        if mismatch_items:
            print("\n  [버전 불일치]")
            for pkg, req, cur in mismatch_items:
                print(f"  {pkg:<28} 요구={req}  현재={cur}")

        if missing_items:
            print("\n  [미설치]")
            for pkg, req in missing_items:
                ver_str = f" (요구={req})" if req else ""
                print(f"  {pkg:<28}{ver_str}")

        if not mismatch_items and not missing_items:
            print("\n  모든 패키지 일치")

        print()

    return results
