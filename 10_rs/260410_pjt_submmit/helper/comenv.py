# helper/comenv.py — 환경설정 & 최적화 (import 제외)
# ============================================================================
# 사용법:
#   from helper import comenv
#   cfg = comenv.setup()                           # 기본값
#   cfg = comenv.setup(seed=123, cpu_ratio=0.75)   # 옵션 변경
#   cfg = comenv.setup(debug_cuda=True)             # CUDA 디버깅 모드
#
#   comenv.clear_gpu()                             # GPU 메모리 긴급 해제
#
# 반환값 (dict):
#   cfg["device"]       → "cuda" 또는 "cpu"
#   cfg["seed"]         → 사용된 시드값
#   cfg["use_cuda"]     → bool
#   cfg["cpu_threads"]  → 설정된 스레드 수
# ============================================================================

import os
import sys
import site
import random
import platform

def setup(
seed: int = 42,                          # 랜덤 시드: 실험 재현성을 위한 핵심 번호
    cpu_ratio: float = 0.5,              # CPU 사용률: 전체 코어 중 할당할 비율 (0.5 = 50%)
    tokenizers_parallel: bool = True,    # 토크나이저 병렬화: HF 토크나이저 속도 향상 여부
    float32_precision: str = "high",     # FP32 정밀도: "highest"(느림/정확), "high", "medium"(빠름)
    tf32: bool = True,                   # TF32 가속: Ampere 아키텍처 이상 GPU에서 속도 대폭 향상
    cudnn_benchmark: bool = True,        # cuDNN 벤치마크: 입력 크기가 고정일 때 최적 알고리즘 선택
    suppress_warnings: bool = True,      # 경고 숨김: 콘솔을 깔끔하게 유지
    verbose: bool = True,                # 정보 출력: 설정된 환경 정보를 화면에 표시
    # ── 추가 옵션 ──
force_utf8: bool = True,                # UTF-8 강제: 윈도우 등에서 한글 인코딩 에러 방지
    pandas_display: bool = True,        # pd 옵션: 표 출력 형식 자동 최적화
    pd_max_columns: int = 50,           # pd 최대 컬럼 수
    pd_max_rows: int = 100,             # pd 최대 행 수
    pd_max_colwidth: int = 80,          # pd 컬럼 너비
    matplotlib_backend: str = "auto",   # 시각화 백엔드: Jupyter 여부에 따라 자동 설정
    debug_cuda: bool = False,           # CUDA 디버깅: 에러 추적용 (활성 시 속도 매우 저하)
) -> dict:
    """
    Parameters
    ----------
    seed : int
        랜덤 시드 (기본 42)
    cpu_ratio : float
        전체 CPU 코어 중 사용 비율 (기본 0.5 = 절반)
    tokenizers_parallel : bool
        HuggingFace 토크나이저 병렬 처리 (기본 True)
    float32_precision : str
        FP32 행렬곱 정밀도 — "highest" | "high" | "medium" (기본 "high")
    tf32 : bool
        TF32 가속 허용 여부 (기본 True)
    cudnn_benchmark : bool
        cuDNN 자동 알고리즘 탐색 (기본 True)
        ※ 입력 크기가 매번 달라지면 False로 설정
    suppress_warnings : bool
        경고 메시지 숨김 (기본 True)
    verbose : bool
        환경 정보 출력 여부 (기본 True)
    force_utf8 : bool
        UTF-8 인코딩 강제 (기본 True, Windows 한국어 환경 필수)
    pandas_display : bool
        pandas 출력 옵션 설정 (기본 True)
    pd_max_columns : int
        pandas 최대 표시 컬럼 수 (기본 50)
    pd_max_rows : int
        pandas 최대 표시 행 수 (기본 100)
    pd_max_colwidth : int
        pandas 컬럼당 최대 문자 폭 (기본 80)
    matplotlib_backend : str
        matplotlib 백엔드 — "auto" | "inline" | "agg" | "tkagg" 등
        "auto": Jupyter면 inline, 아니면 agg (기본 "auto")
    debug_cuda : bool
        CUDA 디버깅 모드 (기본 False)
        ※ True 시 정확한 에러 위치 추적 가능하지만 속도 대폭 저하

    Returns
    -------
    dict : device, seed, use_cuda, cpu_threads 등 설정 정보
    """
    import warnings
    import numpy as np
    import torch

    issues = []  # 호환성 문제 수집

    # ── 경고 제어 ──
    if suppress_warnings:
        warnings.filterwarnings("ignore")

    # ── [1] 완전한 재현성 (Python random + PYTHONHASHSEED 포함) ──
    random.seed(seed)                         # 파이썬 내장 random 모듈 시드 고정
    np.random.seed(seed)                      # Numpy 시드 고정
    torch.manual_seed(seed)                   # PyTorch CPU 시드 고정
    os.environ["PYTHONHASHSEED"] = str(seed)  # 딕셔너리 키 순서 등 파이썬 해시 함수 무작위성 고정
    
    if torch.cuda.is_available():          # GPU(CUDA)를 사용 환경이면
        torch.cuda.manual_seed_all(seed)   # 멀티 GPU를 포함한 모든 GPU 연산 시드 고정
        
        # 재현성 극대화 (속도 약간 저하)
        # cuDNN이 내부적으로 사용하는 연산 알고리즘 중 '결정론적(Deterministic)'인 것만 사용하도록 강제
        # 같은 입력에 대해 항상 같은 연산 결과를 보장하지만, 최적화 속도는 조금 느려질 수 있음
        torch.backends.cudnn.deterministic = True

    # ── [2] UTF-8 인코딩 강제 ── (Windows의 CP949 인코딩으로 인한 한글 깨짐 방지)
    if force_utf8:                                            
        os.environ["PYTHONIOENCODING"] = "utf-8"   # 파이썬 표준 입출력(stdin/stdout/stderr) 인코딩 UTF-8로 지정
        os.environ["PYTHONUTF8"] = "1"             # 파이썬 3.7+에서 지원하는 UTF-8 모드를 강제 활성화
        
        # Windows 환경 특화 설정
        if sys.platform == "win32":                           # 윈도우인 경우 실행
            os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"  # 윈도우의 레거시 표준 입출력 방식 UTF-8로 교체

    # ── [3] pandas 표시 옵션 ── (데이터프레임 출력 가독성 최적화)
    if pandas_display:                                        
        try:
            import pandas as pd                                    # pandas 라이브러리 로드 시도
            pd.set_option("display.max_columns", pd_max_columns)   # 한 번에 출력하는 최대 컬럼 수
            pd.set_option("display.max_rows", pd_max_rows)         # 한 번에 출력하는 최대 행(row) 수
            pd.set_option("display.max_colwidth", pd_max_colwidth) # 컬럼 내용 길 때 보여줄 최대 글자 수
            pd.set_option("display.width", None)                   # 터미널/콘솔 너비를 자동으로 감지하여 줄바꿈 방지
            pd.set_option("display.float_format", "{:.4f}".format) # 수치형 데이터(float)를 소수점 4자리까지 고정해서 표시
        except ImportError:                                        # pandas가 설치되어 있지 않아도 에러 없이 통과
            pass

    # ── [4] matplotlib 백엔드 + 한글 폰트 ── (시각화 엔진 및 폰트 설정)
    korean_font = None                                        # 탐지된 한글 폰트명을 저장할 변수 초기화
    if matplotlib_backend:                                    # 백엔드 설정 옵션이 활성화된 경우
        try:
            import matplotlib                                
            if matplotlib_backend == "auto":                  # 백엔드를 자동으로 감지하도록 설정했다면
                # Jupyter 환경 여부를 체크 (내부 함수 _is_jupyter 호출)
                in_jupyter = _is_jupyter() 
                # 주피터면 결과창에 바로 보여주는 inline, 아니면 파일 저장 등에 유리한 Agg 사용
                backend = "module://matplotlib_inline.backend_inline" if in_jupyter else "Agg"
            else:
                backend = matplotlib_backend                  # 사용자가 직접 지정한 백엔드 사용
            matplotlib.use(backend, force=False)              # 실제 시각화 엔진을 시스템에 적용

            import matplotlib.pyplot as plt                   

            korean_font = _find_korean_font()                 # 한글 폰트 자동 탐지 (내부 함수 _find_korean_font 호출)
            if korean_font:                                   # 한글 폰트(나눔고딕, 맑은고딕 등)를 찾았다면
                plt.rcParams["font.family"] = korean_font     # 찾은 폰트를 전체 그래프의 기본 폰트로 적용
            else:                                             # 못 찾았다면 기본 영문 폰트 유지
                plt.rcParams["font.family"] = "DejaVu Sans"
                issues.append(                                # 경고 리스트에 폰트 미설치 안내 추가
                    "한글 폰트 미설치 — 시각화에서 한글이 깨질 수 있습니다. "
                    "(Linux/Docker: apt install -y fonts-nanum 후 커널 재시작)"
                )

            # 그래프 축에서 마이너스(-) 기호가 'ㅁ'으로 깨지는 현상 방지 (유니코드 마이너스 비활성화)
            plt.rcParams["axes.unicode_minus"] = False 

            # 폰트 캐시 강제 리빌드: 폰트를 새로 설치했을 때 바로 인식하도록 폰트 매니저 초기화
            import matplotlib.font_manager as fm
            fm._load_fontmanager(try_read_cache=False) 
        except ImportError:                                   # 라이브러리가 없는 경우 예외 처리
            pass

    # ── CPU 스레드 설정 ── (병렬 연산 자원 할당 제어)
    cpu_count = os.cpu_count() or 4                           # 시스템 전체 CPU 코어 수 확인 (탐지 실패 시 기본 4개)
    cpu_threads = max(1, int(cpu_count * cpu_ratio))          # 설정된 비율(cpu_ratio)만큼만 사용할 스레드 수 계산

    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)          # OpenMP 라이브러리에서 사용할 스레드 수 제한
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)          # Intel MKL 수치 연산 라이브러리 스레드 수 제한
    
    os.environ["TOKENIZERS_PARALLELISM"] = str(tokenizers_parallel).lower()  # HuggingFace 토크나이저의 병렬 처리 여부 설정 (True/False 문자열로 환경변수 등록)

    torch.set_num_threads(cpu_threads)                        # 파이토치 내부 연산에 할당할 CPU 스레드 고정
    try:
        # 연산 간(Inter-op) 병렬 처리를 위한 스레드 수 설정 (너무 많으면 오버헤드 발생하므로 제한적 설정)
        torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2))) 
    except RuntimeError:                                      # 이미 설정되어 변경 불가능한 경우 예외 처리
        pass

    # ── GPU 설정 ── (CUDA 가속 최적화)
    use_cuda = torch.cuda.is_available()                      # 시스템 내 GPU(NVIDIA) 사용 가능 여부 확인
    device = "cuda" if use_cuda else "cpu"                    # 기본 연산 장치 결정

    if use_cuda:                                              # GPU가 있을 때만 실행
        # FP32(32비트 부동소수점) 행렬곱 정밀도 설정 (Tensor Core 활용도 결정)
        torch.set_float32_matmul_precision(float32_precision) 
        torch.backends.cuda.matmul.allow_tf32 = tf32          # Ampere 아키텍처(RTX 30+) 이상에서 TF32 연산 허용
        torch.backends.cudnn.allow_tf32 = tf32                # 딥러닝 라이브러리(cuDNN)에서도 TF32 허용
        # cuDNN이 여러 알고리즘 중 현재 하드웨어에 최적화된 것을 찾도록 설정 (입력 크기 고정 시 유리)
        torch.backends.cudnn.benchmark = cudnn_benchmark      

    # ── [6] CUDA 디버깅 모드 ── (문제 발생 시 추적용)
    if debug_cuda:
        # GPU 연산을 비동기가 아닌 동기(Synchronous) 방식으로 강제 실행 (에러 발생 위치를 정확히 알 수 있음)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"              
        torch.autograd.set_detect_anomaly(True)        # 역전파(Backprop) 중 숫자가 튀는(NaN 등) 현상 감지 활성
        if not suppress_warnings:
            warnings.warn(
                "[comenv] CUDA 디버깅 모드 활성화 — 속도가 대폭 저하됩니다.",
                UserWarning,
            )

    # ── [7] torch-CUDA 버전 호환성 체크 ──
    cuda_compat = _check_cuda_compat(torch) if use_cuda else None   # 설치된 PyTorch와 현재 시스템의 GPU 드라이버 버전이 맞는지 확인 (내부 함수 호출 가정)
    if cuda_compat and not cuda_compat["compatible"]:               # 버전이 서로 맞지 않아 문제가 생길 우려가 있다면
        issues.append(cuda_compat["message"])                       # 경고 메시지 리스트에 추가

    # ── 정보 출력 ──
    if verbose:
        print("=" * 70)
        print(f"Platform             : {platform.platform()}")
        print(f"Working directory    : {os.getcwd()}")
        print(f"Python version       : {sys.version.split()[0]}")
        print(f"Python executable    : {sys.executable}")
        print(f"Python prefix        : {sys.prefix}")
        print(f"site-packages        : {site.getsitepackages()}")
        print(f"Encoding (stdout)    : {sys.stdout.encoding}")
        print("-" * 70)
        print(f"PyTorch              : {torch.__version__}")
        print(f"CUDA available       : {use_cuda}")
        print(f"Device               : {device}")
        print(f"Random Seed          : {seed}")
        print(f"CPU count            : {cpu_count}")
        print(f"CPU threads (used)   : {cpu_threads}  (ratio={cpu_ratio})")
        print(f"Korean font          : {korean_font or '없음 (한글 깨짐 주의)'}")

        if use_cuda:
            props = torch.cuda.get_device_properties(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            print("-" * 70)
            print(f"GPU name             : {props.name}")
            print(f"GPU count            : {torch.cuda.device_count()}")
            print(f"GPU memory (total)   : {total_mem / (1024**3):.2f} GB")
            print(f"GPU memory (free)    : {free_mem / (1024**3):.2f} GB")
            print(f"TF32                 : {tf32}")
            print(f"cuDNN benchmark      : {cudnn_benchmark}")
            print(f"cuDNN deterministic  : True")
            print(f"FP32 precision       : {float32_precision}")
            if cuda_compat:
                print(f"CUDA (torch built)   : {cuda_compat['torch_cuda']}")
                print(f"CUDA (system driver) : {cuda_compat['system_cuda']}")

        if debug_cuda:
            print("-" * 70)
            print("⚠  CUDA DEBUG MODE ON — 속도 저하 주의")

        # 호환성 경고 출력
        if issues:
            print("-" * 70)
            for issue in issues:
                print(f"⚠  {issue}")

        print("=" * 70)

    return {
        "device": device,
        "seed": seed,
        "use_cuda": use_cuda,
        "cpu_count": cpu_count,
        "cpu_threads": cpu_threads,
        "korean_font": korean_font,
    }


# ============================================================================
# [5] GPU 메모리 긴급 해제 유틸
# ============================================================================
def clear_gpu(verbose: bool = True):
    """
    OOM 발생 후 노트북 재시작 없이 GPU 메모리를 해제합니다.

    사용법:
        from helper import comenv
        comenv.clear_gpu()
    """
    import gc
    import torch

    if not torch.cuda.is_available():
        if verbose:
            print("[comenv] CUDA 사용 불가 — 해제할 GPU 메모리 없음")
        return

    before = torch.cuda.memory_allocated() / (1024**3)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # 남아있는 tensor 참조 정리
    torch.cuda.synchronize()

    after = torch.cuda.memory_allocated() / (1024**3)

    if verbose:
        print(f"[comenv] GPU 메모리 해제: {before:.2f} GB → {after:.2f} GB "
              f"({before - after:.2f} GB 회수)")


# ============================================================================
# 내부 유틸 함수
# ============================================================================
def _is_jupyter() -> bool:
    """현재 실행 환경이 Jupyter 노트북인지 판별"""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = type(shell).__name__
        return shell_name in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except (ImportError, NameError):
        return False


def _find_korean_font() -> str | None:
    """
    환경별 한글 폰트를 자동 탐지하여 폰트 패밀리명을 반환.
    탐지 순서: Malgun Gothic (Win) → NanumGothic (Linux/Docker) → AppleGothic (Mac)
    설치된 한글 폰트가 없으면 None 반환.
    """
    try:
        import matplotlib.font_manager as fm
    except ImportError:
        return None

    # 우선순위 순서대로 탐색
    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic"]

    installed = {f.name for f in fm.fontManager.ttflist}

    for font_name in candidates:
        if font_name in installed:
            return font_name

    # ttflist에 안 잡히는 경우 — 직접 경로 탐색 (Docker/Linux에서 apt 설치 직후)
    fallback_paths = [
        ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic"),
        ("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf", "NanumBarunGothic"),
    ]
    for path, name in fallback_paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            return name

    return None


def _check_cuda_compat(torch_module) -> dict:
    """
    [7] torch가 빌드된 CUDA 버전과 시스템 CUDA 드라이버 버전을 비교합니다.
    메이저 버전이 다르면 경고를 반환합니다.
    """
    result = {"compatible": True, "message": "", "torch_cuda": "", "system_cuda": ""}

    try:
        # torch가 빌드된 CUDA 버전
        torch_cuda = torch_module.version.cuda or "N/A"
        result["torch_cuda"] = torch_cuda

        # 시스템 CUDA 런타임 버전 (nvidia-smi 기반)
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            driver_ver = out.stdout.strip().split("\n")[0]
            result["system_cuda"] = f"driver {driver_ver}"
        else:
            result["system_cuda"] = "nvidia-smi 실행 불가"
            return result

        # CUDA 런타임 버전 비교 (torch 빌드 vs 실제)
        runtime_ver = getattr(torch_module.cuda, "runtime_version", None)
        if callable(runtime_ver):
            runtime_ver = runtime_ver()

        if runtime_ver and torch_cuda != "N/A":
            torch_major = torch_cuda.split(".")[0]
            runtime_major = str(runtime_ver).split(".")[0] if runtime_ver else None

            if runtime_major and torch_major != runtime_major:
                result["compatible"] = False
                result["message"] = (
                    f"CUDA 버전 불일치: torch 빌드={torch_cuda}, "
                    f"시스템 런타임={runtime_ver} → 예기치 않은 에러 가능"
                )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        result["system_cuda"] = "확인 불가"

    return result