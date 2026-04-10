# helper 패키지 사용 설명서

## 1. 패키지 개요

`helper` 패키지는 데이터 분석과 ML 노트북에서 반복되는 초기 환경 설정, 데이터 로드, 빠른 EDA, API 키 관리, 실행 환경 점검 작업을 한 폴더에 모아 둔 로컬 Python 헬퍼 패키지입니다. 매번 `pandas` 출력 옵션을 다시 맞추고, 파일 경로와 인코딩을 확인하고, 결측치와 이상치를 점검하고, `.env` 또는 Colab 비밀에서 API 키를 꺼내고, 현재 Python 환경 버전을 정리하는 수고를 줄여 줍니다. 특히 `comenv.setup()`은 `eda`의 표 출력 폭, matplotlib backend, 한글 폰트 처리에 직접 영향을 주고, `dataload.load()`로 읽은 DataFrame은 `eda` 함수들로 바로 이어지는 흐름을 전제로 설계되어 있습니다.

## 2. 설치 및 초기 설정

### 디렉토리 구조

`helper`는 별도 배포 패키지라기보다 프로젝트 안에 두고 바로 import하는 로컬 패키지입니다.

```text
project/
├─ helper/
│  ├─ __init__.py
│  ├─ apikeys.py
│  ├─ comenv.py
│  ├─ dataload.py
│  ├─ eda.py
│  ├─ pyenv.py
│  ├─ helper_guide.md
│  └─ helper_cheatsheet.md
├─ data/
│  └─ train.csv
├─ notebooks/
│  └─ analysis.ipynb
└─ .env
```

### 권장 의존성 설치

최소 분석 환경:

```bash
pip install pandas numpy matplotlib seaborn
```

기능별 추가 의존성:

```bash
pip install torch openpyxl xlrd pyarrow python-dotenv huggingface_hub
```

- `torch`: `comenv.setup()`, `comenv.clear_gpu()`, `pyenv`의 CUDA 정보 출력
- `openpyxl`: `.xlsx` 로드
- `xlrd`: `.xls` 로드
- `pyarrow`: `.parquet`, `.feather` 로드
- `python-dotenv`: `.env` 파싱 보조
- `huggingface_hub`: `apikeys.login_hf()`

### 노트북 첫 셀 세팅

아래 셀은 노트북이 프로젝트 루트 또는 그 하위 디렉토리에서 실행되는 경우를 안전하게 커버합니다.

```python
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "helper").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from helper import apikeys, comenv, dataload, eda, pyenv

cfg = comenv.setup(
    seed=42,
    cpu_ratio=0.5,
    pandas_display=True,
    matplotlib_backend="auto",
    verbose=True,
)
```

- `%autoreload 2`: `helper/*.py` 수정 사항을 커널 재시작 없이 반영
- `sys.path.append(...)`: 노트북 위치가 프로젝트 루트가 아닐 때도 `helper` import 가능
- `comenv.setup()`: 이후 `eda`의 표 출력, 그래프 backend, 한글 폰트 설정에 영향

### `.env` 예시

```env
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

`apikeys.get()` 탐색 순서:

1. `.env` 파일
2. `os.environ`
3. Colab 보안 비밀

### 모듈 간 연관관계

- `comenv.setup()`의 `pandas_display`와 `matplotlib_backend` 설정은 `eda.quick_look()` 및 모든 plot 함수 출력 형식에 바로 반영됩니다.
- `dataload.load()`와 `dataload.load_many()`의 결과는 그대로 `eda` 함수 입력으로 연결됩니다.
- `apikeys.get(..., set_env=True)` 또는 `apikeys.load_all(set_env=True)`는 외부 라이브러리가 `os.environ`에서 키를 읽도록 맞춰 줍니다.
- `pyenv.export_requirements()`는 현재 분석 환경을 파일로 남겨 이후 재현과 배포에 연결합니다.

## 3. 모듈별 레퍼런스

공개 함수만 정리했습니다. `_`로 시작하는 함수는 내부 유틸이므로 문서 대상에서 제외했습니다.

### 3.1 apikeys

한 줄 설명: `.env`, 환경변수, Colab 보안 비밀에서 API 키를 찾아 로드하고 필요하면 환경변수로 등록하는 모듈입니다.

#### 함수 목록

| 함수명 | 설명 | 주요 파라미터 |
|---|---|---|
| `get` | 단일 API 키 로드 | `key_name`, `env_path='auto'`, `set_env=False`, `required=False`, `verbose=False` |
| `load_all` | 여러 키 일괄 로드 | `keys=None`, `env_path='auto'`, `set_env=False`, `verbose=True` |
| `show` | 현재 로드 가능한 키 상태 출력 | `keys=None`, `env_path='auto'` |
| `login_hf` | Hugging Face Hub 로그인 | `token=None`, `env_path='auto'` |

#### 공통 준비 코드

아래 예제는 복붙 후 바로 실행할 수 있도록 임시 `.env` 파일을 만듭니다.

```python
from pathlib import Path
from helper import apikeys

demo_env_path = Path(".env.helper_demo")
demo_env_path.write_text(
    "OPENAI_API_KEY=sk-demo-12345678\nHF_TOKEN=hf-demo-12345678\n",
    encoding="utf-8",
)
```

#### `get`

```python
from helper import apikeys

openai_key = apikeys.get(
    "OPENAI_API_KEY",
    env_path=".env.helper_demo",
    set_env=False,
    required=True,
    verbose=True,
)

print(openai_key)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `key_name` | `str` | 필수 | 없음 | 찾을 키 이름 |
| `env_path` | `str` | 선택 | `'auto'` | `.env` 파일 경로. `'auto'`면 현재 디렉토리, 상위 2단계, `~/.env`까지 탐색 |
| `set_env` | `bool` | 선택 | `False` | 찾은 값을 `os.environ[key_name]`에 등록 |
| `required` | `bool` | 선택 | `False` | `True`면 키를 못 찾을 때 `KeyError` 발생 |
| `verbose` | `bool` | 선택 | `False` | 키를 어디서 찾았는지 로그 출력 |

반환값:

- 타입: `str | None`
- 키를 찾으면 문자열 반환
- 못 찾으면 `None` 반환
- `required=True`인데 못 찾으면 `KeyError` 발생

#### `load_all`

```python
from helper import apikeys

keys = apikeys.load_all(
    keys=["OPENAI_API_KEY", "HF_TOKEN", "WANDB_API_KEY"],
    env_path=".env.helper_demo",
    set_env=True,
    verbose=True,
)

print(keys)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `keys` | `list[str] \| None` | 선택 | `None` | `None`이면 `DEFAULT_KEYS` 사용 |
| `env_path` | `str` | 선택 | `'auto'` | `.env` 탐색 경로 |
| `set_env` | `bool` | 선택 | `False` | 찾은 키를 모두 `os.environ`에 등록 |
| `verbose` | `bool` | 선택 | `True` | 성공/실패 요약 출력 |

반환값:

- 타입: `dict`
- 형식: `{키이름: 값}`
- 찾지 못한 키는 결과 dict에 포함되지 않음

#### `show`

```python
from helper import apikeys

apikeys.show(
    keys=["OPENAI_API_KEY", "HF_TOKEN", "WANDB_API_KEY"],
    env_path=".env.helper_demo",
)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `keys` | `list[str] \| None` | 선택 | `None` | 확인할 키 목록. `None`이면 `DEFAULT_KEYS` 사용 |
| `env_path` | `str` | 선택 | `'auto'` | `.env` 탐색 경로 |

반환값:

- 반환값 없음
- `.env` 위치, 현재 환경, 키 존재 여부를 마스킹해서 출력

#### `login_hf`

```python
from helper import apikeys

# 실제 HF_TOKEN이 있는 .env를 사용할 때
# apikeys.login_hf(env_path=".env")

# 또는 직접 토큰 지정
# apikeys.login_hf(token="hf_xxx")
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `token` | `str \| None` | 선택 | `None` | 직접 토큰을 넘기면 그 값을 사용 |
| `env_path` | `str` | 선택 | `'auto'` | `token=None`일 때 `HF_TOKEN` 탐색 경로 |

반환값:

- 반환값 없음
- 내부적으로 `huggingface_hub.login()` 호출
- `token=None`인데 `HF_TOKEN`을 찾지 못하면 `KeyError` 발생

#### 모듈 연관관계

- `apikeys.get(..., set_env=True)`는 `openai`, `wandb`, `langchain`, `transformers`처럼 환경변수 기반 인증 라이브러리와 바로 연결됩니다.
- Colab에서는 `.env` 없이도 같은 함수로 보안 비밀까지 커버할 수 있어 노트북 코드가 단순해집니다.

### 3.2 comenv

한 줄 설명: seed, CPU 스레드, pandas 출력, matplotlib backend, GPU 최적화를 한 번에 맞추는 환경 설정 모듈입니다.

#### 함수 목록

| 함수명 | 설명 | 주요 파라미터 |
|---|---|---|
| `setup` | 분석 환경 전체 초기화 | `seed=42`, `cpu_ratio=0.5`, `pandas_display=True`, `matplotlib_backend='auto'`, `debug_cuda=False` |
| `clear_gpu` | GPU 메모리 캐시 정리 | `verbose=True` |

#### `setup`

```python
from helper import comenv

cfg = comenv.setup(
    seed=42,
    cpu_ratio=0.5,
    tokenizers_parallel=True,
    float32_precision="high",
    tf32=True,
    cudnn_benchmark=True,
    suppress_warnings=True,
    verbose=True,
    force_utf8=True,
    pandas_display=True,
    pd_max_columns=50,
    pd_max_rows=100,
    pd_max_colwidth=80,
    matplotlib_backend="auto",
    debug_cuda=False,
)

print(cfg)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `seed` | `int` | 선택 | `42` | `random`, `numpy`, `torch` 시드 고정 |
| `cpu_ratio` | `float` | 선택 | `0.5` | 전체 CPU 중 사용할 비율 |
| `tokenizers_parallel` | `bool` | 선택 | `True` | `TOKENIZERS_PARALLELISM` 설정 |
| `float32_precision` | `str` | 선택 | `'high'` | `torch.set_float32_matmul_precision()` 인자 |
| `tf32` | `bool` | 선택 | `True` | CUDA에서 TF32 허용 여부 |
| `cudnn_benchmark` | `bool` | 선택 | `True` | cuDNN 벤치마크 사용 여부 |
| `suppress_warnings` | `bool` | 선택 | `True` | warning 출력 억제 |
| `verbose` | `bool` | 선택 | `True` | 환경 정보 출력 여부 |
| `force_utf8` | `bool` | 선택 | `True` | UTF-8 관련 환경변수 강제 |
| `pandas_display` | `bool` | 선택 | `True` | pandas 표시 옵션 자동 설정 |
| `pd_max_columns` | `int` | 선택 | `50` | pandas 최대 컬럼 표시 수 |
| `pd_max_rows` | `int` | 선택 | `100` | pandas 최대 행 표시 수 |
| `pd_max_colwidth` | `int` | 선택 | `80` | pandas 문자열 컬럼 폭 |
| `matplotlib_backend` | `str` | 선택 | `'auto'` | `'auto'`면 Jupyter에서 inline, 그 외에는 `Agg` |
| `debug_cuda` | `bool` | 선택 | `False` | `CUDA_LAUNCH_BLOCKING=1` 및 anomaly detection 활성화 |

반환값:

- 타입: `dict`
- 주요 키:

```python
{
    "device": "cuda" or "cpu",
    "seed": 42,
    "use_cuda": True or False,
    "cpu_count": 16,
    "cpu_threads": 8,
    "korean_font": "Malgun Gothic" or "NanumGothic" or "AppleGothic" or None,
}
```

#### `clear_gpu`

```python
from helper import comenv

comenv.clear_gpu(verbose=True)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `verbose` | `bool` | 선택 | `True` | 해제 전후 GPU 메모리 출력 여부 |

반환값:

- 반환값 없음
- CUDA를 사용할 수 없으면 안내 메시지만 출력하고 종료

#### 모듈 연관관계

- `comenv.setup(pandas_display=True)`는 `eda.quick_look()`과 일반 DataFrame 출력 가독성을 높입니다.
- `comenv.setup(matplotlib_backend="auto")`와 한글 폰트 자동 감지는 `eda`의 모든 시각화 함수 동작에 직접 영향을 줍니다.
- GPU 실험 중 OOM이 나면 `clear_gpu()`로 일부 상황에서 커널 재시작 없이 메모리 회수가 가능합니다.

### 3.3 dataload

한 줄 설명: 로컬 파일, Colab 업로드, Google Drive 파일을 같은 인터페이스로 읽어 `DataFrame`으로 반환하는 데이터 로더 모듈입니다.

#### 함수 목록

| 함수명 | 설명 | 주요 파라미터 |
|---|---|---|
| `where` | 실행 환경 확인 | 없음 |
| `mount_drive` | Colab에서 Google Drive 마운트 | `mount_point='/content/drive'`, `force=False` |
| `load` | 단일 파일 로드 | `path`, `encoding='utf-8'`, `drive=False`, `drive_root='/content/drive/MyDrive'`, `verbose=True` |
| `load_many` | 여러 파일 일괄 로드 | `paths`, `encoding='utf-8'`, `drive=False`, `drive_root='/content/drive/MyDrive'`, `verbose=True` |

지원 형식:

- `.csv`
- `.tsv`
- `.xlsx`
- `.xls`
- `.json`
- `.jsonl`
- `.parquet`
- `.feather`
- `.pkl`
- `.pickle`

#### 공통 준비 코드

```python
from pathlib import Path
import pandas as pd

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

pd.DataFrame(
    {
        "order_id": [1, 2, 3],
        "region": ["Seoul", "Busan", "Seoul"],
        "sales": [12000, 18000, 15000],
        "order_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
    }
).to_csv(data_dir / "demo_sales.csv", index=False)
```

#### `where`

```python
from helper import dataload

env_name = dataload.where()
print(env_name)
```

파라미터 상세:

없음

반환값:

- 타입: `str`
- Colab이면 `'colab'`, 아니면 `'local'`

#### `mount_drive`

```python
from helper import dataload

dataload.mount_drive(mount_point="/content/drive", force=False)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `mount_point` | `str` | 선택 | `'/content/drive'` | Drive를 마운트할 위치 |
| `force` | `bool` | 선택 | `False` | 이미 마운트되어 있어도 다시 마운트할지 여부 |

반환값:

- 반환값 없음
- 로컬 환경이면 `[dataload] 로컬 환경 — Drive 마운트 불필요` 출력 후 종료

#### `load`

```python
from helper import dataload

df = dataload.load(
    "data/demo_sales.csv",
    encoding="utf-8",
    drive=False,
    verbose=True,
)

print(df)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `path` | `str` | 필수 | 없음 | 파일 경로 또는 파일명 |
| `encoding` | `str` | 선택 | `'utf-8'` | CSV/TSV 기본 인코딩 |
| `drive` | `bool` | 선택 | `False` | Colab에서 Drive 기준 경로 해석 여부 |
| `drive_root` | `str` | 선택 | `'/content/drive/MyDrive'` | `drive=True`일 때 상대경로 기준 루트 |
| `verbose` | `bool` | 선택 | `True` | 파일명, shape, 용량, 환경 출력 여부 |
| `**kwargs` | 가변 | 선택 | 없음 | `pandas` reader로 그대로 전달되는 추가 옵션 |

반환값:

- 타입: `pd.DataFrame`
- 파일이 없으면 `FileNotFoundError`
- 지원하지 않는 확장자면 `ValueError`
- `.csv`, `.tsv`는 UTF-8 실패 시 `cp949` 자동 재시도

추가 예시:

```python
from helper import dataload

df_excel = dataload.load("data/report.xlsx", sheet_name="Sheet1")
df_tsv = dataload.load("data/sample.tsv", encoding="cp949")
```

#### `load_many`

```python
from pathlib import Path
import pandas as pd
from helper import dataload

pd.DataFrame({"x": [1, 2]}).to_csv("data/train_demo.csv", index=False)
pd.DataFrame({"x": [3, 4]}).to_csv("data/test_demo.csv", index=False)

dfs = dataload.load_many(
    ["data/train_demo.csv", "data/test_demo.csv"],
    verbose=False,
)

print(dfs["train_demo.csv"])
print(dfs["test_demo.csv"])
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `paths` | `list[str]` | 필수 | 없음 | 읽을 파일 경로 목록 |
| `encoding` | `str` | 선택 | `'utf-8'` | 각 파일에 공통 적용되는 기본 인코딩 |
| `drive` | `bool` | 선택 | `False` | Colab Drive 기준 경로 해석 여부 |
| `drive_root` | `str` | 선택 | `'/content/drive/MyDrive'` | `drive=True`일 때 기준 루트 |
| `verbose` | `bool` | 선택 | `True` | 각 파일 로드 로그 출력 여부 |
| `**kwargs` | 가변 | 선택 | 없음 | 각 `pandas` reader에 공통 전달 |

반환값:

- 타입: `dict`
- 형식: `{파일명: DataFrame}`
- key는 `os.path.basename(path)`이므로 같은 파일명이 여러 개면 덮어써질 수 있음

#### 모듈 연관관계

- `dataload.load()` 결과는 바로 `eda.quick_look()`, `eda.clean_columns()`, `eda.parse_dates()`로 넘기면 됩니다.
- Colab에서는 `where()`로 환경 확인 후 `mount_drive()` 또는 `load(..., drive=True)`를 연결하는 흐름이 가장 자연스럽습니다.

### 3.4 eda

한 줄 설명: 데이터 구조 파악, 전처리 보조, 기초 시각화를 빠르게 수행하는 EDA 모듈입니다.

#### 함수 목록

| 함수명 | 설명 | 주요 파라미터 |
|---|---|---|
| `quick_look` | shape, dtype, 결측, 샘플값 요약 출력 | `df`, `name='df'` |
| `split_cols` | 수치형/비수치형 컬럼 분리 | `df` |
| `check_duplicates` | 중복 행 확인 | `df`, `subset=None` |
| `clean_columns` | 컬럼명 정리 | `df` |
| `missing_report` | 결측치 리포트 및 시각화 | `df`, `plot=True` |
| `detect_outliers_iqr` | IQR 기반 이상치 탐지 | `df`, `cols=None`, `factor=1.5` |
| `parse_dates` | 날짜 컬럼을 `datetime`으로 변환하고 파생 컬럼 추가 | `df`, `cols=None`, `add_parts=True` |
| `plot_histograms` | 수치형 분포 히스토그램 | `df`, `cols=None`, `bins=30`, `figsize_per_ax=(4, 3)` |
| `plot_countplots` | 범주형 분포 countplot | `df`, `cols=None`, `top_n=10`, `figsize_per_ax=(5, 3)` |
| `plot_corr_heatmap` | 상관행렬 히트맵 | `df`, `cols=None`, `figsize=(8, 7)`, `annot_size=9` |
| `plot_target_comparison` | target 기준 수치형 분포 비교 | `df`, `target`, `cols=None`, `kind='box'`, `figsize_per_ax=(5, 3)` |

#### 공통 준비 코드

아래 셀을 먼저 실행하면 이후 모든 `eda` 예제가 그대로 동작합니다.

```python
import pandas as pd
from helper import eda

df = pd.DataFrame(
    {
        "Order Date": ["2026-04-01", "2026-04-02", "2026-04-02", None],
        "Region": ["Seoul", "Busan", "Busan", "Incheon"],
        "Sales": [12000, 18000, 18000, 999999],
        "Discount": [0.10, None, 0.20, 0.15],
        "Category": ["A", "B", "B", "A"],
    }
)

df = eda.clean_columns(df)
```

#### `quick_look`

```python
from helper import eda

eda.quick_look(df, name="sales_demo")
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 확인할 데이터프레임 |
| `name` | `str` | 선택 | `'df'` | 출력용 이름 |

반환값:

- 반환값 없음
- shape, 메모리, dtype, 결측수, 결측%, 유니크 수, 첫 행 샘플값 출력

#### `split_cols`

```python
from helper import eda

num_cols, cat_cols = eda.split_cols(df)
print(num_cols)
print(cat_cols)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 분리할 데이터프레임 |

반환값:

- 타입: `tuple[list[str], list[str]]`
- `(수치형 컬럼 리스트, 범주형 컬럼 리스트)` 반환
- dtype 기준 분류라 숫자로 인코딩된 범주형은 수치형으로 잡힐 수 있음

#### `check_duplicates`

```python
from helper import eda

dups = eda.check_duplicates(df, subset=["order_date", "region", "sales"])
print(dups)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 중복 확인 대상 |
| `subset` | `list[str] \| None` | 선택 | `None` | 특정 컬럼 기준으로만 중복 판정 |

반환값:

- 타입: `pd.DataFrame`
- 중복으로 판정된 행만 반환

#### `clean_columns`

```python
import pandas as pd
from helper import eda

raw_df = pd.DataFrame({" Order Date ": ["2026-04-01"], "매출 금액(원)": [12000]})
clean_df = eda.clean_columns(raw_df)
print(clean_df.columns.tolist())
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 컬럼명을 정리할 DataFrame |

반환값:

- 타입: `pd.DataFrame`
- 원본을 바꾸지 않고 복사본 반환
- 규칙: `strip -> lower -> 특수문자 '_' 치환 -> 연속 '_' 정리 -> 양끝 '_' 제거`

#### `missing_report`

```python
from helper import eda

missing_df = eda.missing_report(df, plot=True)
print(missing_df)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 결측치 분석 대상 |
| `plot` | `bool` | 선택 | `True` | 결측치 막대 그래프 출력 여부 |

반환값:

- 타입: `pd.DataFrame`
- 컬럼별 `결측수`, `결측%` 반환
- 결측치가 전혀 없으면 빈 DataFrame 반환

#### `detect_outliers_iqr`

```python
from helper import eda

outlier_df = eda.detect_outliers_iqr(
    df,
    cols=["sales", "discount"],
    factor=1.5,
)

print(outlier_df)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 이상치 탐지 대상 |
| `cols` | `list[str] \| None` | 선택 | `None` | `None`이면 전체 수치형 컬럼 대상 |
| `factor` | `float` | 선택 | `1.5` | IQR 배수. 클수록 덜 민감 |

반환값:

- 타입: `pd.DataFrame`
- 컬럼별 `이상치수`, `비율%`, `하한`, `상한` 반환
- 이상치가 없으면 빈 DataFrame 반환

#### `parse_dates`

```python
from helper import eda

df_dates = eda.parse_dates(
    df,
    cols=["order_date"],
    add_parts=True,
)

print(df_dates[["order_date", "order_date_year", "order_date_month", "order_date_dow"]])
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 날짜 변환 대상 |
| `cols` | `list[str] \| None` | 선택 | `None` | `None`이면 object 컬럼 중 날짜로 변환 가능한 컬럼 자동 탐지 |
| `add_parts` | `bool` | 선택 | `True` | `_year`, `_month`, `_day`, `_dow`, `_hour` 파생 컬럼 추가 여부 |

반환값:

- 타입: `pd.DataFrame`
- 날짜 컬럼이 `datetime64`로 바뀐 복사본 반환
- 변환 실패 값은 `NaT`

#### `plot_histograms`

```python
from helper import eda

eda.plot_histograms(
    df,
    cols=["sales", "discount"],
    bins=20,
    figsize_per_ax=(4, 3),
)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 시각화 대상 |
| `cols` | `list[str] \| None` | 선택 | `None` | `None`이면 전체 수치형 컬럼 |
| `bins` | `int` | 선택 | `30` | 히스토그램 구간 수 |
| `figsize_per_ax` | `tuple[int, int]` | 선택 | `(4, 3)` | 서브플롯 하나당 figure 크기 |

반환값:

- 반환값 없음
- 수치형 컬럼이 없으면 메시지 출력 후 종료

#### `plot_countplots`

```python
from helper import eda

eda.plot_countplots(
    df,
    cols=["region", "category"],
    top_n=10,
    figsize_per_ax=(5, 3),
)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 시각화 대상 |
| `cols` | `list[str] \| None` | 선택 | `None` | `None`이면 전체 비수치형 컬럼 |
| `top_n` | `int` | 선택 | `10` | 카테고리가 많을 때 상위 `top_n`개만 표시 |
| `figsize_per_ax` | `tuple[int, int]` | 선택 | `(5, 3)` | 서브플롯 하나당 figure 크기 |

반환값:

- 반환값 없음
- 범주형 컬럼이 없으면 메시지 출력 후 종료

#### `plot_corr_heatmap`

```python
from helper import eda

eda.plot_corr_heatmap(
    df,
    cols=["sales", "discount"],
    figsize=(6, 5),
    annot_size=10,
)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 시각화 대상 |
| `cols` | `list[str] \| None` | 선택 | `None` | `None`이면 전체 수치형 컬럼 |
| `figsize` | `tuple[int, int]` | 선택 | `(8, 7)` | 전체 히트맵 크기 |
| `annot_size` | `int` | 선택 | `9` | 셀 내부 숫자 폰트 크기 |

반환값:

- 반환값 없음
- 수치형 컬럼이 2개 미만이면 메시지 출력 후 종료

#### `plot_target_comparison`

```python
from helper import eda

eda.plot_target_comparison(
    df,
    target="category",
    cols=["sales", "discount"],
    kind="box",
    figsize_per_ax=(5, 3),
)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `df` | `pd.DataFrame` | 필수 | 없음 | 시각화 대상 |
| `target` | `str` | 필수 | 없음 | 비교 기준 컬럼. 보통 범주형 권장 |
| `cols` | `list[str] \| None` | 선택 | `None` | `None`이면 전체 수치형 중 `target` 제외 |
| `kind` | `str` | 선택 | `'box'` | 권장값은 `'box'` 또는 `'violin'` |
| `figsize_per_ax` | `tuple[int, int]` | 선택 | `(5, 3)` | 서브플롯 하나당 figure 크기 |

반환값:

- 반환값 없음
- 비교 가능한 수치형 컬럼이 없으면 메시지 출력 후 종료

#### 모듈 연관관계

- `dataload.load()` 후 가장 먼저 `eda.clean_columns()`와 `eda.parse_dates()`를 거치면 이후 분석 함수 재사용성이 높아집니다.
- `comenv.setup()`을 먼저 실행해 두면 `eda`의 표 출력, plot backend, 한글 폰트 깨짐 문제가 줄어듭니다.
- `split_cols()`와 여러 plot 함수는 dtype 기준으로 동작하므로 숫자형으로 저장된 범주 컬럼은 직접 `cols=`로 지정하는 편이 안전합니다.

### 3.5 pyenv

한 줄 설명: 현재 Python/패키지 환경을 출력하고, 핵심 패키지 목록을 requirements 파일로 저장하거나 비교하는 모듈입니다.

#### 함수 목록

| 함수명 | 설명 | 주요 파라미터 |
|---|---|---|
| `show_versions` | 핵심 패키지 버전 출력 | `only_installed=True`, `extra_packages=None` |
| `export_requirements` | 핵심 패키지 목록을 requirements 형식으로 저장 | `path='requirements_helper.txt'`, `only_installed=True`, `extra_packages=None`, `pin_versions=True` |
| `compare_requirements` | requirements 파일과 현재 환경 비교 | `path`, `verbose=True` |

참고:

- 모듈 파일명은 `pyenv.py`이지만 출력 메시지 prefix는 일부 함수에서 `[envcheck]`로 표시됩니다.

#### `show_versions`

```python
from helper import pyenv

versions = pyenv.show_versions(
    only_installed=True,
    extra_packages=["polars", "catboost"],
)

print(versions["pandas"])
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `only_installed` | `bool` | 선택 | `True` | `True`면 설치된 패키지만 출력 |
| `extra_packages` | `list[str] \| None` | 선택 | `None` | 기본 목록 외에 추가로 확인할 패키지 |

반환값:

- 타입: `dict[str, str | None]`
- 형식: `{패키지명: 버전}`
- 미설치 패키지는 값이 `None`

#### `export_requirements`

```python
from helper import pyenv

req_path = pyenv.export_requirements(
    path="requirements_helper_demo.txt",
    only_installed=True,
    extra_packages=["polars"],
    pin_versions=True,
)

print(req_path)
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `path` | `str` | 선택 | `'requirements_helper.txt'` | 저장할 파일 경로 |
| `only_installed` | `bool` | 선택 | `True` | 설치된 패키지만 기록할지 여부 |
| `extra_packages` | `list[str] \| None` | 선택 | `None` | 추가로 포함할 패키지 |
| `pin_versions` | `bool` | 선택 | `True` | `True`면 `pkg==ver`, `False`면 패키지명만 기록 |

반환값:

- 타입: `str`
- 저장된 requirements 파일의 절대경로 반환

#### `compare_requirements`

```python
from helper import pyenv

result = pyenv.compare_requirements(
    "requirements_helper_demo.txt",
    verbose=True,
)

print(result.get("pandas"))
```

파라미터 상세:

| 파라미터 | 타입 | 필수 여부 | 기본값 | 설명 |
|---|---|---|---|---|
| `path` | `str` | 필수 | 없음 | 비교할 requirements 파일 경로 |
| `verbose` | `bool` | 선택 | `True` | 비교 결과 출력 여부 |

반환값:

- 타입: `dict[str, dict]`
- 형식: `{패키지명: {"required": 요구버전, "installed": 설치버전, "status": 상태}}`
- 현재 구현 기준 `status`는 `match`, `mismatch`, `missing`을 사용
- requirements 파일에 없는 현재 환경의 extra 패키지는 별도로 결과에 추가하지 않음

#### 모듈 연관관계

- 실험 시작 전에 `show_versions()`로 환경을 확인하고, 결과가 의미 있게 나온 시점에 `export_requirements()`로 스냅샷을 남기면 재현성이 좋아집니다.
- `comenv.setup()`과 함께 쓰면 실행 장치, CUDA 여부, 패키지 버전을 동시에 점검할 수 있습니다.

## 4. 실전 워크플로우 예제

아래는 노트북에서 셀 1~5 순서로 바로 실행할 수 있는 예시입니다. 실제 데이터가 없어도 동작하도록 셀 2에서 데모 CSV를 생성한 뒤 `dataload.load()`로 다시 읽습니다.

### 셀 1. 환경 준비

```python
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "helper").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from helper import apikeys, comenv, dataload, eda, pyenv

cfg = comenv.setup(
    seed=42,
    cpu_ratio=0.5,
    pandas_display=True,
    matplotlib_backend="auto",
    verbose=True,
)

pyenv.show_versions(only_installed=True, extra_packages=["pyarrow"])
```

### 셀 2. 데모 데이터 생성 후 로드

```python
from pathlib import Path
import pandas as pd

Path("data").mkdir(exist_ok=True)

pd.DataFrame(
    {
        "Order Date": ["2026-04-01", "2026-04-02", "2026-04-02", None, "2026-04-04"],
        "Region": ["Seoul", "Busan", "Busan", "Incheon", "Seoul"],
        "Sales": [12000, 18000, 18000, 999999, 15000],
        "Discount": [0.10, None, 0.20, 0.15, 0.05],
        "Category": ["A", "B", "B", "A", "A"],
    }
).to_csv("data/demo_sales.csv", index=False)

df = dataload.load("data/demo_sales.csv", verbose=True)
df.head()
```

### 셀 3. 구조 파악

```python
eda.quick_look(df, name="demo_sales_raw")
num_cols, cat_cols = eda.split_cols(df)
dups = eda.check_duplicates(df, subset=["Order Date", "Region", "Sales"])

print("num_cols:", num_cols)
print("cat_cols:", cat_cols)
print("duplicate_rows:", len(dups))
```

### 셀 4. 전처리 및 품질 점검

```python
df = eda.clean_columns(df)
df = eda.parse_dates(df, cols=["order_date"], add_parts=True)

missing_df = eda.missing_report(df, plot=True)
outlier_df = eda.detect_outliers_iqr(
    df,
    cols=["sales", "discount"],
    factor=1.5,
)

display(df.head())
display(missing_df)
display(outlier_df)
```

### 셀 5. 시각화 및 환경 스냅샷

```python
eda.plot_histograms(df, cols=["sales", "discount"], bins=20)
eda.plot_countplots(df, cols=["region", "category"], top_n=10)
eda.plot_corr_heatmap(df, cols=["sales", "discount", "order_date_month", "order_date_day"])
eda.plot_target_comparison(df, target="category", cols=["sales", "discount"], kind="box")

req_path = pyenv.export_requirements(
    path="requirements_helper_demo.txt",
    only_installed=True,
    pin_versions=True,
)
print(req_path)
```

이 흐름의 핵심은 다음과 같습니다.

- `comenv.setup()`이 `eda`가 사용할 출력/시각화 환경을 먼저 고정합니다.
- `dataload.load()`가 파일 형식과 인코딩 차이를 흡수합니다.
- `eda.clean_columns()`와 `eda.parse_dates()`가 후속 분석에서 재사용하기 좋은 형태로 컬럼을 정리합니다.
- `pyenv.export_requirements()`는 분석을 재현 가능한 환경 파일로 마무리합니다.

## 5. FAQ / 트러블슈팅

### `ModuleNotFoundError: No module named 'helper'`

원인:

- 노트북 실행 위치가 프로젝트 루트가 아니어서 `helper/`를 import path에서 못 찾는 경우입니다.

해결:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "helper").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
```

### `KeyError: [apikeys] '...'을(를) 찾을 수 없습니다`

원인:

- `.env` 경로가 다르거나, 키 이름이 실제 파일/환경변수와 다릅니다.

해결:

```python
from helper import apikeys

apikeys.show(env_path="auto")
```

- `.env` 위치와 키 존재 여부를 먼저 확인합니다.
- 외부 라이브러리가 환경변수만 읽는 경우 `set_env=True`를 사용합니다.

### `ImportError: No module named 'huggingface_hub'`

원인:

- `apikeys.login_hf()`에 필요한 패키지가 설치되지 않았습니다.

해결:

```bash
pip install huggingface_hub
```

### `[dataload] 파일을 찾을 수 없습니다`

원인:

- 상대경로 기준이 현재 작업 디렉토리와 다르거나, Colab인데 `drive=True`를 빠뜨렸습니다.

해결:

```python
import os
from helper import dataload

print(os.getcwd())
print(dataload.where())
```

Colab Drive 파일이면:

```python
from helper import dataload

df = dataload.load(
    "datasets/train.csv",
    drive=True,
    drive_root="/content/drive/MyDrive",
)
```

### CSV 한글이 깨지거나 `UnicodeDecodeError`가 납니다

원인:

- 파일 인코딩이 UTF-8이 아니라 `cp949` 또는 `euc-kr`일 수 있습니다.

해결:

```python
from helper import dataload

df = dataload.load("data/korean.csv", encoding="cp949")
```

참고:

- `.csv`, `.tsv`는 UTF-8 실패 시 내부적으로 `cp949` 재시도를 이미 수행합니다.

### `eda.parse_dates()` 후 기대한 컬럼명이 안 보입니다

원인:

- 파생 컬럼명은 `_년`, `_월`이 아니라 실제 코드 기준 `_year`, `_month`, `_day`, `_dow`, `_hour`입니다.

해결:

```python
from helper import eda

df = eda.parse_dates(df, cols=["order_date"], add_parts=True)
print(df.filter(like="order_date").columns.tolist())
```

### `split_cols()`나 `plot_countplots()`가 범주형 컬럼을 놓칩니다

원인:

- 함수들이 dtype 기준으로 동작합니다. 숫자 인코딩된 범주형은 수치형으로 분류됩니다.

해결:

```python
from helper import eda

eda.plot_countplots(df, cols=["season_code"])
```

- 자동 감지에 맡기지 말고 `cols=`를 직접 지정합니다.

### 그래프가 안 보이거나 한글이 깨집니다

원인:

- matplotlib backend 설정 또는 한글 폰트 부재 문제입니다.

해결:

```python
from helper import comenv

comenv.setup(matplotlib_backend="auto", verbose=True)
```

추가 팁:

- Linux/Docker에서는 `fonts-nanum` 설치 후 커널 재시작이 필요할 수 있습니다.
- `comenv.setup()`을 `eda` plot 함수보다 먼저 실행하는 것이 안전합니다.

### GPU OOM 이후 계속 CUDA 메모리 부족이 납니다

원인:

- 이전 tensor 참조와 cache가 남아 있을 수 있습니다.

해결:

```python
from helper import comenv

comenv.clear_gpu(verbose=True)
```

### `pyenv.compare_requirements()` 결과가 기대보다 단순합니다

원인:

- 현재 구현은 requirements 파일에 있는 패키지만 비교하며, 현재 환경의 extra 패키지를 따로 `status='extra'`로 추가하지 않습니다.

해결:

- `compare_requirements()`는 일치/불일치/미설치 확인용으로 사용합니다.
- 현재 환경 전체 스냅샷이 필요하면 `export_requirements()`를 다시 생성해 기준 파일로 삼는 것이 더 확실합니다.
