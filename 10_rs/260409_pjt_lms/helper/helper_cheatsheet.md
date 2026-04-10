# helper 패키지 치트시트

## 노트북 첫 셀 템플릿

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

핵심 연결:

- `comenv.setup()`은 `eda`의 표 출력, 그래프 backend, 한글 폰트 처리에 영향
- `dataload.load()` 결과는 그대로 `eda` 입력으로 연결
- `apikeys.load_all(set_env=True)`는 외부 라이브러리 인증에 바로 사용 가능
- `pyenv.export_requirements()`는 현재 환경 스냅샷 저장용

---

## apikeys

탐색 순서: `.env` -> `os.environ` -> `Colab secret`

### `get()`

```python
key = apikeys.get(
    "OPENAI_API_KEY",
    env_path="auto",
    set_env=False,
    required=False,
    verbose=False,
)
```

- 반환: `str | None`
- `required=True`면 못 찾을 때 `KeyError`

### `load_all()`

```python
keys = apikeys.load_all(
    keys=None,
    env_path="auto",
    set_env=True,
    verbose=True,
)
```

- 반환: `{키이름: 값}`
- `keys=None`이면 `DEFAULT_KEYS` 사용

### `show()` / `login_hf()`

```python
apikeys.show(keys=None, env_path="auto")

# 실제 HF_TOKEN이 준비된 경우
# apikeys.login_hf(token=None, env_path="auto")
```

---

## comenv

### `setup()`

```python
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
```

- 반환 주요 키: `device`, `seed`, `use_cuda`, `cpu_count`, `cpu_threads`, `korean_font`
- `matplotlib_backend="auto"`면 Jupyter에서 inline, 그 외엔 `Agg`

### `clear_gpu()`

```python
comenv.clear_gpu(verbose=True)
```

- GPU OOM 후 cache 정리용

---

## dataload

지원 형식: `.csv` `.tsv` `.xlsx` `.xls` `.json` `.jsonl` `.parquet` `.feather` `.pkl` `.pickle`

### `where()`

```python
env_name = dataload.where()
```

- 반환: `'colab'` 또는 `'local'`

### `mount_drive()`

```python
dataload.mount_drive(
    mount_point="/content/drive",
    force=False,
)
```

- 로컬에서는 no-op에 가깝고 안내 메시지만 출력

### `load()`

```python
df = dataload.load(
    "data/train.csv",
    encoding="utf-8",
    drive=False,
    drive_root="/content/drive/MyDrive",
    verbose=True,
)
```

- 반환: `pd.DataFrame`
- `.csv`/`.tsv`는 UTF-8 실패 시 `cp949` 자동 재시도
- 추가 인자는 `pd.read_*()`로 그대로 전달

예시:

```python
df_excel = dataload.load("data/report.xlsx", sheet_name="Sheet1")
df_tsv = dataload.load("data/sample.tsv", encoding="cp949")
```

### `load_many()`

```python
dfs = dataload.load_many(
    ["data/train.csv", "data/test.csv"],
    encoding="utf-8",
    drive=False,
    drive_root="/content/drive/MyDrive",
    verbose=False,
)
```

- 반환: `{basename: DataFrame}`
- 같은 basename이 여러 개면 덮어써질 수 있음

---

## eda

공통 준비:

```python
df = eda.clean_columns(df)
df = eda.parse_dates(df, cols=["order_date"], add_parts=True)
```

### 데이터 파악

```python
eda.quick_look(df, name="train")
num_cols, cat_cols = eda.split_cols(df)
dups = eda.check_duplicates(df, subset=None)
```

- `quick_look(df, name='df')`
- `split_cols(df)` -> `(num_cols, cat_cols)`
- `check_duplicates(df, subset=None)` -> 중복 행 DataFrame

### 전처리

```python
df = eda.clean_columns(df)
missing_df = eda.missing_report(df, plot=True)
outlier_df = eda.detect_outliers_iqr(df, cols=None, factor=1.5)
df = eda.parse_dates(df, cols=None, add_parts=True)
```

- `clean_columns()` 규칙:
  `strip -> lower -> 특수문자 '_' 치환 -> 연속 '_' 정리 -> 양끝 '_' 제거`
- `parse_dates()` 파생 컬럼:
  `_year`, `_month`, `_day`, `_dow`, `_hour`

### 시각화

```python
eda.plot_histograms(df, cols=None, bins=30, figsize_per_ax=(4, 3))
eda.plot_countplots(df, cols=None, top_n=10, figsize_per_ax=(5, 3))
eda.plot_corr_heatmap(df, cols=None, figsize=(8, 7), annot_size=9)
eda.plot_target_comparison(df, target="category", cols=None, kind="box", figsize_per_ax=(5, 3))
```

주의:

- `split_cols()`와 `plot_countplots()`는 dtype 기준이라 숫자 인코딩 범주는 자동으로 못 잡을 수 있음
- `plot_target_comparison()`의 권장 `kind`는 `"box"` 또는 `"violin"`

---

## pyenv

참고: 파일명은 `pyenv.py`지만 로그 prefix는 일부 함수에서 `[envcheck]`

### `show_versions()`

```python
versions = pyenv.show_versions(
    only_installed=True,
    extra_packages=["polars"],
)
```

- 반환: `{패키지명: 버전 | None}`

### `export_requirements()`

```python
req_path = pyenv.export_requirements(
    path="requirements_helper.txt",
    only_installed=True,
    extra_packages=None,
    pin_versions=True,
)
```

- 반환: 저장된 파일의 절대경로

### `compare_requirements()`

```python
result = pyenv.compare_requirements(
    "requirements_helper.txt",
    verbose=True,
)
```

- 반환 상태: `match`, `mismatch`, `missing`
- 현재 구현은 extra 패키지를 별도 status로 추가하지 않음

---

## 5셀 워크플로우

### 셀 1. 환경 설정

```python
from helper import comenv, pyenv

cfg = comenv.setup(seed=42, verbose=True)
pyenv.show_versions(only_installed=True)
```

### 셀 2. 데이터 생성 또는 로드

```python
from pathlib import Path
import pandas as pd
from helper import dataload

Path("data").mkdir(exist_ok=True)

pd.DataFrame(
    {
        "Order Date": ["2026-04-01", "2026-04-02", "2026-04-02", None],
        "Region": ["Seoul", "Busan", "Busan", "Incheon"],
        "Sales": [12000, 18000, 18000, 999999],
        "Discount": [0.10, None, 0.20, 0.15],
        "Category": ["A", "B", "B", "A"],
    }
).to_csv("data/demo_sales.csv", index=False)

df = dataload.load("data/demo_sales.csv")
```

### 셀 3. 구조 파악

```python
eda.quick_look(df, name="demo_sales")
num_cols, cat_cols = eda.split_cols(df)
dups = eda.check_duplicates(df, subset=["Order Date", "Region", "Sales"])
```

### 셀 4. 전처리 및 품질 점검

```python
df = eda.clean_columns(df)
df = eda.parse_dates(df, cols=["order_date"], add_parts=True)
missing_df = eda.missing_report(df, plot=True)
outlier_df = eda.detect_outliers_iqr(df, cols=["sales", "discount"], factor=1.5)
```

### 셀 5. 시각화 및 환경 저장

```python
eda.plot_histograms(df, cols=["sales", "discount"])
eda.plot_countplots(df, cols=["region", "category"])
eda.plot_corr_heatmap(df, cols=["sales", "discount", "order_date_month", "order_date_day"])
eda.plot_target_comparison(df, target="category", cols=["sales", "discount"], kind="box")

pyenv.export_requirements(path="requirements_helper_demo.txt")
```

---

## 빠른 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `No module named 'helper'` | 프로젝트 루트가 `sys.path`에 없음 | 첫 셀에서 `PROJECT_ROOT`를 `sys.path`에 추가 |
| API 키를 못 찾음 | `.env` 경로 또는 키 이름 불일치 | `apikeys.show()`로 먼저 확인 |
| CSV 한글 깨짐 | 인코딩 불일치 | `encoding="cp949"` 지정 |
| 그래프가 안 보임 | backend 설정 문제 | `comenv.setup(matplotlib_backend="auto")`를 먼저 실행 |
| 한글 폰트 깨짐 | 시스템 한글 폰트 없음 | Linux/Docker면 `fonts-nanum` 설치 후 재시작 |
| `parse_dates()` 후 컬럼명이 다름 | 실제 파생명은 `_year`, `_month`, `_day`, `_dow`, `_hour` | `df.filter(like="order_date")`로 확인 |
| `load_many()` 결과가 덮어씀 | basename이 같은 파일을 함께 로드 | 경로 대신 파일명을 구분해서 관리 |
| `compare_requirements()`가 extra 패키지를 안 보여줌 | 현재 구현은 requirements 파일 기준 비교만 수행 | `export_requirements()`로 기준 파일을 다시 생성 |
