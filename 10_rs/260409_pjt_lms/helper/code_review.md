# Helper 모듈 코드 리뷰

> 검토일: 2026-04-06  
> 대상: `/mnt/d/pytest/src_test/helper/` (8개 모듈, 약 50개 함수)

---

## 총평 (선두 배치)

**판정: "개인용 스니펫 묶음"에서 "재사용 가능한 내부 라이브러리 초안"으로 넘어가는 과도기.**

현재 상태로는 팀 프로젝트에 바로 투입하기 어렵습니다. 핵심 문제는 (1) 함수 책임 혼합, (2) 전역 상태 변경의 암묵적 부작용, (3) 모듈 간 중복 코드, (4) 시각화 함수의 `plt.show()` 하드코딩입니다. 그러나 파일 분리 의도는 합리적이고, 대부분의 함수가 리팩토링 후 모듈화 가능한 수준입니다.

---

## 1. 함수별 리팩토링 제안표

### logger.py

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `log` | 한줄 로그 출력 | B | print 부작용 + 반환값 혼재 | 중간 | utils/logging | 반환 문자열 형식 검증 | print와 return 중 하나만 책임지게 분리 필요 |
| `info/step/warn/error/success` | 레벨별 로그 래퍼 | A | 없음 (log 개선 시 자동 해결) | 낮음 | utils/logging | log와 동일 | log 개선에 종속 |
| `kv` | key-value 출력 | A | 반환값 없음 (print only) | 낮음 | utils/logging | 출력 문자열 캡처 검증 | 단순하고 재사용 가능 |
| `df` | DataFrame 요약 출력 | B | logger와 eda.quick_look 역할 중복 | 중간 | utils/logging | shape/memory 정확성 | diag.df_snapshot과 역할 겹침 확인 필요 |
| `timer` | 코드 블록 시간 측정 | A | 없음 | 낮음 | utils/logging | elapsed 정확성, 예외 시 동작 | 깔끔한 context manager |
| `line/section` | 구분선 출력 | A | 없음 | 낮음 | utils/logging | 없음 (trivial) | 단순 포매팅 |

### pathutil.py

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `find_project_root` | 프로젝트 루트 탐색 | B | `DEFAULT_MARKERS`에 "helper" 포함 — 이 패키지 자신에 종속 | 중간 | utils/paths | marker별 탐색 결과, max_depth 경계 | markers 기본값을 프로젝트-비종속으로 변경 필요 |
| `project_path` | 루트 기준 경로 생성 | A | 없음 | 낮음 | utils/paths | 경로 조합 정확성 | 단순 조합 |
| `data_path/models_path/outputs_path` | 하위 디렉토리 경로 | A | `models_path`의 `create=True` 기본값 — 암묵적 디렉토리 생성 | 중간 | utils/paths | create 시 실제 생성 여부 | create 기본값을 False로 통일 권장 |
| `ensure_dir` | 디렉토리 보장 | A | 없음 | 낮음 | utils/paths | 생성 여부 | 범용 유틸 |
| `list_files` | 파일 목록 | A | 없음 | 낮음 | utils/paths | glob 패턴, recursive 동작 | 범용 유틸 |
| `describe` | 경로 상태 조회 | A | diag.check_paths와 중복 | 낮음 | utils/paths | 존재/미존재 파일 | diag와 통합 검토 필요 |

### apikeys.py

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `get` | API 키 단건 로드 | B | `.env` 파싱을 매 호출마다 반복 (캐싱 없음) | 중간 | utils/secrets | 탐색 순서 정확성, required=True 시 예외 | 핵심 함수, 캐싱 추가 권장 |
| `load_all` | 키 일괄 로드 | A | get에 종속, 자체 문제 없음 | 낮음 | utils/secrets | found/missing 분류 | get 개선 시 자동 개선 |
| `show` | 키 상태 마스킹 출력 | B | get을 키 수만큼 반복 호출 (매번 파일 파싱) | 중간 | utils/secrets | 마스킹 정확성 | 성능 이슈 (N회 파일 읽기) |
| `login_hf` | HuggingFace 로그인 | C | HF 전용, 도메인 종속 | 낮음 | notebook-only | 로그인 성공 여부 | 범용화 불필요, 노트북에 남기거나 별도 분리 |
| `_parse_env_file` | .env 직접 파싱 | B | dotenv 재구현, 엣지 케이스 미처리 (멀티라인, 이스케이프) | 중간 | utils/secrets | 다양한 .env 형식 | dotenv 있으면 위임하는 현재 전략은 합리적 |

### dataload.py

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `load` | 파일 로드 (확장자 자동 감지) | B | Colab 업로드 UI 폴백이 순수 로드와 혼재, 전역 `_drive_mounted` 상태 | 높음 | data/io | 확장자별 로드, 인코딩 폴백 | 파일 로드와 Colab 로직 분리 필요 |
| `load_many` | 복수 파일 로드 | B | basename 키 충돌 가능 (`a/train.csv`, `b/train.csv`) | 중간 | data/io | 동명 파일 처리 | 키 충돌 방어 필요 |
| `_read_file` | 확장자별 pandas reader | A | 인코딩 폴백 로직은 유용 | 낮음 | data/io | 확장자별 동작 | 핵심 유틸, 분리 가능 |
| `mount_drive` | Colab Drive 마운트 | C | Colab 전용, 전역 상태 변경 | 낮음 | notebook-only | Colab에서만 테스트 가능 | 로컬 테스트 불가 |

### comenv.py

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `setup` | 전체 환경 초기화 (시드+GPU+폰트+pandas+UTF8) | B | **God function** — 6개 이상의 책임 혼합, 240줄, 19개 파라미터, 전역 상태 대량 변경 | **높음** | 분할 필요 | 각 하위 기능별 독립 테스트 | 반드시 분할해야 팀에서 사용 가능 |
| `clear_gpu` | GPU 메모리 해제 | A | 없음 | 낮음 | utils/gpu | 메모리 변화 검증 | 독립적, 재사용 가능 |
| `_find_korean_font` | 한글 폰트 탐색 | A | 없음 | 낮음 | visualization/fonts | 플랫폼별 동작 | 시각화 모듈로 이동 권장 |
| `_check_cuda_compat` | CUDA 버전 호환성 체크 | A | subprocess 호출 포함 | 낮음 | utils/gpu 또는 diag | 버전 불일치 감지 | diag.check_gpu와 통합 가능 |

### pyenv.py (파일명 vs 헤더 불일치: 파일은 `pyenv.py`, 헤더는 `envcheck.py`)

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `show_versions` | 패키지 버전 출력 | B | CORE_PACKAGES 하드코딩, 출력과 데이터 반환 혼재 | 중간 | diag 또는 utils/env | 설치/미설치 분류 | diag.check_imports와 기능 중복 |
| `export_requirements` | requirements.txt 생성 | B | 파일 쓰기 부작용, CORE_PACKAGES 종속 | 중간 | utils/env | 파일 내용 정확성 | 유용하지만 `pip freeze`와 역할 겹침 |
| `compare_requirements` | 환경 비교 | A | 없음 | 낮음 | utils/env | 일치/불일치/미설치 분류 | 팀 프로젝트에서 유용 |

### diag.py

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `snapshot` | 전체 진단 한 번에 수집 | B | apikeys 모듈 직접 import (결합도) | 중간 | diag | 반환 dict 구조 | 핵심 진단 함수 |
| `check_paths` | 경로 상태 점검 | A | pathutil.describe와 중복 | 낮음 | diag | exists/readable/writable | 통합 검토 필요 |
| `check_imports` | 패키지 import 확인 | A | pyenv.show_versions와 중복 | 낮음 | diag | installed/version | 중복 정리 필요 |
| `check_gpu` | GPU 상태 확인 | A | comenv._check_cuda_compat과 부분 중복 | 낮음 | diag | torch 미설치 시 동작 | 진단 모듈에 적합 |
| `df_snapshot` | DataFrame 진단 | B | logger.df, eda.quick_look과 3중 중복 | 중간 | diag | 결측/중복/메모리 정확성 | 역할 경계 재정의 필요 |
| `catch` | 예외 래핑 + 진단 출력 | B | `reraise=False` 기본값 — 예외를 삼킴 | 중간 | diag | 예외 발생 시 snapshot 호출 여부 | reraise=True가 안전한 기본값 |
| `save_report` | 진단 파일 저장 | A | 없음 | 낮음 | diag | 파일 내용 정확성 | 디버깅에 유용 |

### eda.py

| 함수명 | 현재 역할 | 분류 | 주요 문제 | 리팩토링 필요도 | 권장 위치 | 테스트 포인트 | 한줄 판단 |
|--------|----------|------|----------|---------------|----------|-------------|----------|
| `quick_look` | DataFrame 전체 요약 출력 | C | print-only, 반환값 없음, logger.df/diag.df_snapshot과 중복 | 중간 | notebook-only | 없음 (출력만) | 노트북 전용 |
| `normalize_column_names` | 컬럼명 정규화 | A | 없음 | 낮음 | data/cleaning | 정규화 규칙, 중복 검증 | 즉시 모듈화 가능 |
| `split_cols` | 수치/범주 컬럼 분리 | B | verbose print 혼재 | 낮음 | data/cleaning | 타입별 분류 정확성 | print 제거하면 A |
| `check_duplicates` | 중복행 검사 | B | print 혼재 | 낮음 | data/cleaning | 중복 카운트 정확성 | print 제거하면 A |
| `clean_columns` | 컬럼명 정제 (copy 반환) | A | 없음 | 낮음 | data/cleaning | 복사본 반환 확인 | in-place 없음, 안전 |
| `summarize_missingness` | 결측치 요약표 반환 | A | 없음 | 낮음 | data/cleaning | 결측수/비율 정확성 | 순수 함수, 즉시 모듈화 |
| `plot_missingness` | 결측치 막대 그래프 | B | Axes 반환하지만 plt.show() 미호출 (inconsistent) | 중간 | visualization | 그래프 생성 여부 | 다른 plot 함수와 반환 규약 불일치 |
| `missing_report` | 결측 요약 + 출력 + 시각화 | C | **3중 책임** (계산+print+plot) | 높음 | notebook-only | 없음 (통합 함수) | 노트북 편의용, src 부적합 |
| `detect_outliers_iqr` | IQR 이상치 탐지 | B | print 내장 | 중간 | features | 이상치 개수/경계값 정확성 | print 제거하면 A |
| `_detect_date_columns` | 날짜 컬럼 자동 탐지 | B | sample_size 고정, 오탐 가능 | 중간 | data/cleaning | 다양한 날짜 형식 | 탐지 정확도 보장 어려움 |
| `coerce_datetime_columns` | datetime 변환 (copy) | A | 없음 | 낮음 | data/cleaning | 변환 정확성, errors 동작 | 순수 함수, 안전 |
| `add_datetime_parts` | 시간 파생 변수 추가 | A | 없음 | 낮음 | features | 파생 컬럼 생성 정확성 | 순수 함수, 안전 |
| `parse_dates` | 날짜 탐지 + 변환 + 파생 통합 | C | print, 자동 탐지, 3단계 통합 | 중간 | notebook-only | 자동 탐지 정확도 | 노트북 편의용 |
| `plot_histograms` | 수치형 히스토그램 | B | `plt.show()` 하드코딩, Axes 미반환 | 중간 | visualization | 서브플롯 레이아웃 | plt.show() 제거, Axes 반환 필요 |
| `plot_countplots` | 범주형 카운트플롯 | B | `plt.show()` 하드코딩, Axes 미반환 | 중간 | visualization | 서브플롯 레이아웃 | 위와 동일 |
| `plot_corr_heatmap` | 상관행렬 히트맵 | B | `plt.show()` 하드코딩, Axes 미반환 | 중간 | visualization | 마스크/상관계수 | 위와 동일 |
| `plot_target_comparison` | 타겟 기준 분포 비교 | B | `plt.show()` 하드코딩, Axes 미반환 | 중간 | visualization | 서브플롯 레이아웃 | 위와 동일 |

---

## 2. 함수별 상세 코멘트

### 모듈 횡단 이슈 (가장 먼저 해결해야 함)

**`_is_colab()` 4회 중복** (apikeys.py:44, dataload.py:33, diag.py:53, comenv.py:301)
**`_is_jupyter()` 2회 중복** (comenv.py:301, diag.py:61)
**`_get_version()` 2회 중복** (pyenv.py:48, diag.py:73)

- 이들은 `utils/env_detect.py` 같은 단일 모듈로 통합해야 합니다.
- 현재 각 모듈이 독립적으로 동일 함수를 정의하고 있어, 수정 시 4곳을 동시에 변경해야 하는 위험이 있습니다.

**DEFAULT_KEYS / DEFAULT_ENV_KEYS 중복** (apikeys.py:31 vs diag.py:43)
- 동일한 키 목록이 두 곳에 하드코딩되어 있습니다. 한 곳에서 관리해야 합니다.

**DataFrame 요약 기능 3중 중복**: `logger.df`, `diag.df_snapshot`, `eda.quick_look`
- 각각 다른 상세도를 가지지만, 사용자 입장에서 어느 것을 써야 하는지 불분명합니다.
- 역할을 명확히 구분하거나 통합해야 합니다.

---

### logger.py

- **현재 문제점**: `log()` 함수가 print(부작용)와 return(값 반환)을 동시에 수행합니다. 테스트 시 출력 캡처가 필요하고, 반환값만 쓰고 싶어도 print가 발생합니다.
- **공용화 가능 여부**: 가능. 노트북 로깅이라는 목적이 명확하고 외부 의존 없음.
- **개선 방향**: print와 문자열 포매팅 책임 분리. `format_log() → str`과 `print_log() → None`으로 나누면 테스트와 확장이 용이해짐.
- **public API 권장**: `log`, `info`, `step`, `warn`, `error`, `kv`, `timer` 공개. `line`, `section`은 내부 또는 선택 공개.

### pathutil.py

- **현재 문제점**: `DEFAULT_MARKERS = ("helper", ...)` — "helper"는 이 패키지 자체의 디렉토리명. 다른 프로젝트에서 사용하면 의미 없는 marker가 됨. `models_path`의 `create=True` 기본값은 함수 호출만으로 파일시스템을 변경하는 암묵적 부작용.
- **공용화 가능 여부**: markers 기본값과 create 기본값 수정 후 즉시 가능.
- **개선 방향**: markers에서 "helper" 제거, `create=False`를 모든 `*_path` 함수의 기본값으로 통일.
- **public API 권장**: `find_project_root`, `project_path`, `data_path`, `ensure_dir`, `list_files` 공개. `models_path`, `outputs_path`는 cookiecutter 구조에 맞춰 사용 시에만 공개.

### apikeys.py

- **현재 문제점**: `get()` 호출마다 `.env` 파일을 매번 파싱합니다. `show()`는 키 수 N번 × 파일 파싱 = N회 IO. `_parse_env_file`은 dotenv의 부분 재구현으로, 멀티라인 값, `\n` 이스케이프, 변수 참조(`${VAR}`) 등 미지원.
- **공용화 가능 여부**: 핵심 로직(키 탐색 체인)은 재사용 가치 높음.
- **개선 방향**: 파싱 결과를 `functools.lru_cache` 또는 모듈 수준 캐시로 보관. `login_hf`는 별도 모듈이나 노트북으로 분리.
- **public API 권장**: `get`, `load_all`, `show` 공개. `login_hf`는 공개하되 별도 네임스페이스 권장.

### dataload.py

- **현재 문제점**: `load()`가 "파일 읽기" + "Colab 업로드 UI 실행" + "Drive 마운트" 3가지를 한 함수에서 처리. 전역 `_drive_mounted` 플래그로 상태 관리. `load_many`의 basename 키는 경로가 다른 동명 파일에서 충돌.
- **공용화 가능 여부**: `_read_file` (확장자별 reader)은 즉시 분리 가능. Colab 관련 로직은 분리 필요.
- **개선 방향**: `_read_file`을 `load_file(path, encoding, **kwargs)`로 승격시키고, Colab 폴백은 별도 함수 또는 데코레이터로 분리. `load_many` 키를 full path 또는 충돌 시 경고.
- **public API 권장**: `load`, `load_many` 공개. `mount_drive`, `where`는 Colab 전용으로 분리.

### comenv.py — `setup()`

- **현재 문제점**: **이 리뷰에서 가장 큰 문제.** 19개 파라미터, 240줄, 최소 6개 독립 책임(시드 고정, UTF-8 설정, pandas 옵션, matplotlib 설정, CPU 스레드, GPU 설정)을 하나의 함수에 집약. `np.random.seed()`는 NumPy에서 deprecated (1.x 호환이지만 `np.random.default_rng()` 권장). 들여쓰기 불일치 (25행 `seed`, 34행 `force_utf8`이 0열에 위치).
- **공용화 가능 여부**: 분할 후 가능. 현재 형태로는 불가.
- **개선 방향**: `set_seed()`, `set_utf8()`, `set_pandas_display()`, `set_matplotlib()`, `set_cpu_threads()`, `set_gpu()` 등으로 분할. `setup()`은 이들을 조합하는 facade로 유지.
- **public API 권장**: 분할된 각 함수를 공개. `setup()`은 편의용 facade로 공개.

### pyenv.py

- **현재 문제점**: 파일명(`pyenv.py`)과 모듈 헤더(`helper/envcheck.py`)가 불일치 — 혼란 유발. `CORE_PACKAGES`가 이 프로젝트의 기술 스택에 하드코딩. `show_versions`는 `diag.check_imports`와 기능 중복.
- **공용화 가능 여부**: `compare_requirements`는 팀 프로젝트에서 유용. 나머지는 diag 모듈과 통합 검토 필요.
- **개선 방향**: 파일명 통일. `diag` 모듈과 역할 경계 재정의. `CORE_PACKAGES`를 설정 파일이나 인자로 외부화.
- **public API 권장**: `compare_requirements` 공개. `show_versions`, `export_requirements`는 diag 통합 후 결정.

### diag.py

- **현재 문제점**: `apikeys` 모듈을 직접 import하여 내부 함수(`_find_env_file`, `_parse_env_file`)에 접근 — 모듈 간 강결합. `catch()`의 `reraise=False` 기본값은 예외를 삼키므로 디버깅을 오히려 방해할 수 있음.
- **공용화 가능 여부**: 진단이라는 목적이 명확하여 모듈화 적합.
- **개선 방향**: apikeys 내부 함수 대신 public API(`apikeys.get`)를 사용하거나, 환경 키 조회를 diag 자체적으로 처리. `catch()`의 기본값을 `reraise=True`로 변경.
- **public API 권장**: `snapshot`, `check_paths`, `check_imports`, `check_gpu`, `save_report` 공개. `catch`는 주의사항과 함께 공개. `df_snapshot`은 logger.df와 역할 정리 후 결정.

### eda.py

- **현재 문제점**:
  1. **모듈 임포트 시 전역 부작용**: 22행 `plt.rcParams["axes.unicode_minus"] = False` — `import eda`만 해도 matplotlib 전역 설정이 변경됨.
  2. **시각화 함수의 `plt.show()` 하드코딩**: `plot_histograms`, `plot_countplots`, `plot_corr_heatmap`, `plot_target_comparison` 모두 `plt.show()`를 내부에서 호출. 서브플롯 조합, 파일 저장, 비대화형 환경에서 사용 불가.
  3. **Axes 미반환**: 위 4개 함수가 모두 `None` 반환. 호출자가 후처리(제목 변경, 축 조정 등) 불가.
  4. **`missing_report`의 3중 책임**: 데이터 계산 + print + plot을 하나에서 수행.
  5. **`detect_outliers_iqr`의 print 내장**: 순수 계산 함수여야 할 곳에 출력이 혼재.
- **공용화 가능 여부**: `normalize_column_names`, `summarize_missingness`, `coerce_datetime_columns`, `add_datetime_parts`는 즉시 가능. 시각화 함수는 `plt.show()` 제거 + Axes 반환 후 가능.
- **개선 방향**: 계산 함수와 시각화 함수를 물리적으로 분리 (파일 수준). 모든 plot 함수에서 `plt.show()` 제거, `fig, axes` 또는 `ax` 반환. 모듈 레벨 rcParams 설정 제거.
- **public API 권장**:
  - 공개: `normalize_column_names`, `split_cols`, `summarize_missingness`, `detect_outliers_iqr`, `coerce_datetime_columns`, `add_datetime_parts`
  - 시각화: `plot_missingness`, `plot_histograms`, `plot_countplots`, `plot_corr_heatmap`, `plot_target_comparison` — 리팩토링 후 공개
  - notebook-only: `quick_look`, `missing_report`, `parse_dates`

---

## 3. cookiecutter-data-science 통합 가이드

### 추천 디렉토리 구조

```
src/<project_name>/
├── __init__.py
├── utils/
│   ├── __init__.py           # public API re-export
│   ├── env_detect.py         # _is_colab, _is_jupyter, _get_version (중복 통합)
│   ├── logging.py            # logger.py 현재 내용
│   ├── paths.py              # pathutil.py 현재 내용
│   ├── secrets.py            # apikeys.py (login_hf 제외)
│   └── gpu.py                # clear_gpu, _check_cuda_compat
├── data/
│   ├── __init__.py
│   ├── io.py                 # dataload._read_file → load_file, load_many
│   └── cleaning.py           # normalize_column_names, clean_columns,
│                             # summarize_missingness, coerce_datetime_columns,
│                             # split_cols, check_duplicates
├── features/
│   ├── __init__.py
│   └── datetime.py           # add_datetime_parts
│                             # (detect_outliers_iqr도 여기 또는 data/stats.py)
├── visualization/
│   ├── __init__.py
│   ├── plots.py              # plot_histograms, plot_countplots, plot_corr_heatmap,
│   │                         # plot_target_comparison, plot_missingness
│   └── style.py              # _find_korean_font, matplotlib 설정 함수
├── diag/
│   ├── __init__.py
│   └── diagnostics.py        # diag.py + pyenv.py 통합
│                             # (snapshot, check_*, compare_requirements, save_report)
└── setup.py                  # comenv.setup → 분할된 facade
                              # (set_seed, set_utf8, set_pandas_display 등)

notebooks/
├── exploration/              # quick_look, missing_report, parse_dates 등
│   └── eda_template.ipynb    # notebook-only 함수 사용
tests/
├── test_utils/
│   ├── test_paths.py
│   ├── test_secrets.py
│   └── test_logging.py
├── test_data/
│   ├── test_io.py
│   └── test_cleaning.py
├── test_features/
│   └── test_datetime.py
├── test_visualization/
│   └── test_plots.py         # 이미지 비교 또는 Axes 속성 검증
└── test_diag/
    └── test_diagnostics.py
```

### 분류 기준

| 기준 | utils | data / features / visualization | notebook-only |
|------|-------|-------------------------------|---------------|
| 도메인 의존 | 없음 | pandas/sklearn/matplotlib 의존 있지만 범용 | 특정 분석 흐름에 종속 |
| 부작용 | 없거나 최소 (로깅, 디렉토리 생성) | DataFrame 반환 (copy), Axes 반환 | print, plt.show(), 자동 탐지 |
| 테스트 가능 | mock 없이 단위 테스트 | fixture DataFrame으로 테스트 | 수동 확인 필요 |
| 예시 | paths, secrets, logging, gpu | cleaning, io, datetime, plots | quick_look, missing_report, parse_dates |

### `__init__.py` 공개 API 설계

```python
# src/<project_name>/data/__init__.py
from .io import load_file, load_many
from .cleaning import (
    normalize_column_names,
    clean_columns,
    summarize_missingness,
    split_cols,
    check_duplicates,
    coerce_datetime_columns,
)

# src/<project_name>/visualization/__init__.py
from .plots import (
    plot_histograms,
    plot_countplots,
    plot_corr_heatmap,
    plot_target_comparison,
    plot_missingness,
)
from .style import setup_matplotlib  # 한글 폰트 + rcParams

# src/<project_name>/utils/__init__.py
from .paths import find_project_root, project_path, data_path, ensure_dir
from .secrets import get as get_secret, load_all as load_secrets
from .logging import log, info, warn, error, timer
```

### 파일 분리 기준

- **io.py**: 파일 읽기/쓰기만. Colab 로직 제외.
- **cleaning.py**: DataFrame → DataFrame 변환. print/plot 없음.
- **plots.py**: 모든 시각화. `plt.show()` 호출하지 않음. Axes 반환.
- **style.py**: matplotlib rcParams, 폰트 설정. 한 번만 호출.
- **env_detect.py**: 환경 감지 유틸. 다른 모듈이 import하여 사용.
- **secrets.py**: API 키 로드. `.env` 파싱 + 환경변수 + Colab.
- **diagnostics.py**: 디버깅/진단 전용. 에러 상황에서만 호출.

---

## 4. 개선된 예시 코드 스니펫

### 예시 1: `detect_outliers_iqr` — 계산과 출력 분리

```python
# data/stats.py
from __future__ import annotations
import pandas as pd


def detect_outliers_iqr(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    IQR 방식 이상치 탐지 결과를 DataFrame으로 반환.

    Parameters
    ----------
    df : pd.DataFrame
        분석 대상 데이터프레임.
    cols : list[str] | None
        대상 컬럼. None이면 수치형 전체.
    factor : float
        IQR 가중치 (기본 1.5).

    Returns
    -------
    pd.DataFrame
        컬럼: 이상치수, 비율%, 하한, 상한.
        이상치가 없으면 빈 DataFrame 반환.

    Raises
    ------
    KeyError
        존재하지 않는 컬럼 지정 시.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["이상치수", "비율%", "하한", "상한"])

    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        _validate_columns(df, cols)

    rows = []
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_out = int(((df[c] < lower) | (df[c] > upper)).sum())

        if n_out > 0:
            rows.append({
                "컬럼": c,
                "이상치수": n_out,
                "비율%": round(n_out / len(df) * 100, 1),
                "하한": round(lower, 2),
                "상한": round(upper, 2),
            })

    if not rows:
        return pd.DataFrame(columns=["이상치수", "비율%", "하한", "상한"])

    return pd.DataFrame(rows).set_index("컬럼")
```

변경 핵심: `print()` 제거 → 순수 함수. 노트북에서 출력이 필요하면 `display(detect_outliers_iqr(df))`.

### 예시 2: `plot_histograms` — plt.show() 제거 + Axes 반환

```python
# visualization/plots.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_histograms(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    bins: int = 30,
    figsize_per_ax: tuple[int, int] = (4, 3),
) -> Figure | None:
    """
    수치형 컬럼 서브플롯 히스토그램을 생성하여 Figure를 반환.

    호출자가 plt.show(), fig.savefig() 등을 직접 제어한다.

    Returns
    -------
    Figure | None
        수치형 컬럼이 없으면 None.
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    if not cols:
        return None

    n = len(cols)
    ncols_grid = min(3, n)
    nrows = -(-n // ncols_grid)  # 올림 나눗셈

    fig, axes = plt.subplots(
        nrows, ncols_grid,
        figsize=(figsize_per_ax[0] * ncols_grid, figsize_per_ax[1] * nrows),
    )
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, c in enumerate(cols):
        axes_flat[i].hist(df[c].dropna(), bins=bins, edgecolor="white", alpha=0.8)
        axes_flat[i].set_title(c, fontsize=11)
        axes_flat[i].tick_params(labelsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("수치형 분포", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
```

변경 핵심: `plt.show()` 제거, `Figure` 반환. 노트북에서 `plot_histograms(df)` → Jupyter가 자동 렌더링. 스크립트에서 `fig.savefig("hist.png")` 가능.

### 예시 3: `comenv.setup()` 분할 패턴

```python
# setup.py
from __future__ import annotations


def set_seed(seed: int = 42) -> None:
    """Python random + numpy + torch 시드를 고정."""
    import os
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def set_pandas_display(
    max_columns: int = 50,
    max_rows: int = 100,
    max_colwidth: int = 80,
) -> None:
    """pandas 출력 옵션을 설정."""
    import pandas as pd
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_colwidth", max_colwidth)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)


def setup(seed: int = 42, verbose: bool = True, **kwargs) -> dict:
    """
    전체 환경 초기화 facade.

    내부적으로 set_seed(), set_pandas_display() 등을 호출.
    개별 설정이 필요하면 각 함수를 직접 호출.
    """
    set_seed(seed)
    set_pandas_display(**{k: v for k, v in kwargs.items()
                          if k.startswith("pd_")})
    # ... 나머지 하위 함수 호출 ...
```

변경 핵심: 각 하위 설정을 독립 함수로 분리. `setup()`은 facade만 담당. 팀원이 "시드만 고정하고 matplotlib은 건드리지 않겠다"는 선택이 가능해짐.

---

## 5. 최종 총평

**판정: 개인용 스니펫 묶음 → 내부 라이브러리 초안 과도기**

**잘 되어 있는 점:**
- 파일 분리 의도(로깅/경로/키/로드/진단/EDA)가 합리적
- DataFrame 조작 함수 대부분이 copy를 반환하여 in-place 부작용 회피
- docstring과 사용 예시가 포함되어 있음
- Colab/로컬 환경 자동 분기라는 실용적 문제를 다루고 있음

**즉시 해결해야 할 구조적 문제 (우선순위 순):**

1. **`_is_colab()` 등 중복 함수 4곳 통합** — 가장 즉각적인 유지보수 리스크
2. **`comenv.setup()` 분할** — 19 파라미터 god function은 팀에서 사용 불가
3. **eda.py 시각화 함수에서 `plt.show()` 제거 + Axes/Figure 반환** — 재사용성의 핵심 차단 요인
4. **모듈 레벨 부작용 제거** — `eda.py` 22행의 `plt.rcParams` 변경, 이것은 import만으로 전역 상태를 바꿈
5. **pyenv.py 파일명/헤더 불일치 수정**
6. **diag.py → apikeys.py 내부 함수 직접 참조 제거**

**팀 프로젝트 투입 전 최소 조건:**
- 중복 유틸 통합 (1번)
- comenv 분할 (2번)
- 시각화 함수 반환 규약 통일 (3번)
- 계산 함수에서 print 제거 (detect_outliers_iqr, split_cols, check_duplicates 등)

이 4가지를 처리하면 "재사용 가능한 내부 라이브러리 초안"으로 격상 가능합니다. 현재 상태에서 팀원이 이 코드를 받으면, 어떤 함수가 전역 상태를 바꾸는지, 어떤 함수가 print를 하는지 예측할 수 없어 신뢰도가 낮을 것입니다.
