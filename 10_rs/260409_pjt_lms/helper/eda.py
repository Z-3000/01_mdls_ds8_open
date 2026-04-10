"""
helper/eda.py — EDA 헬퍼 함수 모음
==============================================
노트북 첫 셀에:
    %load_ext autoreload
    %autoreload 2
    from helper import eda

카테고리:
    [1] 데이터 파악
    [2] 전처리
    [3] EDA 시각화
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 한글 폰트 (Windows 기준, Mac은 'AppleGothic') ---
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# [1] 데이터 파악
# ============================================================

def quick_look(df: pd.DataFrame, name: str = "df") -> None:
    """
    shape, dtypes, 결측치, 메모리, 샘플 한 번에 출력.
    >>> quick_look(df, "매출")
    """
    print(f"\n{'='*55}")
    print(f"  [{name}]  {df.shape[0]:,}행 × {df.shape[1]}열  "
          f"| 메모리 {df.memory_usage(deep=True).sum()/1024**2:.1f}MB")
    print(f"{'='*55}")

    info = pd.DataFrame({
        "dtype": df.dtypes,
        "결측수": df.isnull().sum(),
        "결측%": (df.isnull().mean() * 100).round(1),
        "유니크": df.nunique(),
        "샘플값": df.iloc[0] if len(df) > 0 else None,
    })
    print(info.to_string())
    print()


def _validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"존재하지 않는 컬럼: {missing}")


def normalize_column_names(
    columns: pd.Index,
    *,
    validate_unique: bool = True,
) -> pd.Index:
    """
    컬럼명을 정규화한 새 Index를 반환.

    Parameters
    ----------
    columns : pd.Index
        원본 컬럼 인덱스
    validate_unique : bool
        정규화 후 중복 컬럼이 생기면 ValueError 발생 여부
    """
    normalized = (
        columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z가-힣_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    if validate_unique:
        duplicated = normalized[normalized.duplicated()].unique().tolist()
        if duplicated:
            raise ValueError(f"정규화 후 중복 컬럼 발생: {duplicated}")

    return pd.Index(normalized)


def split_cols(
    df: pd.DataFrame,
    *,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    수치형 / 범주형 컬럼 리스트를 튜플로 반환.
    """
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(exclude="number").columns.tolist()
    if verbose:
        print(f"수치형 {len(num)}개: {num}")
        print(f"범주형 {len(cat)}개: {cat}")
    return num, cat


def check_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    중복행 개수와 비율 출력 + 중복행 df 반환.
    """
    if subset is not None:
        _validate_columns(df, subset)

    dup = df.duplicated(subset=subset, keep=False)
    n = dup.sum()
    pct = (n / len(df) * 100) if len(df) else 0.0

    label = f"subset={subset}" if subset else "전체 컬럼"
    if verbose:
        print(f"중복행: {n:,}개 ({pct:.1f}%)  [{label} 기준]")

    return df[dup]


# ============================================================
# [2] 전처리
# ============================================================

def clean_columns(
    df: pd.DataFrame,
    *,
    validate_unique: bool = True,
) -> pd.DataFrame:
    """
    컬럼명 공백 제거 + 소문자화 + 특수문자를 _로 치환.
    """
    out = df.copy()
    out.columns = normalize_column_names(out.columns, validate_unique=validate_unique)
    return out


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치 요약표를 반환.
    결측 0인 컬럼은 제외.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["결측수", "결측%"])

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return pd.DataFrame(columns=["결측수", "결측%"])

    return pd.DataFrame({
        "결측수": missing.astype(int),
        "결측%": (missing / len(df) * 100).round(1),
    })


def plot_missingness(
    report: pd.DataFrame,
    *,
    figsize: tuple[int, int] | None = None,
):
    """
    결측치 요약표를 가로 막대그래프로 시각화하고 Axes를 반환.
    """
    if report.empty:
        raise ValueError("비어 있지 않은 결측치 요약표가 필요합니다.")

    fig, ax = plt.subplots(figsize=figsize or (8, max(3, len(report) * 0.4)))
    ordered = report["결측%"][::-1]
    ax.barh(report.index[::-1], ordered)
    ax.set_xlabel("결측 비율 (%)")
    ax.set_title("결측치 현황")

    for i, v in enumerate(ordered):
        ax.text(v + 0.3, i, f"{v}%", va="center", fontsize=9)

    return ax


def missing_report(
    df: pd.DataFrame,     # 입력 데이터: 분석할 판다스 데이터프레임
    plot: bool = True,    # 시각화 여부: 기본값은 True (그래프 출력)
) -> pd.DataFrame:        # 반환 타입: 결측치 정보가 담긴 데이터프레임
    """
    결측치 현황을 비율순으로 정리 + 막대 시각화.
    결측 0인 컬럼은 제외.
    """
    report = summarize_missingness(df)

    if report.empty:
        print("결측치 없음")
        return report

    print(report.to_string())

    if plot:
        plot_missingness(report)
        plt.tight_layout()
        plt.show()

    return report

def detect_outliers_iqr(
    df: pd.DataFrame,                # 입력 데이터: 분석 대상이 되는 판다스 데이터프레임
    cols: list[str] | None = None,   # 대상 컬럼: 특정 컬럼 지정 가능 (None일 경우 자동 선택)
    factor: float = 1.5,             # 이상치 가중치: 통계적으로 보통 1.5를 사용 (조절 가능)
) -> pd.DataFrame:
    """
    IQR 방식 이상치 탐지. 컬럼별 이상치 개수 + 경계값 출력.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["이상치수", "비율%", "하한", "상한"])

    if cols is None:                                               # 분석 대상 컬럼을 지정하지 않았을 경우
        cols = df.select_dtypes(include="number").columns.tolist() # 수치형 데이터(int, float 등) 컬럼만 추출
    else:
        _validate_columns(df, cols)

    results = []                                              
    for c in cols:                # 대상 컬럼 하나씩 순회
        q1 = df[c].quantile(0.25) # 제1사분위수(하위 25% 지점) 계산
        q3 = df[c].quantile(0.75) # 제3사분위수(상위 25% 지점) 계산
        iqr = q3 - q1             # IQR(사분위 범위) 계산: 데이터의 중간 50% 폭
        
        # 이상치 판단 경계값(Fence) 계산
        lower, upper = q1 - factor * iqr, q3 + factor * iqr   # Lower Bound, Upper Bound 설정
        
        # 실제 데이터가 경계값을 벗어나는지 확인하여 개수 합산
        n_out = ((df[c] < lower) | (df[c] > upper)).sum()     
        
        if n_out > 0:                                          # 이상치가 1개라도 발견된 경우
            results.append({                                   # 딕셔너리 형태로 결과 리스트에 추가
                "컬럼": c,                                     # 분석한 컬럼명
                "이상치수": n_out,                             # 탐지된 이상치 개수
                "비율%": round(n_out / len(df) * 100, 1),      # 전체 데이터 대비 이상치 비중
                "하한": round(lower, 2),                       # 계산된 하한 경계값
                "상한": round(upper, 2),                       # 계산된 상한 경계값
            })

    if not results:                                           # 모든 컬럼에서 이상치가 없는 경우
        print("IQR 기준 이상치 없음")                         # 안내 메시지 출력
        return pd.DataFrame()                                 # 빈 데이터프레임 반환 후 종료

    out_df = pd.DataFrame(results).set_index("컬럼")          # 결과 리스트를 데이터프레임으로 변환 및 인덱스 설정
    print(out_df.to_string())                                 
    return out_df                                             

def _detect_date_columns(
    df: pd.DataFrame,
    *,
    sample_size: int = 20,
) -> list[str]:
    detected: list[str] = []
    for c in df.select_dtypes(include="object").columns:
        sample = df[c].dropna().head(sample_size)
        if sample.empty:
            continue
        try:
            pd.to_datetime(sample)
            detected.append(c)
        except (ValueError, TypeError):
            continue
    return detected


def coerce_datetime_columns(
    df: pd.DataFrame,
    cols: list[str],
    *,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    지정 컬럼을 datetime으로 변환한 복사본을 반환.
    """
    _validate_columns(df, cols)
    out = df.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c], errors=errors)
    return out


def add_datetime_parts(
    df: pd.DataFrame,
    cols: list[str],
    *,
    parts: tuple[str, ...] = ("year", "month", "day", "dow", "hour"),
) -> pd.DataFrame:
    """
    datetime 컬럼에서 파생 시간 컬럼을 추가한 복사본을 반환.
    """
    _validate_columns(df, cols)
    out = df.copy()

    for c in cols:
        if not pd.api.types.is_datetime64_any_dtype(out[c]):
            raise TypeError(f"datetime 타입이 아닌 컬럼: {c}")

        if "year" in parts:
            out[f"{c}_year"] = out[c].dt.year
        if "month" in parts:
            out[f"{c}_month"] = out[c].dt.month
        if "day" in parts:
            out[f"{c}_day"] = out[c].dt.day
        if "dow" in parts:
            out[f"{c}_dow"] = out[c].dt.dayofweek
        if "hour" in parts:
            out[f"{c}_hour"] = out[c].dt.hour

    return out


def parse_dates(
    df: pd.DataFrame,                 # 입력 데이터: 분석할 판다스 데이터프레임
    cols: list[str] | None = None,    # 대상 컬럼: 변환할 컬럼 리스트 (None이면 자동 탐색)
    add_parts: bool = True,           # 파생 변수 생성: True일 경우 연/월/요일 등 분리 추가
) -> pd.DataFrame:
    """
    날짜 컬럼 → datetime 변환 + 연/월/요일 파생변수 자동 추가.
    cols 미지정 시 object 컬럼 중 자동 탐지.
    """
    if cols is None:                                          # 변환할 컬럼이 명시되지 않은 경우 자동 탐지 로직 실행
        cols = _detect_date_columns(df)
        if cols:                                              # 자동 탐지된 결과가 있을 경우
            print(f"날짜 컬럼 자동 탐지: {cols}")
        else:                                                 # 탐지된 것이 없을 경우 처리
            print("자동 탐지된 날짜 컬럼 없음 → cols를 직접 지정하세요.")
            return df                                         # 변환 없이 그대로 반환

    out = coerce_datetime_columns(df, cols, errors="coerce")
    if add_parts:
        out = add_datetime_parts(out, cols)

    return out                                               


# ============================================================
# [3] EDA 시각화
# ============================================================

def _hide_unused_axes(axes, used: int) -> None:        # axes: 서브플롯 배열, used: 실제 데이터를 그린 그래프 개수
    """서브플롯에서 사용하지 않는 axes를 숨김."""
    for j in range(used, len(axes)):       # 실제 사용된 개수(used)부터 전체 axes의 끝까지 반복
        axes[j].set_visible(False)         # 해당 인덱스의 그래프 창을 화면에서 보이지 않게 설정


def plot_histograms(
    df: pd.DataFrame,                          # 입력 데이터: 분석할 판다스 데이터프레임
    cols: list[str] | None = None,             # 대상 컬럼: 히스토그램을 그릴 컬럼 리스트 (None이면 자동 선택)
    bins: int = 30,                            # 막대 개수: 데이터를 나누는 구간의 수 (기본값 30)
    figsize_per_ax: tuple[int, int] = (4, 3),  # 개별 그래프 크기: 각 서브플롯의 (가로, 세로) 크기
) -> None:
    """
    수치형 컬럼 전체를 서브플롯 히스토그램으로 한 번에 시각화.
    """
    if cols is None:                                               # 컬럼이 지정되지 않은 경우
        cols = df.select_dtypes(include="number").columns.tolist() # 수치형(int, float) 컬럼만 자동으로 추출
    if not cols:                                                   # 수치형 컬럼이 아예 없는 경우 처리
        print("수치형 컬럼 없음")
        return

    n = len(cols)                                # 그려야 할 그래프의 총 개수
    ncols_grid = min(3, n)                       # 한 행에 배치할 최대 그래프 수를 3개로 제한
    nrows = (n + ncols_grid - 1) // ncols_grid   # 필요한 총 행(row) 수 계산 (올림 계산 방식)
    
    # 전체 피겨(Figure) 생성 및 크기 설정: (열 개수 * 단위 너비, 행 개수 * 단위 높이)
    fig, axes = plt.subplots(
        nrows, ncols_grid, 
        figsize=(figsize_per_ax[0]*ncols_grid, figsize_per_ax[1]*nrows)
    )
    
    # axes 처리: 그래프가 1개일 때는 단일 객체이므로, 반복문을 위해 리스트나 numpy 배열로 평탄화(flatten)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, c in enumerate(cols):                                              # 컬럼 리스트를 순회하며 그래프 그리기
        axes[i].hist(df[c].dropna(), bins=bins, edgecolor="white", alpha=0.8) # 결측치 제외 후 히스토그램 생성
        axes[i].set_title(c, fontsize=11)                                     # 각 서브플롯의 제목을 컬럼명으로 설정
        axes[i].tick_params(labelsize=9)                                      # 축 눈금 글자 크기 조정

    _hide_unused_axes(axes, n)               # 데이터가 그려지지 않은 빈 그래프 칸 숨기기 (앞서 정의한 함수)

    plt.suptitle("수치형 분포", fontsize=13, y=1.01)      # 전체 그래프의 중앙 제목 설정 및 위치 조정
    plt.tight_layout()                                    # 서브플롯 간 간격 자동 최적화
    plt.show()                                            # 최종 결과물 화면 출력


def plot_countplots(
    df: pd.DataFrame,                            # 입력 데이터: 분석할 판다스 데이터프레임
    cols: list[str] | None = None,               # 대상 컬럼: 시각화할 범주형 컬럼 리스트 (None이면 자동 선택)
    top_n: int = 10,                             # 표시 개수: 카테고리가 너무 많을 경우 상위 n개만 표시
    figsize_per_ax: tuple[int, int] = (5, 3),    # 개별 그래프 크기: 각 서브플롯의 (가로, 세로) 크기
) -> None:
    """
    범주형 컬럼 전체를 countplot으로 한 번에 시각화.
    카테고리가 top_n개 초과면 상위 top_n개만 표시.
    """
    if cols is None:                                               # 컬럼이 지정되지 않은 경우
        cols = df.select_dtypes(exclude="number").columns.tolist() # 수치형이 아닌(object, category 등) 컬럼만 추출
    if not cols:                                                   # 범주형 컬럼이 아예 없는 경우 처리
        print("범주형 컬럼 없음")
        return

    n = len(cols)                                 # 시각화할 컬럼의 총 개수
    ncols_grid = min(2, n)                        # 한 행에 최대 2개의 그래프 배치 (가로 막대라 공간 확보 필요)
    nrows = (n + ncols_grid - 1) // ncols_grid    # 필요한 총 행(row) 수 계산
    
    # 전체 피겨(Figure) 생성: 열 개수와 행 개수에 맞춰 전체 캔버스 크기 조절
    fig, axes = plt.subplots(
        nrows, ncols_grid, 
        figsize=(figsize_per_ax[0]*ncols_grid, figsize_per_ax[1]*nrows)
    )
    
    # axes 처리: 1개일 때는 객체, 여러 개일 때는 배열이므로 반복문을 위해 1차원 평탄화
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, c in enumerate(cols):                              # 컬럼 리스트를 순회하며 그래프 작성
        # 빈도수가 높은 순서대로 인덱스(범주명)를 추출하여 상위 top_n개만 선정
        order = df[c].value_counts().index[:top_n]
        
        # Seaborn의 countplot 실행: y축에 데이터를 넣어 가로 막대 그래프로 생성 (가독성 향상)
        sns.countplot(y=df[c], order=order, ax=axes[i])
        
        # 각 서브플롯 제목 설정 (표시된 카테고리 개수 명시)
        axes[i].set_title(f"{c} (상위 {min(top_n, len(order))})", fontsize=11)
        axes[i].set_ylabel("")                                # y축 라벨(컬럼명 중복) 제거로 깔끔하게 처리
        axes[i].tick_params(labelsize=9)                      # 범주명 글자 크기 조정

    _hide_unused_axes(axes, n)                                # 데이터가 없는 나머지 빈 칸 숨기기

    plt.suptitle("범주형 분포", fontsize=13, y=1.01)           # 전체 그래프 최상단 제목 설정
    plt.tight_layout()                                        # 서브플롯 간 겹침 방지 및 간격 최적화
    plt.show()                                                # 그래프 출력


def plot_corr_heatmap(
    df: pd.DataFrame,                         # 입력 데이터: 상관계수를 계산할 데이터프레임
    cols: list[str] | None = None,            # 대상 컬럼: 수치형 컬럼 리스트 (None이면 자동 선택)
    figsize: tuple[int, int] = (8, 7),        # 그래프 크기: 히트맵 전체의 (가로, 세로) 크기
    annot_size: int = 9,                      # 글자 크기: 히트맵 내부에 표시될 숫자(상관계수)의 크기
) -> None:
    """
    상관행렬 히트맵 (하삼각형 마스크 적용, 상관계수 숫자 표시).
    """
    if cols is None:                                               # 컬럼이 지정되지 않은 경우
        cols = df.select_dtypes(include="number").columns.tolist() # 수치형 데이터 컬럼만 자동으로 추출
    
    if len(cols) < 2:                                     # 상관분석은 최소 2개의 변수가 있어야 가능함
        print("수치형 컬럼 2개 이상 필요")
        return

    corr = df[cols].corr()                                # 피어슨 상관계수 행렬 계산 (-1부터 1 사이 값)
    
    # 마스크 생성: 대칭인 상관행렬에서 중복되는 상삼각형(Upper Triangle) 부분을 가리기 위함
    # np.triu는 상삼각형 행렬을 반환하며, 이를 True로 설정하여 heatmap의 mask 인자로 전달
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)                    # 그래프를 그릴 도화지(Figure)와 축(Ax) 설정
    
    # Seaborn 히트맵 그리기
    sns.heatmap(
        corr,                            # 상관계수 데이터
        mask=mask,                       # 위에서 만든 마스크 적용 (True인 부분은 빈칸 처리)
        annot=True,                      # 칸 안에 상관계수 숫자 표시 여부
        fmt=".2f",                       # 숫자를 소수점 둘째 자리까지 표시
        cmap="RdBu_r",                   # 컬러맵: 양의 상관은 Blue, 음의 상관은 Red (중앙은 흰색)
        center=0,                        # 0을 기준으로 색상의 중심을 설정 (상관계수의 중립점)
        square=True,                     # 각 칸을 정사각형 모양으로 고정
        annot_kws={"size": annot_size},  # 숫자 글자 크기 등 세부 속성 설정
        ax=ax                            # 미리 생성한 ax 객체에 그래프를 그림
    )
    
    ax.set_title("상관관계 히트맵", fontsize=13)   # 그래프 제목 설정
    plt.tight_layout()                      # 레이아웃 여백 최적화
    plt.show()                              # 결과 출력

def plot_target_comparison(
    df: pd.DataFrame,                          # 입력 데이터: 분석할 데이터프레임
    target: str,                               # 기준 컬럼: 분류 기준이 되는 범주형 타겟 
    cols: list[str] | None = None,             # 비교 컬럼: 분포를 확인할 수치형 피처 리스트
    kind: str = "box",                         # 그래프 종류: 'box'(박스 플롯) 또는 'violin'(바이올린 플롯)
    figsize_per_ax: tuple[int, int] = (5, 3),  # 개별 그래프 크기: 각 서브플롯의 (가로, 세로) 크기
) -> None:
    """
    target(범주형) 기준으로 수치형 피처 분포 비교.
    kind: 'box' 또는 'violin'
    """
    if cols is None:                                          # 비교 대상 컬럼이 지정되지 않은 경우
        # 수치형 컬럼들 중 타겟 컬럼 본인을 제외하고 리스트 생성
        cols = [c for c in df.select_dtypes(include="number").columns 
                if c != target]
                
    if not cols:                                              # 비교할 수 있는 수치형 데이터가 없는 경우
        print("비교할 수치형 컬럼 없음")
        return

    n = len(cols)                                             # 시각화할 피처의 개수
    ncols_grid = min(3, n)                                    # 한 행에 최대 3개까지 배치
    nrows = (n + ncols_grid - 1) // ncols_grid                # 필요한 총 행(row) 수 계산
    
    # 전체 피겨 생성: 그리드 설정 및 동적 크기 조절
    fig, axes = plt.subplots(
        nrows, ncols_grid, 
        figsize=(figsize_per_ax[0]*ncols_grid, figsize_per_ax[1]*nrows)
    )
    
    # axes 처리: 반복문 사용을 위해 1차원 배열로 평탄화
    axes = np.array(axes).flatten() if n > 1 else [axes]

    # 시각화 함수 선택: 'box'면 boxplot, 아니면 violinplot 사용
    plot_fn = sns.boxplot if kind == "box" else sns.violinplot

    for i, c in enumerate(cols):                              # 컬럼별로 순회하며 그래프 작성
        # x축에는 기준(target), y축에는 값(c)을 할당하여 그룹별 분포 비교
        plot_fn(data=df, x=target, y=c, ax=axes[i])
        
        axes[i].set_title(c, fontsize=11)          # 서브플롯 제목을 컬럼명으로 설정
        axes[i].tick_params(labelsize=9)           # 축 눈금 폰트 조절

    _hide_unused_axes(axes, n)                                # 데이터가 그려지지 않은 빈 그리드 칸 숨기기

    plt.suptitle(f"'{target}' 기준 분포 비교", fontsize=13, y=1.01) # 전체 최상단 제목 설정
    plt.tight_layout()                                              # 서브플롯 레이아웃 최적화
    plt.show()                                                      # 결과 출력
