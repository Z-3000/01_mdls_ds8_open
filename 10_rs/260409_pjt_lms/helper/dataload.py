# helper/dataload.py — 데이터 파일 로드 (로컬 우선, Colab 폴백)
# ============================================================================
# 사용법:
#   from helper import dataload
#
#   # 로컬 파일 로드
#   df = dataload.load("data/train.csv")
#   df = dataload.load("data/sales.xlsx", sheet_name="2024")
#   df = dataload.load("data/records.json")
#   df = dataload.load("data/big_table.parquet")
#
#   # 로컬에 없으면 Colab 업로드 UI 자동 실행
#   df = dataload.load("my_data.csv")
#
#   # Colab Google Drive 마운트 후 로드
#   df = dataload.load("my_data.csv", drive=True)
#   df = dataload.load("my_data.csv", drive=True, drive_root="/content/drive/MyDrive/datasets")
#
#   # 여러 파일 한 번에 로드
#   dfs = dataload.load_many(["train.csv", "test.csv", "sample.csv"])
#
#   # 환경 확인
#   dataload.where()  # → "colab" 또는 "local"
# ============================================================================

import os
import pandas as pd


# ============================================================================
# 환경 감지
# ============================================================================
def _is_colab() -> bool:
    """현재 실행 환경이 Google Colab인지 판별"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def where() -> str:
    """현재 환경을 문자열로 반환: 'colab' 또는 'local'"""
    return "colab" if _is_colab() else "local"


# ============================================================================
# Google Drive 마운트
# ============================================================================
_drive_mounted = False


def mount_drive(mount_point: str = "/content/drive", force: bool = False):
    """
    Colab에서 Google Drive를 마운트합니다.
    이미 마운트된 경우 건너뜁니다 (force=True로 재마운트 가능).
    """
    global _drive_mounted
    if not _is_colab():
        print("[dataload] 로컬 환경 — Drive 마운트 불필요")
        return

    if _drive_mounted and not force:
        return

    from google.colab import drive  # type: ignore
    drive.mount(mount_point, force_remount=force)
    _drive_mounted = True


# ============================================================================
# 핵심 로드 함수
# ============================================================================
def load(
    path: str,
    encoding: str = "utf-8",
    drive: bool = False,
    drive_root: str = "/content/drive/MyDrive",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    path : str
        파일 경로 또는 파일명
    encoding : str
        CSV 인코딩 (기본 "utf-8", 한글 깨지면 "cp949" 또는 "euc-kr")
    drive : bool
        True면 Colab Google Drive에서 로드 (기본 False)
    drive_root : str
        Drive 내 기본 디렉토리 (기본 "/content/drive/MyDrive")
    verbose : bool
        로드 정보 출력 (기본 True)
    **kwargs
        pandas read 함수에 전달할 추가 인자
        예: sheet_name="Sheet2", sep="\\t", nrows=1000

    Returns
    -------
    pd.DataFrame
    """
    full_path = _resolve_path(path, drive, drive_root)

    # ── 로컬/Drive에 파일이 있으면 바로 로드 ──
    if os.path.isfile(full_path):
        df = _read_file(full_path, encoding=encoding, **kwargs)
        if verbose:
            _print_info(full_path, df)
        return df

    # ── 파일이 없고 Colab이면 → 업로드 폴백 ──
    if _is_colab() and not drive:
        if verbose:
            print(f"[dataload] '{path}' 로컬에 없음 → Colab 파일 업로드 실행")
        full_path = _colab_upload(path)
        df = _read_file(full_path, encoding=encoding, **kwargs)
        if verbose:
            _print_info(full_path, df)
        return df

    # ── 어디에도 없으면 에러 ──
    raise FileNotFoundError(
        f"[dataload] 파일을 찾을 수 없습니다: '{full_path}'\n"
        f"  환경: {where()}\n"
        f"  현재 디렉토리: {os.getcwd()}\n"
        f"  팁: 경로를 확인하거나, drive=True로 Google Drive에서 로드해보세요."
    )


def load_many(
    paths: list,
    encoding: str = "utf-8",
    drive: bool = False,
    drive_root: str = "/content/drive/MyDrive",
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    여러 파일을 한 번에 로드합니다.

    Parameters
    ----------
    paths : list[str]
        파일 경로 리스트

    Returns
    -------
    dict : {파일명: DataFrame}

    사용법:
        dfs = dataload.load_many(["train.csv", "test.csv"])
        train_df = dfs["train.csv"]
    """
    result = {}
    for p in paths:
        name = os.path.basename(p)
        result[name] = load(
            p, encoding=encoding, drive=drive,
            drive_root=drive_root, verbose=verbose, **kwargs
        )
    return result


# ============================================================================
# 내부 함수
# ============================================================================
def _resolve_path(path: str, drive: bool, drive_root: str) -> str:
    """경로를 환경에 맞게 해석"""
    if drive and _is_colab():
        mount_drive()
        # 절대경로가 아니면 drive_root 기준으로 결합
        if not os.path.isabs(path):
            return os.path.join(drive_root, path)
    return path


def _read_file(path: str, encoding: str = "utf-8", **kwargs) -> pd.DataFrame:
    """확장자에 따라 적절한 pandas reader 호출"""
    ext = os.path.splitext(path)[1].lower()

    readers = {
        ".csv":     lambda: pd.read_csv(path, encoding=encoding, **kwargs),
        ".tsv":     lambda: pd.read_csv(path, sep="\t", encoding=encoding, **kwargs),
        ".xlsx":    lambda: pd.read_excel(path, engine="openpyxl", **kwargs),
        ".xls":     lambda: pd.read_excel(path, engine="xlrd", **kwargs),
        ".json":    lambda: pd.read_json(path, encoding=encoding, **kwargs),
        ".jsonl":   lambda: pd.read_json(path, lines=True, encoding=encoding, **kwargs),
        ".parquet": lambda: pd.read_parquet(path, **kwargs),
        ".feather": lambda: pd.read_feather(path, **kwargs),
        ".pkl":     lambda: pd.read_pickle(path, **kwargs),
        ".pickle":  lambda: pd.read_pickle(path, **kwargs),
    }

    reader = readers.get(ext)
    if reader is None:
        raise ValueError(
            f"[dataload] 지원하지 않는 파일 형식: '{ext}'\n"
            f"  지원 형식: {', '.join(readers.keys())}"
        )

    # CSV/TSV: utf-8 실패 시 cp949 자동 재시도
    if ext in (".csv", ".tsv"):
        try:
            return reader()
        except UnicodeDecodeError:
            fallback_enc = "cp949" if encoding == "utf-8" else encoding
            print(f"[dataload] UTF-8 디코딩 실패 → {fallback_enc}로 재시도")
            kwargs_copy = dict(kwargs)
            if ext == ".tsv":
                return pd.read_csv(path, sep="\t", encoding=fallback_enc, **kwargs_copy)
            return pd.read_csv(path, encoding=fallback_enc, **kwargs_copy)

    return reader()


def _colab_upload(expected_name: str) -> str:
    """Colab 파일 업로드 UI를 실행하고 업로드된 파일 경로를 반환"""
    from google.colab import files  # type: ignore
    uploaded = files.upload()

    if not uploaded:
        raise FileNotFoundError("[dataload] 파일 업로드가 취소되었습니다.")

    # 업로드된 파일 중 기대하는 파일명이 있으면 우선 반환
    if expected_name in uploaded:
        return expected_name

    # 없으면 첫 번째 파일 반환
    first = list(uploaded.keys())[0]
    print(f"[dataload] '{expected_name}' 대신 '{first}'가 업로드됨 → 해당 파일 사용")
    return first


def _print_info(path: str, df: pd.DataFrame):
    """로드 결과 요약 출력"""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[dataload] ✓ {os.path.basename(path)}  |  "
          f"{df.shape[0]:,} rows × {df.shape[1]} cols  |  "
          f"{size_mb:.1f} MB  |  {where()}")