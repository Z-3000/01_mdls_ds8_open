"""
helper/logger.py — 노트북용 간단 로그 모듈
===========================================================================
사용법:
    from helper import logger

    logger.section("데이터 로드")
    logger.info("CSV 읽는 중", path="data/train.csv")
    logger.step("전처리 시작")
    logger.df("train_df", df)

    with logger.timer("모델 학습"):
        ...
===========================================================================
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
from typing import Any


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _format_fields(fields: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in fields.items()]
    return " | ".join(parts)


def line(char: str = "-", width: int = 72) -> None:
    """구분선을 출력합니다."""
    print(char * width)


def section(title: str, char: str = "=", width: int = 72) -> None:
    """섹션 제목을 출력합니다."""
    print(char * width)
    print(title)
    print(char * width)


def log(
    message: str,
    level: str = "INFO",
    show_time: bool = True,
    **fields,
) -> str:
    """
    한 줄 로그를 출력합니다.
    """
    prefix = f"[{_now()}] " if show_time else ""
    suffix = _format_fields(fields) if fields else ""
    text = f"{prefix}[{level}] {message}"
    if suffix:
        text += f" | {suffix}"
    print(text)
    return text


def info(message: str, **fields) -> str:
    """INFO 로그."""
    return log(message, level="INFO", **fields)


def step(message: str, **fields) -> str:
    """STEP 로그."""
    return log(message, level="STEP", **fields)


def warn(message: str, **fields) -> str:
    """WARN 로그."""
    return log(message, level="WARN", **fields)


def error(message: str, **fields) -> str:
    """ERROR 로그."""
    return log(message, level="ERROR", **fields)


def success(message: str, **fields) -> str:
    """SUCCESS 로그."""
    return log(message, level="OK", **fields)


def kv(title: str | None = None, **fields) -> None:
    """
    key=value 목록을 보기 좋게 출력합니다.
    """
    if title:
        print(title)
    for key, value in fields.items():
        print(f"  {key:<20}: {value}")


def df(
    name: str,
    df_obj: Any,
    show_columns: bool = False,
    max_cols: int = 10,
) -> None:
    """
    DataFrame의 shape / memory / 컬럼 미리보기를 출력합니다.
    pandas가 없어도 import 자체는 가능합니다.
    """
    shape = getattr(df_obj, "shape", None)
    columns = list(getattr(df_obj, "columns", []))

    memory_mb = None
    try:
        memory_mb = round(float(df_obj.memory_usage(deep=True).sum()) / (1024**2), 2)
    except Exception:
        pass

    info(
        f"DataFrame snapshot: {name}",
        rows=shape[0] if shape else None,
        cols=shape[1] if shape else None,
        memory_mb=memory_mb,
    )

    if show_columns:
        print(f"  columns: {columns[:max_cols]}")


@contextmanager
def timer(label: str, level: str = "INFO"):
    """
    코드 블록 실행 시간을 측정합니다.

    사용법:
        with logger.timer("모델 학습"):
            train_model()
    """
    log(f"{label} 시작", level=level)
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        log(f"{label} 완료", level=level, elapsed_sec=round(elapsed, 3))
