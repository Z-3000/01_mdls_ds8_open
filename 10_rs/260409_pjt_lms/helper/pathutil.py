"""
helper/pathutil.py — 프로젝트 경로 헬퍼
===========================================================================
사용법:
    from helper import pathutil

    root = pathutil.find_project_root()
    train_path = pathutil.data_path("train.csv")
    model_dir = pathutil.models_path(create=True)
    out_dir = pathutil.outputs_path("figures", create=True, is_dir=True)
    files = pathutil.list_files("data", pattern="*.csv")
===========================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_MARKERS = ("helper", "data", ".git", ".env")


def _to_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _resolve(path: str | Path) -> Path:
    return _to_path(path).resolve(strict=False)


def _prepare_target(target: Path, create: bool, is_dir: bool) -> Path:
    if create:
        if is_dir:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
    return target


def cwd() -> str:
    """현재 작업 디렉토리 절대경로를 반환합니다."""
    return str(Path.cwd().resolve(strict=False))


def find_project_root(
    start: str | Path = ".",
    markers: tuple[str, ...] = DEFAULT_MARKERS,
    max_depth: int = 5,
    verbose: bool = False,
) -> str:
    """
    프로젝트 루트를 탐색합니다.

    규칙:
    - 현재 폴더 또는 상위 폴더에 `helper/`, `data/`, `.git`, `.env` 중 하나가 있으면 후보로 간주
    - 가장 먼저 찾은 폴더를 반환
    - 못 찾으면 start의 절대경로를 반환
    """
    current = _resolve(start)
    if current.is_file():
        current = current.parent

    candidates = [current] + list(current.parents[:max_depth])
    for candidate in candidates:
        if any((candidate / marker).exists() for marker in markers):
            if verbose:
                print(f"[pathutil] project root: {candidate}")
            return str(candidate)

    if verbose:
        print(f"[pathutil] project root fallback: {current}")
    return str(current)


def project_path(
    *parts: str,
    start: str | Path = ".",
    markers: tuple[str, ...] = DEFAULT_MARKERS,
    create: bool = False,
    is_dir: bool = False,
) -> str:
    """
    프로젝트 루트 기준 경로를 생성합니다.
    """
    root = Path(find_project_root(start=start, markers=markers))
    target = root.joinpath(*parts)
    return str(_prepare_target(target, create=create, is_dir=is_dir).resolve(strict=False))


def data_path(
    *parts: str,
    start: str | Path = ".",
    subdir: str | None = None,
    create: bool = False,
    is_dir: bool = False,
) -> str:
    """
    `data/` 기준 경로를 생성합니다.
    """
    base_parts = ["data"]
    if subdir:
        base_parts.append(subdir)
    base_parts.extend(parts)
    return project_path(*base_parts, start=start, create=create, is_dir=is_dir)


def models_path(
    *parts: str,
    start: str | Path = ".",
    create: bool = True,
    is_dir: bool = False,
) -> str:
    """
    `models/` 기준 경로를 생성합니다.
    """
    return project_path("models", *parts, start=start, create=create, is_dir=is_dir)


def outputs_path(
    *parts: str,
    start: str | Path = ".",
    create: bool = True,
    is_dir: bool = False,
) -> str:
    """
    `outputs/` 기준 경로를 생성합니다.
    """
    return project_path("outputs", *parts, start=start, create=create, is_dir=is_dir)


def ensure_dir(path: str | Path) -> str:
    """
    디렉토리가 없으면 생성하고 절대경로를 반환합니다.
    """
    target = _resolve(path)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def list_files(
    path: str | Path = ".",
    pattern: str = "*",
    recursive: bool = False,
    absolute: bool = True,
) -> list[str]:
    """
    폴더 내 파일 목록을 반환합니다.
    """
    base = _resolve(path)
    matches = base.rglob(pattern) if recursive else base.glob(pattern)
    files = [p.resolve(strict=False) if absolute else p for p in matches if p.is_file()]
    return sorted(str(p) for p in files)


def describe(path: str | Path) -> dict[str, Any]:
    """
    경로 상태를 딕셔너리로 반환합니다.
    """
    target = _resolve(path)
    exists = target.exists()
    return {
        "input": str(path),
        "resolved": str(target),
        "exists": exists,
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "parent": str(target.parent),
        "parent_exists": target.parent.exists(),
        "suffix": target.suffix,
        "size_bytes": target.stat().st_size if exists and target.is_file() else None,
    }
