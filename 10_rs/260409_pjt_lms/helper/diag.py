"""
helper/diag.py — 에러 대응용 진단 모듈
===========================================================================
사용법:
    from helper import diag

    diag.snapshot()
    diag.check_paths(["data/train.csv", ".env"])
    diag.check_imports(["pandas", "numpy", "torch"])
    diag.check_env(["OPENAI_API_KEY", "HF_TOKEN"])
    diag.check_gpu()
    diag.save_report("debug_report.txt")

    # 예외를 잡고 traceback + 환경 진단 출력
    result = diag.catch(my_func, arg1, arg2, reraise=False)
===========================================================================
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any, Callable

DEFAULT_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "torch",
    "xgboost",
    "lightgbm",
    "transformers",
]

DEFAULT_ENV_KEYS = [
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "LANGCHAIN_API_KEY",
    "WANDB_API_KEY",
]


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return type(shell).__name__ in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except (ImportError, NameError):
        return False


def _get_version(pkg: str) -> str | None:
    try:
        return pkg_version(pkg)
    except PackageNotFoundError:
        return None


def _mask(value: str | None, enabled: bool = True) -> str | None:
    if value is None:
        return None
    if not enabled:
        return value
    if len(value) <= 12:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _print_header(title: str, width: int = 72) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def _print_kv(data: dict[str, Any], indent: str = "  ") -> None:
    for key, value in data.items():
        print(f"{indent}{key:<20}: {value}")


def _safe_resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _get_env_status(
    keys: list[str] | None = None,
    env_path: str = "auto",
    mask: bool = True,
) -> dict[str, Any]:
    from . import apikeys

    if keys is None:
        keys = DEFAULT_ENV_KEYS

    resolved_env = apikeys._find_env_file(env_path)
    env_values = apikeys._parse_env_file(resolved_env)

    result: dict[str, Any] = {
        "env_path": resolved_env,
        "env_exists": os.path.isfile(resolved_env),
        "keys": {},
    }

    for key_name in keys:
        source = None
        value = None

        if key_name in env_values:
            value = env_values[key_name]
            source = ".env"
        elif key_name in os.environ:
            value = os.environ[key_name]
            source = "os.environ"
        elif _is_colab():
            try:
                from google.colab import userdata  # type: ignore

                value = userdata.get(key_name)
                source = "colab_secret" if value else None
            except Exception:
                value = None
                source = None

        result["keys"][key_name] = {
            "found": value is not None,
            "source": source,
            "value": _mask(value, enabled=mask),
        }

    return result


def _get_system_status() -> dict[str, Any]:
    if _is_colab():
        env_name = "colab"
    elif _is_jupyter():
        env_name = "jupyter"
    else:
        env_name = "local"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env_name,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "cwd": os.getcwd(),
        "stdout_encoding": sys.stdout.encoding or "unknown",
        "sys_path_count": len(sys.path),
        "helper_importable": importlib.util.find_spec("helper") is not None,
    }


def check_paths(
    paths: list[str] | tuple[str, ...],
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    파일/폴더 경로 상태를 진단합니다.

    Parameters
    ----------
    paths : list[str] | tuple[str, ...]
        확인할 경로 목록
    verbose : bool
        출력 여부 (기본 True)

    Returns
    -------
    dict : {원본경로: 상태정보}
    """
    results: dict[str, dict[str, Any]] = {}

    for raw_path in paths:
        resolved = _safe_resolve(raw_path)
        exists = resolved.exists()
        info = {
            "input": raw_path,
            "resolved": str(resolved),
            "exists": exists,
            "is_file": resolved.is_file(),
            "is_dir": resolved.is_dir(),
            "parent_exists": resolved.parent.exists(),
            "readable": os.access(resolved, os.R_OK) if exists else False,
            "writable": os.access(resolved, os.W_OK) if exists else False,
            "size_bytes": resolved.stat().st_size if exists and resolved.is_file() else None,
        }
        results[raw_path] = info

    if verbose:
        _print_header("diag.check_paths()")
        for raw_path, info in results.items():
            print(f"[{raw_path}]")
            _print_kv(info, indent="  ")
            print("-" * 72)

    return results


def check_imports(
    packages: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    패키지 import 가능 여부와 버전을 확인합니다.
    """
    if packages is None:
        packages = DEFAULT_PACKAGES

    results: dict[str, dict[str, Any]] = {}
    for pkg in packages:
        spec = importlib.util.find_spec(pkg)
        ver = _get_version(pkg)
        results[pkg] = {
            "installed": spec is not None,
            "version": ver,
        }

    if verbose:
        _print_header("diag.check_imports()")
        for pkg, info in results.items():
            version_text = info["version"] or "(미설치)"
            print(f"  {pkg:<20}: {version_text}")

    return results


def check_env(
    keys: list[str] | None = None,
    env_path: str = "auto",
    mask: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    .env / 환경변수 / Colab secret 기준으로 키 존재 여부를 확인합니다.
    """
    result = _get_env_status(keys=keys, env_path=env_path, mask=mask)

    if verbose:
        _print_header("diag.check_env()")
        print(f"  env_path            : {result['env_path']}")
        print(f"  env_exists          : {result['env_exists']}")
        print("-" * 72)
        for key_name, info in result["keys"].items():
            print(
                f"  {key_name:<20}: found={info['found']}  "
                f"source={info['source']}  value={info['value']}"
            )

    return result


def check_gpu(verbose: bool = True) -> dict[str, Any]:
    """
    PyTorch / CUDA / GPU 상태를 확인합니다.
    """
    result: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "torch_cuda": None,
        "device_count": 0,
        "device_name": None,
        "memory_total_gb": None,
        "memory_free_gb": None,
    }

    try:
        import torch

        result["torch_installed"] = True
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda or "N/A"

        if torch.cuda.is_available():
            result["device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            result["device_name"] = props.name

            try:
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                result["memory_total_gb"] = round(total_mem / (1024**3), 2)
                result["memory_free_gb"] = round(free_mem / (1024**3), 2)
            except Exception:
                pass
    except ImportError:
        pass

    if verbose:
        _print_header("diag.check_gpu()")
        _print_kv(result)

    return result


def df_snapshot(
    df: Any,
    name: str = "df",
    max_cols: int = 20,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    DataFrame 상태를 에러 대응용으로 짧게 요약합니다.
    pandas가 없어도 함수 정의 자체는 import 가능합니다.
    """
    shape = getattr(df, "shape", None)
    columns = list(getattr(df, "columns", []))
    dtypes = getattr(df, "dtypes", None)

    info: dict[str, Any] = {
        "name": name,
        "rows": shape[0] if shape else None,
        "cols": shape[1] if shape else None,
        "columns_preview": columns[:max_cols],
        "column_count": len(columns),
        "duplicate_rows": None,
        "memory_mb": None,
        "missing_top": {},
        "dtypes_top": {},
    }

    if dtypes is not None:
        try:
            info["dtypes_top"] = {str(k): str(v) for k, v in dtypes.head(max_cols).items()}
        except Exception:
            pass

    try:
        info["duplicate_rows"] = int(df.duplicated().sum())
    except Exception:
        pass

    try:
        memory_bytes = int(df.memory_usage(deep=True).sum())
        info["memory_mb"] = round(memory_bytes / (1024**2), 2)
    except Exception:
        pass

    try:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(max_cols)
        info["missing_top"] = {str(k): int(v) for k, v in missing.items()}
    except Exception:
        pass

    if verbose:
        _print_header(f"diag.df_snapshot() — {name}")
        _print_kv(
            {
                "rows": info["rows"],
                "cols": info["cols"],
                "memory_mb": info["memory_mb"],
                "duplicate_rows": info["duplicate_rows"],
            }
        )
        print("-" * 72)
        print(f"  columns_preview      : {info['columns_preview']}")
        print(f"  dtypes_top          : {info['dtypes_top']}")
        print(f"  missing_top         : {info['missing_top']}")

    return info


def snapshot(
    paths: list[str] | None = None,
    env_keys: list[str] | None = None,
    packages: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    현재 세션의 핵심 진단 정보를 한 번에 수집합니다.
    """
    if paths is None:
        paths = []

    result = {
        "system": _get_system_status(),
        "paths": check_paths(paths, verbose=False) if paths else {},
        "imports": check_imports(packages=packages, verbose=False),
        "env": _get_env_status(keys=env_keys, env_path="auto", mask=True),
        "gpu": check_gpu(verbose=False),
    }

    if verbose:
        _print_header("diag.snapshot()")
        print("[system]")
        _print_kv(result["system"])
        print("-" * 72)

        print("[imports]")
        for pkg, info in result["imports"].items():
            version_text = info["version"] or "(미설치)"
            print(f"  {pkg:<20}: {version_text}")
        print("-" * 72)

        print("[env]")
        print(f"  env_path            : {result['env']['env_path']}")
        print(f"  env_exists          : {result['env']['env_exists']}")
        for key_name, info in result["env"]["keys"].items():
            print(
                f"  {key_name:<20}: found={info['found']}  "
                f"source={info['source']}  value={info['value']}"
            )
        print("-" * 72)

        print("[gpu]")
        _print_kv(result["gpu"])

        if result["paths"]:
            print("-" * 72)
            print("[paths]")
            for raw_path, info in result["paths"].items():
                print(f"  {raw_path}")
                print(f"    resolved          : {info['resolved']}")
                print(f"    exists            : {info['exists']}")
                print(f"    is_file           : {info['is_file']}")
                print(f"    is_dir            : {info['is_dir']}")

    return result


def save_report(
    path: str = "debug_report.txt",
    paths: list[str] | None = None,
    env_keys: list[str] | None = None,
    packages: list[str] | None = None,
) -> str:
    """
    현재 진단 결과를 텍스트 파일로 저장합니다.
    """
    report = snapshot(paths=paths, env_keys=env_keys, packages=packages, verbose=False)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("debug_report")
    lines.append("=" * 72)

    lines.append("[system]")
    for key, value in report["system"].items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("[imports]")
    for key, value in report["imports"].items():
        lines.append(f"{key}: installed={value['installed']} version={value['version']}")
    lines.append("")

    lines.append("[env]")
    lines.append(f"env_path: {report['env']['env_path']}")
    lines.append(f"env_exists: {report['env']['env_exists']}")
    for key, value in report["env"]["keys"].items():
        lines.append(
            f"{key}: found={value['found']} source={value['source']} value={value['value']}"
        )
    lines.append("")

    lines.append("[gpu]")
    for key, value in report["gpu"].items():
        lines.append(f"{key}: {value}")
    lines.append("")

    if report["paths"]:
        lines.append("[paths]")
        for key, value in report["paths"].items():
            lines.append(f"{key}: {value}")
        lines.append("")

    target = Path(path).expanduser()
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[diag] 리포트 저장 완료: {target.resolve(strict=False)}")
    return str(target.resolve(strict=False))


def catch(
    func: Callable[..., Any],
    *args,
    reraise: bool = False,
    paths: list[str] | None = None,
    env_keys: list[str] | None = None,
    packages: list[str] | None = None,
    **kwargs,
) -> Any:
    """
    함수를 실행하고 예외 발생 시 traceback + snapshot을 출력합니다.
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        _print_header("diag.catch() — 예외 발생")
        traceback.print_exc()
        print("-" * 72)
        snapshot(paths=paths, env_keys=env_keys, packages=packages, verbose=True)
        if reraise:
            raise
        return None
