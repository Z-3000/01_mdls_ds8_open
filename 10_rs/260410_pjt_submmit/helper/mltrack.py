# helper/mltrack.py
"""MLflow 백엔드 스위처.

환경변수:
    MLFLOW_TRACKING_URI       : file:./mlruns | https://dagshub.../x.mlflow | http://host:5000
    MLFLOW_TRACKING_USERNAME  : (선택) Basic Auth
    MLFLOW_TRACKING_PASSWORD  : (선택) Basic Auth
    MLFLOW_EXPERIMENT_NAME    : 실험 이름 (없으면 default)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlflow
from dotenv import load_dotenv


def setup_mlflow(
    experiment: Optional[str] = None,
    env_file: str | os.PathLike = '.env',
    verbose: bool = True,
) -> str:
    """환경변수 기준으로 MLflow tracking 설정.

    반환값: 실제 적용된 tracking URI
    """
    # 1) .env 로드 (있으면)
    if Path(env_file).exists():
        load_dotenv(env_file, override=False)

    uri = os.environ.get('MLFLOW_TRACKING_URI', 'file:./mlruns')
    exp = experiment or os.environ.get('MLFLOW_EXPERIMENT_NAME', 'default')

    mlflow.set_tracking_uri(uri)

    # 2) DagsHub 인 경우: dagshub 헬퍼가 있으면 우선 사용 (OAuth 캐시 활용)
    if 'dagshub.com' in uri:
        try:
            import dagshub
            # URI 에서 user/repo 추출: https://dagshub.com/<user>/<repo>.mlflow
            path = uri.split('dagshub.com/')[-1].removesuffix('.mlflow')
            owner, repo = path.split('/', 1)
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        except Exception:
            # 실패하면 환경변수(Basic Auth)로 fallback — MLflow 가 알아서 사용
            pass

    # 3) Experiment 준비
    mlflow.set_experiment(exp)

    if verbose:
        print(f'[mltrack] tracking URI : {mlflow.get_tracking_uri()}')
        print(f'[mltrack] experiment   : {exp}')

    return mlflow.get_tracking_uri()