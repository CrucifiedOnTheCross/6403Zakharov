from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def ensure_parquet(csv_path: Path | str, parquet_path: Path | str, *, usecols: Optional[Iterable[str]] = None):
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        return
    df = pd.read_csv(csv_path, usecols=usecols)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


def compare_read_speed(csv_path: Path | str, parquet_path: Path | str, *, usecols: Optional[Iterable[str]] = None) -> tuple[float, float]:
    import time
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    t0 = time.perf_counter()
    pd.read_csv(csv_path)  # чтение целиком (без генератора)
    t_csv = time.perf_counter() - t0

    t1 = time.perf_counter()
    pd.read_parquet(parquet_path)
    t_parquet = time.perf_counter() - t1

    return t_csv, t_parquet