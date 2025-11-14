from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Optional

import pandas as pd


def read_csv_chunks(
    path: Path | str,
    chunksize: int = 100_000,
    usecols: Optional[Iterable[str]] = None,
    dtype: Optional[dict] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Генератор построчного чтения CSV крупными частями.

    - Использует pandas.read_csv с параметром chunksize.
    - Позволяет ограничивать столбцы через usecols для экономии памяти.
    """
    path = Path(path)
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols, dtype=dtype):
        yield chunk


def pass_through(df_iter: Iterable[pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
    """Простой генератор-прослойка для композиции пайплайнов."""
    for df in df_iter:
        yield df