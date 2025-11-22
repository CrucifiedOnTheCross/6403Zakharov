from __future__ import annotations

"""Компоненты пайплайна чтения CSV чанками и простые генераторы."""

from pathlib import Path
from typing import Generator, Optional, Iterable

import pandas as pd


def read_csv_chunks(
    path: Path | str,
    chunksize: int = 100,
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
