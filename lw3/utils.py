from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _normalize_colname(name: str) -> str:
    """Нормализовать имя столбца: нижний регистр и замена не-алфавитных на пробелы."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def find_first_match(columns: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    """Найти первый столбец, содержащий все указанные ключевые слова.

    Сравнение ведется регистронезависимо по нормализованным именам столбцов.
    """
    keys = [_normalize_colname(k) for k in keywords]
    for col in columns:
        norm = _normalize_colname(col)
        if all(k in norm for k in keys):
            return col
    return None


def find_all_matches(columns: Sequence[str], keywords: Sequence[str]) -> List[str]:
    """Вернуть все столбцы, содержащие указанные ключевые слова."""
    keys = [_normalize_colname(k) for k in keywords]
    result: List[str] = []
    for col in columns:
        norm = _normalize_colname(col)
        if all(k in norm for k in keys):
            result.append(col)
    return result


@dataclass
class Welford:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: float):
        """Обновить накопители по алгоритму Вэлфорда для нового значения x."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return float("nan")
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        v = self.variance
        return float("nan") if math.isnan(v) else math.sqrt(v)


def ci95_mean(values: Iterable[float]) -> tuple[float, float, float, int]:
    """Вернуть (mean, ci_low, ci_high, n) для 95% CI.

    Используется нормальное приближение: CI = mean ± 1.96 * (std / sqrt(n)).
    """
    arr = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    z = 1.96
    return (mean, mean - z * se, mean + z * se, n)


def moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    """Скользящее среднее по ряду с окном `window` и min_periods=1."""
    return series.rolling(window=window, min_periods=1).mean()


def ensure_dir(path: Path | str):
    """Создать директорию (включая родительские) при необходимости."""
    Path(path).mkdir(parents=True, exist_ok=True)