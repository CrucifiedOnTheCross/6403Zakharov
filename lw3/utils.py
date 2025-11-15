from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Welford:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: float):
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


def moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ensure_dir(path: Path | str):
    Path(path).mkdir(parents=True, exist_ok=True)
