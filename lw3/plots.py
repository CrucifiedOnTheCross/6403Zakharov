from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt


def _wrap_labels(labels: list[str], max_chars: int = 12) -> list[str]:
    """Перенос длинных меток по словам на несколько строк для читаемости."""
    wrapped = []
    for s in labels:
        if len(s) <= max_chars:
            wrapped.append(s)
        else:
            # перенос по словам, если есть пробелы; иначе по символам
            parts = []
            line = ""
            for w in s.split(" "):
                if len(line) + len(w) + 1 <= max_chars:
                    line = (line + " " + w).strip()
                else:
                    if line:
                        parts.append(line)
                    line = w
            if line:
                parts.append(line)
            wrapped.append("\n".join(parts))
    return wrapped


def save_bar(
    labels: Iterable[str],
    values: Iterable[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path | str,
    ci_low: Optional[Iterable[float]] = None,
    ci_high: Optional[Iterable[float]] = None,
    rotation: int = 45
    
):
    """Сохранить статический bar chart с опциональными доверительными интервалами."""
    labels = list(labels)
    values = list(values)
    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, values)
    if ci_low is not None and ci_high is not None:
        lows = list(ci_low)
        highs = list(ci_high)
        yerr = [
            [max(0.0, v - l) for v, l in zip(values, lows)],
            [max(0.0, h - v) for v, h in zip(values, highs)],
        ]
        plt.errorbar(x, values, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    plt.xticks(x, labels, rotation=rotation, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def _save_bar_single(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path | str,
    ci_low: Optional[Iterable[float]] = None,
    ci_high: Optional[Iterable[float]] = None,
    rotation: int = 45,
    horizontal: Optional[bool] = None,
):
    """Внутренний помощник для отрисовки одного bar chart (гориз/верт)."""
    n = len(labels)
    # Автовыбор: при большом числе меток делаем горизонтальный бар
    if horizontal is None:
        horizontal = n > 30

    # Оборачиваем длинные метки
    labels_wrapped = _wrap_labels(labels, max_chars=12)

    # Динамические размеры фигуры
    if horizontal:
        fig_w = 10
        fig_h = max(6, min(20, 0.35 * n + 2))
        plt.figure(figsize=(fig_w, fig_h))
        y = range(n)
        plt.barh(y, values)
        if ci_low is not None and ci_high is not None:
            lows = list(ci_low)
            highs = list(ci_high)
            xerr = [
                [max(0.0, v - l) for v, l in zip(values, lows)],
                [max(0.0, h - v) for v, h in zip(values, highs)],
            ]
            plt.errorbar(values, y, xerr=xerr, fmt="none", ecolor="black", capsize=4)
        plt.yticks(y, labels_wrapped, fontsize=8)
        plt.xlabel(ylabel)
    else:
        fig_w = max(10, min(24, 0.25 * n + 4))
        fig_h = 6
        plt.figure(figsize=(fig_w, fig_h))
        x = range(n)
        plt.bar(x, values)
        if ci_low is not None and ci_high is not None:
            lows = list(ci_low)
            highs = list(ci_high)
            yerr = [
                [max(0.0, v - l) for v, l in zip(values, lows)],
                [max(0.0, h - v) for v, h in zip(values, highs)],
            ]
            plt.errorbar(x, values, yerr=yerr, fmt="none", ecolor="black", capsize=4)
        plt.xticks(x, labels_wrapped, rotation=rotation, ha="right", fontsize=8)
        plt.ylabel(ylabel)

    plt.title(title)
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_line(
    labels: Iterable[str],
    values: Iterable[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path | str,
):
    """Сохранить линейный график по меткам оси X и значениям."""
    labels = list(labels)
    values = list(values)
    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.plot(x, values, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_scatter(
    x_values: Iterable[float],
    y_values: Iterable[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path | str,
):
    """Сохранить scatter plot для двух массивов значений."""
    x_values = list(x_values)
    y_values = list(y_values)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()