from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt


def _wrap_labels(labels: list[str], max_chars: int = 12) -> list[str]:
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
    rotation: int = 45,
    horizontal: Optional[bool] = None,
    max_bars_per_plot: Optional[int] = None,
):
    labels = list(labels)
    values = list(values)

    # Авто-пагинация при очень большом числе столбцов
    n = len(labels)
    if max_bars_per_plot is None:
        max_bars_per_plot = 60  # разумный лимит по ширине

    if n > max_bars_per_plot:
        # разбиваем на части и сохраняем несколько файлов
        output_path = Path(output_path)
        for i in range(0, n, max_bars_per_plot):
            sl_labels = labels[i : i + max_bars_per_plot]
            sl_values = values[i : i + max_bars_per_plot]
            sl_ci_low = list(ci_low)[i : i + max_bars_per_plot] if ci_low is not None else None
            sl_ci_high = list(ci_high)[i : i + max_bars_per_plot] if ci_high is not None else None
            # первый кусок сохраняем в исходное имя, остальные — с суффиксом
            if i == 0:
                out = output_path
            else:
                out = output_path.with_name(f"{output_path.stem}_part{(i // max_bars_per_plot) + 1}{output_path.suffix}")
            _save_bar_single(sl_labels, sl_values, title=title + f" (part {(i // max_bars_per_plot) + 1})" if i else title, ylabel=ylabel, output_path=out, ci_low=sl_ci_low, ci_high=sl_ci_high, rotation=rotation, horizontal=horizontal)
        return

    _save_bar_single(labels, values, title=title, ylabel=ylabel, output_path=output_path, ci_low=ci_low, ci_high=ci_high, rotation=rotation, horizontal=horizontal)


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


def save_line(labels: Iterable[str], values: Iterable[float], *, title: str, ylabel: str, output_path: Path | str):
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


def save_scatter(x_values: Iterable[float], y_values: Iterable[float], *, title: str, xlabel: str, ylabel: str, output_path: Path | str):
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