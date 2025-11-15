from pathlib import Path
from typing import Iterable, Optional
import matplotlib.pyplot as plt


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
    """Сохранить вертикальную столбчатую диаграмму с опциональными ДИ."""
    labels = list(labels)
    values = list(values)
    lows = list(ci_low) if ci_low is not None else None
    highs = list(ci_high) if ci_high is not None else None

    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    if lows is not None:
        lows = [lows[i] for i in order]
    if highs is not None:
        highs = [highs[i] for i in order]
    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, values)

    if lows is not None and highs is not None:
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


def save_line(
    labels: Iterable[str],
    values: Iterable[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path | str
):
    """Сохранить линейный график по меткам оси X и значениям."""
    labels = list(labels)
    values = list(values)
    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.plot(x, values, marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
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
    output_path: Path | str
):
    """Сохранить scatter plot для двух массивов значений."""
    x_values = list(x_values)
    y_values = list(y_values)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_line_dual(
    labels: Iterable[str],
    values1: Iterable[float],
    values2: Iterable[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path | str,
    legend1: str = "Value",
    legend2: str = "MA",
):
    labels = list(labels)
    v1 = list(values1)
    v2 = list(values2)
    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.plot(x, v1, marker="o", label=legend1)
    plt.plot(x, v2, linestyle="--", label=legend2)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
