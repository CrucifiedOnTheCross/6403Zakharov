from __future__ import annotations
"""Анализ Global Emissions: выбросы на душу, разброс по странам, корреляции."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from lw3.pipeline import read_csv_chunks
from lw3.plots import save_bar, save_scatter
from lw3.utils import Welford, ensure_dir, find_first_match


def _resolve_columns(cols: Iterable[str]) -> Dict[str, str]:
    """Определить столбцы для Global Emissions по ключевым словам."""
    cols = list(cols)
    return {
        "country": (
            find_first_match(cols, ["country"]) or
            find_first_match(cols, ["nation"]) or
            find_first_match(cols, ["state"])
        ),
        "year": (
            find_first_match(cols, ["year"]) or
            find_first_match(cols, ["time", "year"])
        ),
        "emissions": (
            find_first_match(cols, ["emissions", "total"]) or
            find_first_match(cols, ["co2"]) or
            find_first_match(cols, ["ghg"])
        ),
        "population": (
            find_first_match(cols, ["population"]) or
            find_first_match(cols, ["pop"])
        ),
        "gdp": (
            find_first_match(cols, ["gdp"]) or
            find_first_match(cols, ["gross", "product"]) or
            find_first_match(cols, ["economy"])
        ),
    }


def run_all(csv_path: Path | str, parquet_path: Path | str, output_dir: Path | str):
    """Запустить анализ Global Emissions: per-capita, разброс, корреляции.

    Сохраняет графики и текстовую сводку по результатам расчетов.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    first = next(read_csv_chunks(csv_path, chunksize=50_000))
    col = _resolve_columns(first.columns)

    # Task 1: per-capita emissions per country
    sum_em_pop = defaultdict(lambda: {"e": 0.0, "p": 0.0})

    # Task 2: разброс суммы выбросов по странам
    welford_country = defaultdict(Welford)

    # Task 3: общие GDP и выбросы
    total_gdp = 0.0
    total_em = 0.0

    # Extra: корреляция population vs emissions
    corr_x = []
    corr_y = []

    for chunk in read_csv_chunks(csv_path, chunksize=200_000):
        country = col["country"]
        em = col["emissions"]
        pop = col["population"]
        gdp = col["gdp"]

        for cc in [em, pop, gdp]:
            if cc and cc in chunk.columns:
                chunk[cc] = pd.to_numeric(chunk[cc], errors="coerce")

        if (
            country and em and pop and
            country in chunk.columns and em in chunk.columns and pop in chunk.columns
        ):
            g1 = chunk[[country]].copy()
            g1["e"] = chunk[em].fillna(0)
            g1["p"] = chunk[pop].fillna(0)
            agg1 = g1.groupby(country).sum()
            for c, row in agg1.iterrows():
                sum_em_pop[str(c)]["e"] += float(row["e"])
                sum_em_pop[str(c)]["p"] += float(row["p"])

        if country and em and country in chunk.columns and em in chunk.columns:
            for c, e in zip(chunk[country], chunk[em]):
                if pd.notna(e):
                    welford_country[str(c)].update(float(e))

        if gdp and gdp in chunk.columns:
            total_gdp += float(pd.to_numeric(chunk[gdp], errors="coerce").fillna(0).sum())

        if em and em in chunk.columns:
            total_em += float(pd.to_numeric(chunk[em], errors="coerce").fillna(0).sum())

        if pop and em and pop in chunk.columns and em in chunk.columns:
            corr_x.extend(pd.to_numeric(chunk[pop], errors="coerce").fillna(0).tolist())
            corr_y.extend(pd.to_numeric(chunk[em], errors="coerce").fillna(0).tolist())

    per_capita = [(c, (v["e"] / v["p"])) for c, v in sum_em_pop.items() if v["p"]]
    per_capita_sorted = sorted(per_capita, key=lambda x: x[1])
    low3 = per_capita_sorted[:3]
    high3 = per_capita_sorted[-3:][::-1]
    save_bar(
        [c for c, _ in per_capita_sorted],
        [v for _, v in per_capita_sorted],
        title="Emissions: Выбросы на душу населения",
        ylabel="Per Capita",
        output_path=output_dir / "emissions_per_capita.png",
        rotation=90,
    )

    spread = sorted(
        [(c, w.std) for c, w in welford_country.items() if w.count > 1],
        key=lambda x: x[1],
    )
    save_bar(
        [c for c, _ in spread],
        [v for _, v in spread],
        title="Emissions: Разброс суммы выбросов по странам",
        ylabel="Std",
        output_path=output_dir / "emissions_spread.png",
        rotation=90,
    )

    corr = (
        float(pd.Series(corr_x).corr(pd.Series(corr_y)))
        if corr_x and corr_y
        else float("nan")
    )
    save_scatter(
        corr_x,
        corr_y,
        title=f"Emissions: Population vs Emissions (r={corr:.3f})",
        xlabel="Population",
        ylabel="Emissions",
        output_path=output_dir / "emissions_corr.png",
    )

    summary = [
        "Самые зеленые страны: " + ", ".join([f"{c}:{v:.4f}" for c, v in low3]),
        "Самые грязные страны: " + ", ".join([f"{c}:{v:.4f}" for c, v in high3]),
        "Минимальный разброс выбросов: "
        + ", ".join([f"{c}:{v:.3f}" for c, v in spread[:3]]),
        "Максимальный разброс выбросов: "
        + ", ".join([f"{c}:{v:.3f}" for c, v in spread[-3:][::-1]]),
        f"Общий GDP: {total_gdp:.2f}",
        f"Общие выбросы: {total_em:.2f}",
        f"Корреляция Population vs Emissions: {corr:.3f}",
    ]
    (output_dir / "emissions_summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )