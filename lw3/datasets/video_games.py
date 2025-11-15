from __future__ import annotations
"""Анализ Video Games: продажи по годам, оценки по издателям и рейтинги."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from lw3.pipeline import read_csv_chunks
from lw3.plots import save_bar, save_line, save_scatter
from lw3.utils import Welford, ensure_dir, find_first_match


def _resolve_columns(cols: Iterable[str]) -> Dict[str, str]:
    """Определить столбцы для Video Games по ключевым словам."""
    cols = list(cols)
    return {
        "year": find_first_match(cols, ["year"]) or find_first_match(cols, ["release", "year"]),
        "sales": find_first_match(cols, ["sales"]) or find_first_match(cols, ["metrics", "sales"]),
        "publisher": find_first_match(cols, ["publisher"]) or find_first_match(cols, ["company"]) or find_first_match(cols, ["developer"]),
        "review": find_first_match(cols, ["review", "score"]) or find_first_match(cols, ["rating", "review"]) or find_first_match(cols, ["critic", "score"]),
        "rating": find_first_match(cols, ["rating"]) or find_first_match(cols, ["content", "rating"]),
    }


def run_all(csv_path: Path | str, parquet_path: Path | str, output_dir: Path | str):
    """Запустить анализ Video Games: продажи, разброс оценок, рейтинги, корреляции."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    first = next(read_csv_chunks(csv_path, chunksize=50_000))
    col = _resolve_columns(first.columns)

    # Task 1: лучший/худший год по продажам
    sales_year = defaultdict(float)

    # Task 2: разброс оценки игр по издателям
    publisher_review = defaultdict(Welford)

    # Task 3: количество игр каждого рейтинга по годам
    rating_year = defaultdict(lambda: defaultdict(int))  # rating -> year -> count

    # Extra: корреляция review vs sales
    corr_x = []
    corr_y = []

    for chunk in read_csv_chunks(csv_path, chunksize=200_000):
        y = col["year"]
        s = col["sales"]
        p = col["publisher"]
        r = col["review"]
        rtg = col["rating"]

        for cc in [s, r]:
            if cc and cc in chunk.columns:
                chunk[cc] = pd.to_numeric(chunk[cc], errors="coerce")

        if y and s and y in chunk.columns and s in chunk.columns:
            g1 = chunk[[y]].copy()
            g1["sales"] = chunk[s].fillna(0)
            agg1 = g1.groupby(y).sum()["sales"]
            for year, val in agg1.items():
                sales_year[int(year)] += float(val)

        if p and r and p in chunk.columns and r in chunk.columns:
            for pub, rev in zip(chunk[p], chunk[r]):
                if pd.notna(rev):
                    publisher_review[str(pub)].update(float(rev))

        if y and rtg and y in chunk.columns and rtg in chunk.columns:
            g3 = chunk[[y, rtg]].copy()
            g3["one"] = 1
            agg3 = g3.groupby([y, rtg]).sum()["one"]
            for (year, rating), count in agg3.items():
                rating_year[str(rating)][int(year)] += int(count)

        if r and s and r in chunk.columns and s in chunk.columns:
            corr_x.extend(pd.to_numeric(chunk[r], errors="coerce").fillna(0).tolist())
            corr_y.extend(pd.to_numeric(chunk[s], errors="coerce").fillna(0).tolist())

    # Итоги
    year_sorted = sorted(sales_year.items(), key=lambda x: x[1])
    worst_year = year_sorted[0] if year_sorted else None
    best_year = year_sorted[-1] if year_sorted else None
    save_bar([str(y) for y, _ in year_sorted], [v for _, v in year_sorted], title="VideoGames: Продажи по годам", ylabel="Sales", output_path=output_dir / "vg_sales_year.png")

    pub_std = sorted([(p, w.std) for p, w in publisher_review.items() if w.count > 1], key=lambda x: x[1])
    save_bar([p for p, _ in pub_std], [v for _, v in pub_std], title="VideoGames: Разброс оценок по издателям", ylabel="Std", output_path=output_dir / "vg_publisher_review_spread.png", rotation=90)

    # Рейтинги по годам (линейные графики)
    for rating, yr_map in rating_year.items():
        ys = sorted(yr_map.items())
        save_line([str(y) for y, _ in ys], [c for _, c in ys], title=f"VideoGames: Кол-во игр рейтинга {rating}", ylabel="Count", output_path=output_dir / f"vg_rating_{rating}_count.png")

    corr = float(pd.Series(corr_x).corr(pd.Series(corr_y))) if corr_x and corr_y else float("nan")
    save_scatter(corr_x, corr_y, title=f"VideoGames: Review vs Sales (r={corr:.3f})", xlabel="Review.Score", ylabel="Sales", output_path=output_dir / "vg_corr.png")

    summary = [
        f"Лучший год по продажам: {best_year}",
        f"Худший год по продажам: {worst_year}",
        "Издатели с минимальным разбросом: " + ", ".join([f"{p}:{v:.3f}" for p, v in pub_std[:3]]),
        "Издатели с максимальным разбросом: " + ", ".join([f"{p}:{v:.3f}" for p, v in pub_std[-3:][::-1]]),
        f"Корреляция Review vs Sales: {corr:.3f}",
    ]
    (output_dir / "vg_summary.txt").write_text("\n".join(summary), encoding="utf-8")
