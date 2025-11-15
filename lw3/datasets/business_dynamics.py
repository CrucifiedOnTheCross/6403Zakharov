from __future__ import annotations
"""Анализ Business Dynamics: NJCR, RR, динамика JDR и корреляции."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from lw3.pipeline import read_csv_chunks
from lw3.plots import save_bar, save_line, save_scatter
from lw3.utils import Welford, ensure_dir, find_first_match, moving_average


def _resolve_columns(cols: Iterable[str]) -> Dict[str, str]:
    """Определить релевантные столбцы для Business Dynamics по ключевым словам."""
    cols = list(cols)
    return {
        "state": (
            find_first_match(cols, ["state"]) or
            find_first_match(cols, ["region"])
        ),
        "year": (
            find_first_match(cols, ["year"]) or
            find_first_match(cols, ["time", "year"])
        ),
        "net_job_creation_rate": (
            find_first_match(cols, ["net", "job", "creation", "rate"]) or
            find_first_match(cols, ["job", "creation", "rate"])
        ),
        "reallocation_rate": (
            find_first_match(cols, ["reallocation", "rate"]) or
            find_first_match(cols, ["labor", "reallocation"])
        ),
        "job_destruction_rate": (
            find_first_match(cols, ["job", "destruction", "rate"]) or
            find_first_match(cols, ["destruction", "rate"])
        ),
        "job_creation_rate": (
            find_first_match(cols, ["job", "creation", "rate"]) or
            find_first_match(cols, ["creation", "rate"])
        ),
    }


def run_all(csv_path: Path | str, parquet_path: Path | str, output_dir: Path | str):
    """Запустить анализ датасета Business Dynamics и сохранить результаты.

    Считает средний Net Job Creation Rate, разброс Reallocation Rate,
    строит динамику JDR для наиболее нестабильного штата и вычисляет корреляции.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    first = next(read_csv_chunks(csv_path, chunksize=50_000))
    col = _resolve_columns(first.columns)

    # Task 1: средний темп создания рабочих мест по штатам
    avg_njcr = defaultdict(lambda: {"sum": 0.0, "cnt": 0})

    # Task 2: разброс показателя Reallocation Rate
    spread_rr = defaultdict(Welford)

    # Task 3: динамика Job Destruction Rate для наиболее нестабильного штата
    ts_jdr = defaultdict(list)  # state -> [(year, value)]

    # Extra: корреляция между Job Creation Rate и Job Destruction Rate
    corr_x = []
    corr_y = []

    for chunk in read_csv_chunks(csv_path, chunksize=200_000):
        st = col["state"]
        y = col["year"]
        njcr = col["net_job_creation_rate"]
        rr = col["reallocation_rate"]
        jdr = col["job_destruction_rate"]
        jcr = col["job_creation_rate"]

        for cc in [njcr, rr, jdr, jcr]:
            if cc and cc in chunk.columns:
                chunk[cc] = pd.to_numeric(chunk[cc], errors="coerce")

        if st and njcr and st in chunk.columns and njcr in chunk.columns:
            g1 = chunk[[st]].copy()
            g1["x"] = chunk[njcr]
            agg1 = g1.groupby(st).agg({"x": ["sum", "count"]})
            agg1.columns = ["sum", "cnt"]
            for s, row in agg1.iterrows():
                avg_njcr[str(s)]["sum"] += float(row["sum"])
                avg_njcr[str(s)]["cnt"] += int(row["cnt"])

        if st and rr and st in chunk.columns and rr in chunk.columns:
            for s, v in zip(chunk[st], chunk[rr]):
                if pd.notna(v):
                    spread_rr[str(s)].update(float(v))

        if (
            st and y and jdr and
            st in chunk.columns and y in chunk.columns and jdr in chunk.columns
        ):
            g3 = chunk[[st, y]].copy()
            g3["v"] = chunk[jdr]
            agg3 = g3.groupby([st, y]).mean()["v"]
            for (state, year), val in agg3.items():
                ts_jdr[str(state)].append((int(year), float(val)))

        if jcr and jdr and jcr in chunk.columns and jdr in chunk.columns:
            corr_x.extend(pd.to_numeric(chunk[jcr], errors="coerce").fillna(0).tolist())
            corr_y.extend(pd.to_numeric(chunk[jdr], errors="coerce").fillna(0).tolist())

    avg_njcr_vals = {
        s: (v["sum"] / v["cnt"]) if v["cnt"] else float("nan")
        for s, v in avg_njcr.items()
    }
    avg_sorted = sorted(
        [(s, v) for s, v in avg_njcr_vals.items() if not pd.isna(v)],
        key=lambda x: x[1],
    )
    save_bar(
        [s for s, _ in avg_sorted],
        [v for _, v in avg_sorted],
        title="BusinessDynamics: Средний Net Job Creation Rate по штатам",
        ylabel="Rate",
        output_path=output_dir / "bd_avg_njcr.png",
        rotation=90,
    )

    spread_sorted = sorted(
        [(s, w.std) for s, w in spread_rr.items() if w.count > 1],
        key=lambda x: x[1],
    )
    save_bar(
        [s for s, _ in spread_sorted],
        [v for _, v in spread_sorted],
        title="BusinessDynamics: Разброс Reallocation Rate",
        ylabel="Std",
        output_path=output_dir / "bd_spread_rr.png",
        rotation=90,
    )

    # Наиболее нестабильный штат по Reallocation Rate
    unstable_state = spread_sorted[-1][0] if spread_sorted else None
    if unstable_state and unstable_state in ts_jdr:
        xs = sorted(ts_jdr[unstable_state])
        labels = [str(y) for y, _ in xs]
        vals = [v for _, v in xs]
        ma = moving_average(pd.Series(vals), window=3)
        save_line(
            labels,
            vals,
            title=f"BusinessDynamics: Job Destruction Rate — {unstable_state}",
            ylabel="Rate",
            output_path=output_dir / "bd_jdr_unstable.png",
        )
        save_line(
            labels,
            ma.tolist(),
            title=f"BusinessDynamics: JDR MA — {unstable_state}",
            ylabel="Rate (MA)",
            output_path=output_dir / "bd_jdr_unstable_ma.png",
        )

    corr = (
        float(pd.Series(corr_x).corr(pd.Series(corr_y)))
        if corr_x and corr_y
        else float("nan")
    )
    save_scatter(
        corr_x,
        corr_y,
        title=f"BusinessDynamics: JCR vs JDR (r={corr:.3f})",
        xlabel="Job Creation Rate",
        ylabel="Job Destruction Rate",
        output_path=output_dir / "bd_corr.png",
    )

    summary = [
        "Штаты с минимальным Net Job Creation Rate: "
        + ", ".join([f"{s}:{v:.3f}" for s, v in avg_sorted[:3]]),
        "Штаты с максимальным Net Job Creation Rate: "
        + ", ".join([f"{s}:{v:.3f}" for s, v in avg_sorted[-3:][::-1]]),
        "Наиболее стабильные штаты (минимальный Std): "
        + ", ".join([f"{s}:{v:.3f}" for s, v in spread_sorted[:3]]),
        "Наиболее турбулентные штаты (максимальный Std): "
        + ", ".join([f"{s}:{v:.3f}" for s, v in spread_sorted[-3:][::-1]]),
        f"Наиболее нестабильный штат (по RR): {unstable_state}",
        f"Корреляция JCR vs JDR: {corr:.3f}",
    ]
    (output_dir / "bd_summary.txt").write_text("\n".join(summary), encoding="utf-8")
