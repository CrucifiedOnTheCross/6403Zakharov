from __future__ import annotations
"""Анализ Business Dynamics: NJCR, RR, динамика JDR и корреляции."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable
import numpy as np

import pandas as pd

from lw3.pipeline import read_csv_chunks
from lw3.plots import save_bar, save_line_dual, save_scatter
from lw3.utils import Welford, ensure_dir, moving_average


def _resolve_columns(cols: Iterable[str]) -> Dict[str, str]:
    """Определить столбцы по известным точным именам из CORGIS."""
    return {
        "state": "State",
        "year": "Year",
        "net_job_creation_rate": "Data.Calculated.Net Job Creation Rate",
        "reallocation_rate": "Data.Calculated.Reallocation Rate",
        "job_destruction_rate": "Data.Job Destruction.Rate",
        "job_creation_rate": "Data.Job Creation.Rate",
    }


def run_all(csv_path: Path | str, parquet_path: Path | str, output_dir: Path | str):
    """Запустить анализ датасета Business Dynamics и сохранить результаты.

    Считает средний Net Job Creation Rate, разброс Reallocation Rate,
    строит динамику JDR для наиболее нестабильного штата и вычисляет корреляции.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    first = next(read_csv_chunks(csv_path, chunksize=1000))
    col = _resolve_columns(first.columns)

    # Task 1: средний темп создания рабочих мест по штатам
    avg_njcr = defaultdict(lambda: {"sum": 0.0, "cnt": 0})
    welford_njcr = defaultdict(Welford)

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
            for s, v in zip(chunk[st], chunk[njcr]):
                if pd.notna(v):
                    welford_njcr[str(s)].update(float(v))

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
    # Топ-3 и боттом-3 по среднему NJCR
    njcr_bottom3 = avg_sorted[:3]
    njcr_top3 = avg_sorted[-3:][::-1]
    njcr_labels = [s for s, _ in (njcr_top3 + njcr_bottom3)]
    njcr_means = [avg_njcr_vals[s] for s in njcr_labels]
    z = 1.96
    means_arr = np.asarray(njcr_means, dtype=float)
    stds_arr = np.asarray([
        (welford_njcr.get(s).std if welford_njcr.get(s) else np.nan)
        for s in njcr_labels
    ], dtype=float)
    cnts_arr = np.asarray([
        (welford_njcr.get(s).count if welford_njcr.get(s) else 0)
        for s in njcr_labels
    ], dtype=int)
    mask = (cnts_arr > 1) & (~np.isnan(stds_arr))
    se = np.full_like(stds_arr, np.nan, dtype=float)
    se[mask] = stds_arr[mask] / np.sqrt(cnts_arr[mask])
    njcr_ci_low = (means_arr - z * se).tolist()
    njcr_ci_high = (means_arr + z * se).tolist()
    save_bar(
        njcr_labels,
        njcr_means,
        title="BusinessDynamics: Топ/Боттом NJCR (95% CI)",
        ylabel="Rate",
        output_path=output_dir / "bd_njcr_top_bottom.png",
        rotation=90,
        ci_low=njcr_ci_low,
        ci_high=njcr_ci_high,
    )

    spread_sorted = sorted(
        [(s, w.std) for s, w in spread_rr.items() if w.count > 1],
        key=lambda x: x[1],
    )
    # Топ-3 стабильные (минимальный Std) и топ-3 турбулентные (максимальный Std)
    rr_bottom3 = spread_sorted[:3]
    rr_top3 = spread_sorted[-3:][::-1]
    rr_labels = [s for s, _ in (rr_top3 + rr_bottom3)]
    rr_means = [spread_rr[s].mean for s in rr_labels]
    means_arr_rr = np.asarray(rr_means, dtype=float)
    stds_arr_rr = np.asarray([
        (spread_rr.get(s).std if spread_rr.get(s) else np.nan)
        for s in rr_labels
    ], dtype=float)
    cnts_arr_rr = np.asarray([
        (spread_rr.get(s).count if spread_rr.get(s) else 0)
        for s in rr_labels
    ], dtype=int)
    mask_rr = (cnts_arr_rr > 1) & (~np.isnan(stds_arr_rr))
    se_rr = np.full_like(stds_arr_rr, np.nan, dtype=float)
    se_rr[mask_rr] = stds_arr_rr[mask_rr] / np.sqrt(cnts_arr_rr[mask_rr])
    rr_ci_low = (means_arr_rr - z * se_rr).tolist()
    rr_ci_high = (means_arr_rr + z * se_rr).tolist()
    save_bar(
        rr_labels,
        rr_means,
        title="BusinessDynamics: Топ/Боттом RR (95% CI)",
        ylabel="Rate",
        output_path=output_dir / "bd_rr_top_bottom_ci.png",
        rotation=90,
        ci_low=rr_ci_low,
        ci_high=rr_ci_high,
    )

    # Наиболее нестабильный штат по Reallocation Rate
    unstable_state = spread_sorted[-1][0] if spread_sorted else None
    if unstable_state and unstable_state in ts_jdr:
        xs = sorted(ts_jdr[unstable_state])
        labels = [str(y) for y, _ in xs]
        vals = [v for _, v in xs]
        ma = moving_average(pd.Series(vals), window=3)
        save_line_dual(
            labels,
            vals,
            ma.tolist(),
            title=f"BusinessDynamics: JDR + MA — {unstable_state}",
            ylabel="Rate",
            output_path=output_dir / "bd_jdr_unstable.png",
            legend1="JDR",
            legend2="MA(3)",
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
