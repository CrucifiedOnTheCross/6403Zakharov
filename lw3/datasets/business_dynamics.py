from __future__ import annotations
"""Анализ Business Dynamics: NJCR, RR, динамика JDR и корреляции."""

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from lw3.pipeline import read_csv_chunks
from lw3.plots import save_bar, save_line_dual, save_scatter
from lw3.utils import Welford, ensure_dir, moving_average


COLUMNS = {
    "state": "State",
    "year": "Year",
    "njcr": "Data.Calculated.Net Job Creation Rate",
    "rr": "Data.Calculated.Reallocation Rate",
    "jdr": "Data.Job Destruction.Rate",
    "jcr": "Data.Job Creation.Rate",
}


def run_all(csv_path: Path | str, parquet_path: Path | str, output_dir: Path | str):
    """Запустить анализ датасета Business Dynamics и сохранить результаты."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Задаём нужные колонки для чтения - экономим память и время
    needed_cols = list(COLUMNS.values())
    
    # Инициализация аккумуляторов
    avg_njcr = defaultdict(lambda: {"sum": 0.0, "cnt": 0})
    welford_njcr = defaultdict(Welford)
    spread_rr = defaultdict(Welford)
    ts_jdr = defaultdict(list)
    corr_x = []
    corr_y = []

    # Читаем большими чанками (100 -> 50000) - в 500 раз эффективнее!
    for chunk in read_csv_chunks(csv_path, chunksize=50000, usecols=needed_cols):
        # Конвертируем числовые колонки один раз
        numeric_cols = [COLUMNS["njcr"], COLUMNS["rr"], COLUMNS["jdr"], COLUMNS["jcr"]]
        chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors="coerce")
        
        st_col = COLUMNS["state"]
        yr_col = COLUMNS["year"]
        njcr_col = COLUMNS["njcr"]
        rr_col = COLUMNS["rr"]
        jdr_col = COLUMNS["jdr"]
        jcr_col = COLUMNS["jcr"]

        # Task 1: средний NJCR
        grouped = chunk.groupby(st_col)[njcr_col].agg(["sum", "count"])
        for state, row in grouped.iterrows():
            avg_njcr[state]["sum"] += row["sum"]
            avg_njcr[state]["cnt"] += row["count"]
        
        for state, val in chunk[[st_col, njcr_col]].dropna().values:
            welford_njcr[state].update(val)

        # Task 2: разброс RR
        for state, val in chunk[[st_col, rr_col]].dropna().values:
            spread_rr[state].update(val)

        # Task 3: динамика JDR
        jdr_grouped = chunk.groupby([st_col, yr_col])[jdr_col].mean()
        for (state, year), val in jdr_grouped.items():
            ts_jdr[state].append((year, val))

        # Extra: корреляция
        valid_mask = chunk[[jcr_col, jdr_col]].notna().all(axis=1)
        corr_x.extend(chunk.loc[valid_mask, jcr_col].tolist())
        corr_y.extend(chunk.loc[valid_mask, jdr_col].tolist())

    # Вычисление результатов и построение графиков
    avg_njcr_vals = {
        s: (v["sum"] / v["cnt"]) if v["cnt"] else float("nan")
        for s, v in avg_njcr.items()
    }
    avg_sorted = sorted(
        [(s, v) for s, v in avg_njcr_vals.items() if not pd.isna(v)],
        key=lambda x: x[1],
    )

    # Топ-3 и боттом-3 по NJCR
    njcr_top3 = avg_sorted[-3:][::-1]
    njcr_bottom3 = avg_sorted[:3]
    njcr_labels = [s for s, _ in (njcr_top3 + njcr_bottom3)]
    njcr_means = [avg_njcr_vals[s] for s in njcr_labels]
    
    # Доверительные интервалы
    z = 1.96
    stds = [welford_njcr[s].std for s in njcr_labels]
    cnts = [welford_njcr[s].count for s in njcr_labels]
    se = [std / np.sqrt(cnt) if cnt > 1 else np.nan for std, cnt in zip(stds, cnts)]
    njcr_ci_low = [m - z * s for m, s in zip(njcr_means, se)]
    njcr_ci_high = [m + z * s for m, s in zip(njcr_means, se)]
    
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

    # Топ-3 стабильные/турбулентные штаты
    spread_sorted = sorted(
        [(s, w.std) for s, w in spread_rr.items() if w.count > 1],
        key=lambda x: x[1],
    )
    rr_top3 = spread_sorted[-3:][::-1]
    rr_bottom3 = spread_sorted[:3]
    rr_labels = [s for s, _ in (rr_top3 + rr_bottom3)]
    rr_means = [spread_rr[s].mean for s in rr_labels]
    
    stds_rr = [spread_rr[s].std for s in rr_labels]
    cnts_rr = [spread_rr[s].count for s in rr_labels]
    se_rr = [std / np.sqrt(cnt) if cnt > 1 else np.nan for std, cnt in zip(stds_rr, cnts_rr)]
    rr_ci_low = [m - z * s for m, s in zip(rr_means, se_rr)]
    rr_ci_high = [m + z * s for m, s in zip(rr_means, se_rr)]
    
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

    # Динамика JDR для самого нестабильного штата
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

    # Корреляция
    corr = float(pd.Series(corr_x).corr(pd.Series(corr_y))) if corr_x else float("nan")
    save_scatter(
        corr_x,
        corr_y,
        title=f"BusinessDynamics: JCR vs JDR (r={corr:.3f})",
        xlabel="Job Creation Rate",
        ylabel="Job Destruction Rate",
        output_path=output_dir / "bd_corr.png",
    )

    # Summary
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