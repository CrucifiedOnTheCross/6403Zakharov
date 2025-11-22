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

    needed_cols = list(COLUMNS.values())
    
    # Аккумуляторы для статистики
    state_stats = defaultdict(lambda: {"njcr_sum": 0.0, "njcr_cnt": 0})
    welford_njcr = defaultdict(Welford)
    welford_rr = defaultdict(Welford)
    ts_jdr_parts = []  # Будем накапливать части, потом concat
    corr_parts = []    # Аналогично для корреляции

    # Читаем CSV чанками
    for chunk in read_csv_chunks(csv_path, chunksize=50000, usecols=needed_cols):
        # Конвертируем все числовые колонки сразу
        numeric_cols = [COLUMNS[k] for k in ["njcr", "rr", "jdr", "jcr"]]
        chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors="coerce")
        
        st = COLUMNS["state"]
        yr = COLUMNS["year"]
        njcr = COLUMNS["njcr"]
        rr = COLUMNS["rr"]
        jdr = COLUMNS["jdr"]
        jcr = COLUMNS["jcr"]

        # === Task 1: Средний NJCR ===
        # Агрегация через pandas
        njcr_agg = chunk.groupby(st)[njcr].agg(["sum", "count"]).reset_index()
        njcr_agg.columns = [st, "sum", "count"]
        
        # Обновление аккумуляторов через itertuples (быстрее iterrows)
        for row in njcr_agg.itertuples(index=False):
            state_stats[row[0]]["njcr_sum"] += row[1]
            state_stats[row[0]]["njcr_cnt"] += row[2]
        
        # Welford для std
        njcr_valid = chunk[[st, njcr]].dropna()
        for state, val in njcr_valid.itertuples(index=False):
            welford_njcr[state].update(val)

        # === Task 2: Разброс RR ===
        rr_valid = chunk[[st, rr]].dropna()
        for state, val in rr_valid.itertuples(index=False):
            welford_rr[state].update(val)

        # === Task 3: Динамика JDR ===
        # Сохраняем агрегированные части вместо списков
        jdr_agg = chunk.groupby([st, yr])[jdr].mean().reset_index()
        jdr_agg.columns = [st, yr, "jdr_mean"]
        ts_jdr_parts.append(jdr_agg)

        # === Extra: Корреляция ===
        # Собираем валидные пары в DataFrame
        corr_valid = chunk[[jcr, jdr]].dropna()
        corr_parts.append(corr_valid)

    # ============== Обработка результатов ==============
    
    # Task 1: Средние NJCR
    avg_njcr = {
        s: stats["njcr_sum"] / stats["njcr_cnt"] if stats["njcr_cnt"] > 0 else np.nan
        for s, stats in state_stats.items()
    }
    avg_sorted = sorted(
        [(s, v) for s, v in avg_njcr.items() if not pd.isna(v)],
        key=lambda x: x[1]
    )

    # Топ-3 и Bottom-3
    top3 = avg_sorted[-3:][::-1]
    bottom3 = avg_sorted[:3]
    labels = [s for s, _ in top3 + bottom3]
    means = [avg_njcr[s] for s in labels]
    
    # Доверительные интервалы (95%)
    z = 1.96
    ci_data = [
        {
            "mean": avg_njcr[s],
            "std": welford_njcr[s].std,
            "cnt": welford_njcr[s].count
        }
        for s in labels
    ]
    se = [d["std"] / np.sqrt(d["cnt"]) if d["cnt"] > 1 else np.nan for d in ci_data]
    ci_low = [m - z * s for m, s in zip(means, se)]
    ci_high = [m + z * s for m, s in zip(means, se)]
    
    save_bar(
        labels, means,
        title="BusinessDynamics: Топ/Боттом NJCR (95% CI)",
        ylabel="Rate",
        output_path=output_dir / "bd_njcr_top_bottom.png",
        rotation=90,
        ci_low=ci_low,
        ci_high=ci_high,
    )

    # Task 2: Разброс RR
    rr_stats = [(s, w.std, w.mean) for s, w in welford_rr.items() if w.count > 1]
    rr_sorted = sorted(rr_stats, key=lambda x: x[1])
    
    rr_top3 = rr_sorted[-3:][::-1]
    rr_bottom3 = rr_sorted[:3]
    rr_labels = [s for s, _, _ in rr_top3 + rr_bottom3]
    rr_means = [welford_rr[s].mean for s in rr_labels]
    
    rr_ci_data = [
        {
            "std": welford_rr[s].std,
            "cnt": welford_rr[s].count
        }
        for s in rr_labels
    ]
    rr_se = [d["std"] / np.sqrt(d["cnt"]) if d["cnt"] > 1 else np.nan for d in rr_ci_data]
    rr_ci_low = [m - z * s for m, s in zip(rr_means, rr_se)]
    rr_ci_high = [m + z * s for m, s in zip(rr_means, rr_se)]
    
    save_bar(
        rr_labels, rr_means,
        title="BusinessDynamics: Топ/Боттом RR (95% CI)",
        ylabel="Rate",
        output_path=output_dir / "bd_rr_top_bottom_ci.png",
        rotation=90,
        ci_low=rr_ci_low,
        ci_high=rr_ci_high,
    )

    # Task 3: Динамика JDR для нестабильного штата
    unstable_state = rr_sorted[-1][0] if rr_sorted else None
    
    if unstable_state and ts_jdr_parts:
        # Объединяем все части через concat
        ts_jdr_df = pd.concat(ts_jdr_parts, ignore_index=True)
        
        # Фильтруем по штату и агрегируем (на случай дубликатов из разных чанков)
        state_ts = ts_jdr_df[ts_jdr_df[COLUMNS["state"]] == unstable_state]
        state_ts = state_ts.groupby(COLUMNS["year"])["jdr_mean"].mean().reset_index()
        state_ts = state_ts.sort_values(COLUMNS["year"])
        
        labels_jdr = state_ts[COLUMNS["year"]].astype(str).tolist()
        vals_jdr = state_ts["jdr_mean"].tolist()
        ma = moving_average(pd.Series(vals_jdr), window=3)
        
        save_line_dual(
            labels_jdr, vals_jdr, ma.tolist(),
            title=f"BusinessDynamics: JDR + MA — {unstable_state}",
            ylabel="Rate",
            output_path=output_dir / "bd_jdr_unstable.png",
            legend1="JDR",
            legend2="MA(3)",
        )

    # Extra: Корреляция
    if corr_parts:
        corr_df = pd.concat(corr_parts, ignore_index=True)
        corr_value = corr_df[COLUMNS["jcr"]].corr(corr_df[COLUMNS["jdr"]])
        
        save_scatter(
            corr_df[COLUMNS["jcr"]].tolist(),
            corr_df[COLUMNS["jdr"]].tolist(),
            title=f"BusinessDynamics: JCR vs JDR (r={corr_value:.3f})",
            xlabel="Job Creation Rate",
            ylabel="Job Destruction Rate",
            output_path=output_dir / "bd_corr.png",
        )
    else:
        corr_value = np.nan

    # Summary
    summary = [
        "Штаты с минимальным Net Job Creation Rate: "
        + ", ".join([f"{s}:{v:.3f}" for s, v in avg_sorted[:3]]),
        "Штаты с максимальным Net Job Creation Rate: "
        + ", ".join([f"{s}:{v:.3f}" for s, v in avg_sorted[-3:][::-1]]),
        "Наиболее стабильные штаты (минимальный Std): "
        + ", ".join([f"{s}:{std:.3f}" for s, std, _ in rr_sorted[:3]]),
        "Наиболее турбулентные штаты (максимальный Std): "
        + ", ".join([f"{s}:{std:.3f}" for s, std, _ in rr_sorted[-3:][::-1]]),
        f"Наиболее нестабильный штат (по RR): {unstable_state}",
        f"Корреляция JCR vs JDR: {corr_value:.3f}",
    ]
    (output_dir / "bd_summary.txt").write_text("\n".join(summary), encoding="utf-8")