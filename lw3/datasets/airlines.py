from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from lw3.pipeline import read_csv_chunks
from lw3.plots import save_bar, save_line, save_scatter
from lw3.utils import Welford, ci95_mean, ensure_dir, find_first_match, moving_average


def _resolve_columns(cols: Iterable[str]) -> Dict[str, str]:
    cols = list(cols)
    mapping = {
        "total": find_first_match(cols, ["flights", "total"]) or find_first_match(cols, ["total", "flights"]),
        "delayed": find_first_match(cols, ["flights", "delay"]) or find_first_match(cols, ["delayed", "flights"]),
        "cancelled": find_first_match(cols, ["flights", "cancel"]) or find_first_match(cols, ["cancelled", "flights"]),
        "month": find_first_match(cols, ["time", "month"]) or find_first_match(cols, ["month"]),
        "year": find_first_match(cols, ["time", "year"]) or find_first_match(cols, ["year"]),
        "airport": find_first_match(cols, ["airport", "code"]) or find_first_match(cols, ["airport", "name"]) or find_first_match(cols, ["airport"]),
    }
    return mapping


def run_all(csv_path: Path | str, parquet_path: Path | str, output_dir: Path | str):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Пайплайн чтения
    first_chunk = next(read_csv_chunks(csv_path, chunksize=50_000))
    col = _resolve_columns(first_chunk.columns)

    # Инициализация накопителей
    month_sums = defaultdict(lambda: {"del": 0, "can": 0, "tot": 0})
    airport_var = defaultdict(Welford)  # разброс по времени для каждого аэропорта
    airport_totals = defaultdict(int)
    busiest_ts = defaultdict(int)  # временной ряд для каждого аэропорта по (year, month)
    monthly_year_rates = defaultdict(list)  # для CI по месяцам
    corr_x = []  # delayed+cancelled
    corr_y = []  # total

    # Обработка всех чанков
    for chunk in read_csv_chunks(csv_path, chunksize=200_000):
        dcol = col["delayed"]
        ccol = col["cancelled"]
        tcol = col["total"]
        mcol = col["month"]
        ycol = col["year"]
        acol = col["airport"]

        # Безопасное преобразование типов
        for cc in [dcol, ccol, tcol]:
            if cc and cc in chunk.columns:
                chunk[cc] = pd.to_numeric(chunk[cc], errors="coerce")

        # Метрики
        delayed = chunk[dcol] if dcol in chunk.columns else 0
        cancelled = chunk[ccol] if ccol in chunk.columns else 0
        total = chunk[tcol] if tcol in chunk.columns else 0
        safe_total = total.replace(0, np.nan)
        rate = (delayed + cancelled) / safe_total

        # Task 1: агрегация по месяцам
        if mcol in chunk.columns:
            g = chunk[[mcol]].copy()
            g["del"] = (delayed + cancelled).fillna(0)
            g["tot"] = total.fillna(0)
            agg = g.groupby(mcol).sum()
            for month, row in agg.iterrows():
                month_sums[int(month)]["del"] += int(row["del"])
                month_sums[int(month)]["tot"] += int(row["tot"])

        # Для CI по месяцам: собираем по годам-месяцам
        if mcol in chunk.columns and ycol in chunk.columns:
            g2 = chunk[[mcol, ycol]].copy()
            g2["del"] = (delayed + cancelled).fillna(0)
            g2["tot"] = total.fillna(0)
            agg2 = g2.groupby([ycol, mcol]).sum()
            for (year, month), row in agg2.iterrows():
                rt = (row["del"] / row["tot"]) if row["tot"] else np.nan
                if not pd.isna(rt):
                    monthly_year_rates[int(month)].append(float(rt))

        # Task 2: разброс по аэропортам
        if acol in chunk.columns:
            r = rate.fillna(np.nan)
            for airport, rt in zip(chunk[acol], r):
                if not pd.isna(rt):
                    airport_var[str(airport)].update(float(rt))

        # Task 3: суммы по аэропортам + временной ряд
        if acol in chunk.columns:
            g3 = chunk[[acol]].copy()
            g3["tot"] = total.fillna(0)
            agg3 = g3.groupby(acol).sum()["tot"]
            for ap, s in agg3.items():
                airport_totals[str(ap)] += int(s)

            if ycol in chunk.columns and mcol in chunk.columns:
                g4 = chunk[[acol, ycol, mcol]].copy()
                g4["tot"] = total.fillna(0)
                agg4 = g4.groupby([acol, ycol, mcol]).sum()["tot"]
                for (ap, year, month), s in agg4.items():
                    key = (str(ap), int(year), int(month))
                    busiest_ts[key] += int(s)

        # Extra: корреляция между задержанными/отмененными и общим числом
        corr_x.extend(((delayed + cancelled).fillna(0)).tolist())
        corr_y.extend((total.fillna(0)).tolist())

    # Итоги Task 1: лучшие/худшие месяцы
    months = sorted(month_sums.keys())
    month_rates = {m: (v["del"] / v["tot"]) if v["tot"] else 0.0 for m, v in month_sums.items()}
    # CI по месяцам
    ci_low = []
    ci_high = []
    values = []
    labels = [str(m) for m in months]
    for m in months:
        values.append(month_rates[m])
        mean, low, high, _ = ci95_mean(monthly_year_rates.get(m, []))
        ci_low.append(low)
        ci_high.append(high)
    save_bar(labels, values, title="Airlines: Доля задержанных/отмененных по месяцам", ylabel="Доля", output_path=output_dir / "airlines_month_rates.png", ci_low=ci_low, ci_high=ci_high)

    best3 = sorted(month_rates.items(), key=lambda x: x[1])[:3]
    worst3 = sorted(month_rates.items(), key=lambda x: x[1], reverse=True)[:3]

    # Итоги Task 2: аэропорты по разбросу
    var_items = [(ap, w.std) for ap, w in airport_var.items() if w.count > 1]
    var_sorted = sorted(var_items, key=lambda x: (float("inf") if pd.isna(x[1]) else x[1]))
    airports_low3 = var_sorted[:3]
    airports_high3 = sorted(var_items, key=lambda x: (float("-inf") if pd.isna(x[1]) else x[1]), reverse=True)[:3]
    save_bar([a for a, _ in var_items], [float(0 if pd.isna(v) else v) for _, v in var_items], title="Airlines: Разброс доли задержанных/отмененных по аэропортам", ylabel="Std", output_path=output_dir / "airlines_airport_spread.png", rotation=90)

    # Итоги Task 3: самый загруженный аэропорт и его динамика
    busiest_airport = max(airport_totals.items(), key=lambda x: x[1])[0] if airport_totals else None
    if busiest_airport:
        # собрать ряд по годам-месяцам
        ts = {(y, m): s for (ap, y, m), s in busiest_ts.items() if ap == busiest_airport}
        keys = sorted(ts.keys())
        labels_ts = [f"{y}-{m:02d}" for (y, m) in keys]
        values_ts = [ts[k] for k in keys]
        ma = moving_average(pd.Series(values_ts), window=3)
        save_line(labels_ts, values_ts, title=f"Airlines: Потоки для {busiest_airport}", ylabel="Рейсы", output_path=output_dir / "airlines_busiest_airport.png")
        save_line(labels_ts, ma.tolist(), title=f"Airlines: Скользящее среднее для {busiest_airport}", ylabel="Рейсы (MA)", output_path=output_dir / "airlines_busiest_airport_ma.png")

    # Extra: корреляция и scatter
    corr = float(pd.Series(corr_x).corr(pd.Series(corr_y))) if corr_x and corr_y else float("nan")
    save_scatter(corr_x, corr_y, title=f"Airlines: Корреляция задержанных+отмененных vs всего (r={corr:.3f})", xlabel="Задерж+Отмен", ylabel="Всего", output_path=output_dir / "airlines_corr.png")

    # Сводка в текстовый файл
    summary = [
        "Лучшие месяцы (минимальная доля): " + ", ".join([f"{m}:{rate:.3f}" for m, rate in best3]),
        "Худшие месяцы (максимальная доля): " + ", ".join([f"{m}:{rate:.3f}" for m, rate in worst3]),
        "Наименьший разброс (Std): " + ", ".join([f"{ap}:{std:.3f}" for ap, std in airports_low3]),
        "Наибольший разброс (Std): " + ", ".join([f"{ap}:{std:.3f}" for ap, std in airports_high3]),
        f"Самый загруженный аэропорт: {busiest_airport}",
        f"Корреляция задержанные+отмененные vs всего: {corr:.3f}",
    ]
    (output_dir / "airlines_summary.txt").write_text("\n".join(summary), encoding="utf-8")