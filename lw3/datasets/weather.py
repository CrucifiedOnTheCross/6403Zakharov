from __future__ import annotations
"""Анализ Weather: температуры по локациям, разброс по штатам и ветер."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from lw3.pipeline import read_csv_chunks
from lw3.plots import save_bar, save_line, save_scatter
from lw3.utils import Welford, ensure_dir, find_first_match, moving_average


def _resolve_columns(cols: Iterable[str]) -> Dict[str, str]:
    """Определить столбцы для Weather по ключевым словам."""
    cols = list(cols)
    return {
        "temp_avg": find_first_match(cols, ["temp", "avg"]) or find_first_match(cols, ["temperature", "average"]) or find_first_match(cols, ["avg", "temperature"]),
        "location": find_first_match(cols, ["location"]) or find_first_match(cols, ["city"]) or find_first_match(cols, ["station"]),
        "state": find_first_match(cols, ["state"]) or find_first_match(cols, ["region"]),
        "year": find_first_match(cols, ["year"]) or find_first_match(cols, ["date", "year"]),
        "month": find_first_match(cols, ["month"]) or find_first_match(cols, ["date", "month"]),
        "wind": find_first_match(cols, ["wind", "speed"]) or find_first_match(cols, ["wind", "spd"]) or find_first_match(cols, ["wind"]),
        "prec": find_first_match(cols, ["precip"]) or find_first_match(cols, ["precipitation"]) or find_first_match(cols, ["rain"]),
    }


def run_all(csv_path: Path | str, parquet_path: Path | str, output_dir: Path | str):
    """Запустить анализ Weather: температура по локациям, разброс по штатам, ветер."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    first = next(read_csv_chunks(csv_path, chunksize=50_000))
    col = _resolve_columns(first.columns)

    # Task 1: среднегодовая температура по локациям
    sum_cnt_loc = defaultdict(lambda: {"sum": 0.0, "cnt": 0})

    # Task 2: разброс среднемесячных температур по штатам
    welford_state = defaultdict(Welford)

    # Task 3: динамика скорости ветра в самом «ветренном» штате
    wind_state_sum = defaultdict(lambda: {"sum": 0.0, "cnt": 0})
    wind_ts = defaultdict(float)  # (state, year, month) -> wind avg

    # Extra: корреляция Wind.Speed vs Precipitation
    corr_x = []
    corr_y = []

    for chunk in read_csv_chunks(csv_path, chunksize=200_000):
        tavg = col["temp_avg"]
        loc = col["location"]
        st = col["state"]
        y = col["year"]
        m = col["month"]
        wind = col["wind"]
        prec = col["prec"]

        # приведение типов
        for cc in [tavg, wind, prec]:
            if cc and cc in chunk.columns:
                chunk[cc] = pd.to_numeric(chunk[cc], errors="coerce")

        # Task 1
        if loc and tavg and loc in chunk.columns and tavg in chunk.columns:
            g1 = chunk[[loc]].copy()
            g1["tavg"] = chunk[tavg]
            agg = g1.groupby(loc).agg({"tavg": ["sum", "count"]})
            agg.columns = ["sum", "cnt"]
            for loc_name, row in agg.iterrows():
                sum_cnt_loc[str(loc_name)]["sum"] += float(row["sum"])
                sum_cnt_loc[str(loc_name)]["cnt"] += int(row["cnt"])

        # Task 2: разброс среднемесячных температур по штатам
        if st and tavg and st in chunk.columns and tavg in chunk.columns:
            for s, v in zip(chunk[st], chunk[tavg]):
                if not pd.isna(v):
                    welford_state[str(s)].update(float(v))

        # Task 3: динамика ветра
        if st and wind and st in chunk.columns and wind in chunk.columns:
            g3 = chunk[[st]].copy()
            g3["wind"] = chunk[wind]
            agg3 = g3.groupby(st).agg({"wind": ["sum", "count"]})
            agg3.columns = ["sum", "cnt"]
            for s, row in agg3.iterrows():
                wind_state_sum[str(s)]["sum"] += float(row["sum"])
                wind_state_sum[str(s)]["cnt"] += int(row["cnt"])

            if y and m and y in chunk.columns and m in chunk.columns:
                g4 = chunk[[st, y, m]].copy()
                g4["wind"] = chunk[wind]
                agg4 = g4.groupby([st, y, m]).mean()["wind"]
                for (state, year, month), val in agg4.items():
                    wind_ts[(str(state), int(year), int(month))] = float(val)

        # Extra: корреляция
        if wind and prec and wind in chunk.columns and prec in chunk.columns:
            corr_x.extend(pd.to_numeric(chunk[wind], errors="coerce").fillna(0).tolist())
            corr_y.extend(pd.to_numeric(chunk[prec], errors="coerce").fillna(0).tolist())

    # Итоги Task 1
    loc_avg = {loc: (v["sum"] / v["cnt"]) if v["cnt"] else np.nan for loc, v in sum_cnt_loc.items()}
    loc_sorted = sorted([(loc, a) for loc, a in loc_avg.items() if not pd.isna(a)], key=lambda x: x[1])
    low3 = loc_sorted[:3]
    high3 = loc_sorted[-3:][::-1]
    save_bar([loc for loc, _ in loc_sorted], [v for _, v in loc_sorted], title="Weather: Средняя температура по локациям", ylabel="Температура", output_path=output_dir / "weather_location_avg_temp.png", rotation=90)

    # Итоги Task 2
    state_std = [(s, w.std) for s, w in welford_state.items() if w.count > 1]
    state_std_sorted = sorted(state_std, key=lambda x: x[1])
    low3_std = state_std_sorted[:3]
    high3_std = state_std_sorted[-3:][::-1]
    save_bar([s for s, _ in state_std_sorted], [v for _, v in state_std_sorted], title="Weather: Разброс среднемесячной температуры по штатам", ylabel="Std", output_path=output_dir / "weather_state_temp_spread.png", rotation=90)

    # Итоги Task 3
    wind_state_avg = {s: (v["sum"] / v["cnt"]) if v["cnt"] else np.nan for s, v in wind_state_sum.items()}
    windy_state = max([(s, a) for s, a in wind_state_avg.items() if not pd.isna(a)], key=lambda x: x[1])[0] if wind_state_avg else None
    if windy_state:
        ts = {(y, m): val for (s, y, m), val in wind_ts.items() if s == windy_state}
        keys = sorted(ts.keys())
        labels = [f"{y}-{m:02d}" for (y, m) in keys]
        vals = [ts[k] for k in keys]
        ma = moving_average(pd.Series(vals), window=3)
        save_line(labels, vals, title=f"Weather: Ветер в штате {windy_state}", ylabel="Скорость ветра", output_path=output_dir / "weather_windy_state.png")
        save_line(labels, ma.tolist(), title=f"Weather: Скользящее среднее ветра {windy_state}", ylabel="Скорость ветра (MA)", output_path=output_dir / "weather_windy_state_ma.png")

    # Extra
    corr = float(pd.Series(corr_x).corr(pd.Series(corr_y))) if corr_x and corr_y else float("nan")
    save_scatter(corr_x, corr_y, title=f"Weather: Wind vs Precip (r={corr:.3f})", xlabel="Wind.Speed", ylabel="Precipitation", output_path=output_dir / "weather_corr.png")
    summary = [
        "Самые холодные локации: " + ", ".join([f"{loc}:{v:.2f}" for loc, v in low3]),
        "Самые теплые локации: " + ", ".join([f"{loc}:{v:.2f}" for loc, v in high3]),
        "Минимальный разброс температур: " + ", ".join([f"{s}:{v:.3f}" for s, v in low3_std]),
        "Максимальный разброс температур: " + ", ".join([f"{s}:{v:.3f}" for s, v in high3_std]),
        f"Самый ветренный штат: {windy_state}",
        f"Корреляция Wind vs Precip: {corr:.3f}",
    ]
    (output_dir / "weather_summary.txt").write_text("\n".join(summary), encoding="utf-8")