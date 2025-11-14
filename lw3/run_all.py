from __future__ import annotations

from pathlib import Path

from lw3.datasets.airlines import run_all as run_airlines
from lw3.datasets.weather import run_all as run_weather
from lw3.datasets.video_games import run_all as run_video_games
from lw3.datasets.global_emissions import run_all as run_emissions
from lw3.datasets.business_dynamics import run_all as run_bd
from lw3.parquet_utils import ensure_parquet, compare_read_speed
from lw3.utils import find_first_match
from lw3.plots import save_scatter
import pandas as pd


BASE = Path(__file__).resolve().parent
CACHE = BASE / "parquet"
DATA = BASE / "csv"
OUT = BASE / "output"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Обеспечить parquet для каждого датасета (используем только релевантные столбцы при необходимости)
    files = {
        "airlines": DATA / "airlines.csv",
        "weather": DATA / "weather.csv",
        "video_games": DATA / "video_games.csv",
        "global_emissions": DATA / "global_emissions.csv",
        "business_dynamics": DATA / "business_dynamics.csv",
    }

    for name, csv in files.items():
        ensure_parquet(csv, CACHE / f"{name}.parquet")
        t_csv, t_parquet = compare_read_speed(csv, CACHE / f"{name}.parquet")
        (OUT / f"perf_{name}.txt").write_text(f"CSV: {t_csv:.4f}s\nParquet: {t_parquet:.4f}s\n", encoding="utf-8")

    # Доп. задания: считать только релевантные столбцы из Parquet и построить scatter
    def parquet_scatter(dataset_name: str, csv_path: Path, parquet_path: Path, out_path: Path):
        # получить схему по CSV (только имена столбцов)
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if dataset_name == "airlines":
            xcol = find_first_match(cols, ["flights", "delay"]) or find_first_match(cols, ["delayed", "flights"])
            ycol = find_first_match(cols, ["flights", "total"]) or find_first_match(cols, ["total", "flights"])
            x2 = find_first_match(cols, ["flights", "cancel"]) or find_first_match(cols, ["cancelled", "flights"])
            columns = [c for c in [xcol, x2, ycol] if c]
            df = pd.read_parquet(parquet_path, columns=columns)
            x = pd.to_numeric(df.get(xcol, 0), errors="coerce").fillna(0) + pd.to_numeric(df.get(x2, 0), errors="coerce").fillna(0)
            y = pd.to_numeric(df.get(ycol, 0), errors="coerce").fillna(0)
            corr = float(x.corr(y)) if len(x) and len(y) else float("nan")
            save_scatter(x.tolist(), y.tolist(), title=f"Airlines Parquet: Delayed+Cancelled vs Total (r={corr:.3f})", xlabel="Delayed+Cancelled", ylabel="Total", output_path=out_path / "airlines_corr_parquet.png")
        elif dataset_name == "weather":
            xcol = find_first_match(cols, ["wind", "speed"]) or find_first_match(cols, ["wind"])
            ycol = find_first_match(cols, ["precip"]) or find_first_match(cols, ["precipitation"]) or find_first_match(cols, ["rain"])
            df = pd.read_parquet(parquet_path, columns=[c for c in [xcol, ycol] if c])
            x = pd.to_numeric(df.get(xcol, 0), errors="coerce").fillna(0)
            y = pd.to_numeric(df.get(ycol, 0), errors="coerce").fillna(0)
            corr = float(x.corr(y)) if len(x) and len(y) else float("nan")
            save_scatter(x.tolist(), y.tolist(), title=f"Weather Parquet: Wind vs Precip (r={corr:.3f})", xlabel="Wind.Speed", ylabel="Precipitation", output_path=out_path / "weather_corr_parquet.png")
        elif dataset_name == "video_games":
            xcol = find_first_match(cols, ["review", "score"]) or find_first_match(cols, ["critic", "score"]) or find_first_match(cols, ["rating", "review"])
            ycol = find_first_match(cols, ["sales"]) or find_first_match(cols, ["metrics", "sales"])
            df = pd.read_parquet(parquet_path, columns=[c for c in [xcol, ycol] if c])
            x = pd.to_numeric(df.get(xcol, 0), errors="coerce").fillna(0)
            y = pd.to_numeric(df.get(ycol, 0), errors="coerce").fillna(0)
            corr = float(x.corr(y)) if len(x) and len(y) else float("nan")
            save_scatter(x.tolist(), y.tolist(), title=f"VideoGames Parquet: Review vs Sales (r={corr:.3f})", xlabel="Review.Score", ylabel="Sales", output_path=out_path / "vg_corr_parquet.png")
        elif dataset_name == "global_emissions":
            xcol = find_first_match(cols, ["population"]) or find_first_match(cols, ["pop"]) 
            ycol = find_first_match(cols, ["emissions", "total"]) or find_first_match(cols, ["co2"]) or find_first_match(cols, ["ghg"]) 
            df = pd.read_parquet(parquet_path, columns=[c for c in [xcol, ycol] if c])
            x = pd.to_numeric(df.get(xcol, 0), errors="coerce").fillna(0)
            y = pd.to_numeric(df.get(ycol, 0), errors="coerce").fillna(0)
            corr = float(x.corr(y)) if len(x) and len(y) else float("nan")
            save_scatter(x.tolist(), y.tolist(), title=f"Emissions Parquet: Pop vs Emissions (r={corr:.3f})", xlabel="Population", ylabel="Emissions", output_path=out_path / "emissions_corr_parquet.png")
        elif dataset_name == "business_dynamics":
            xcol = find_first_match(cols, ["job", "creation", "rate"]) or find_first_match(cols, ["creation", "rate"]) 
            ycol = find_first_match(cols, ["job", "destruction", "rate"]) or find_first_match(cols, ["destruction", "rate"]) 
            df = pd.read_parquet(parquet_path, columns=[c for c in [xcol, ycol] if c])
            x = pd.to_numeric(df.get(xcol, 0), errors="coerce").fillna(0)
            y = pd.to_numeric(df.get(ycol, 0), errors="coerce").fillna(0)
            corr = float(x.corr(y)) if len(x) and len(y) else float("nan")
            save_scatter(x.tolist(), y.tolist(), title=f"BusinessDynamics Parquet: JCR vs JDR (r={corr:.3f})", xlabel="Job Creation Rate", ylabel="Job Destruction Rate", output_path=out_path / "bd_corr_parquet.png")

    # Запустить задачи для каждого датасета (основные + графики + дополнительные)
    run_airlines(files["airlines"], CACHE / "airlines.parquet", OUT / "airlines")
    run_weather(files["weather"], CACHE / "weather.parquet", OUT / "weather")
    run_video_games(files["video_games"], CACHE / "video_games.parquet", OUT / "video_games")
    run_emissions(files["global_emissions"], CACHE / "global_emissions.parquet", OUT / "emissions")
    run_bd(files["business_dynamics"], CACHE / "business_dynamics.parquet", OUT / "business_dynamics")

    parquet_scatter("airlines", files["airlines"], CACHE / "airlines.parquet", OUT)
    parquet_scatter("weather", files["weather"], CACHE / "weather.parquet", OUT)
    parquet_scatter("video_games", files["video_games"], CACHE / "video_games.parquet", OUT)
    parquet_scatter("global_emissions", files["global_emissions"], CACHE / "global_emissions.parquet", OUT)
    parquet_scatter("business_dynamics", files["business_dynamics"], CACHE / "business_dynamics.parquet", OUT)


if __name__ == "__main__":
    main()