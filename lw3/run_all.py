from __future__ import annotations

from pathlib import Path

from lw3.datasets.business_dynamics import run_all as run_bd
from lw3.parquet_utils import ensure_parquet, compare_read_speed
from lw3.plots import save_scatter
import pandas as pd


BASE = Path(__file__).resolve().parent
CACHE = BASE / "parquet"
DATA = BASE / "csv"
OUT = BASE / "output"


def main():
    """Запустить полный пайплайн: подготовка Parquet, анализ и доп. задания."""
    OUT.mkdir(parents=True, exist_ok=True)

    # Обеспечить parquet для каждого датасета
    files = {
        "business_dynamics": DATA / "business_dynamics.csv",
    }

    for name, csv in files.items():
        ensure_parquet(csv, CACHE / f"{name}.parquet")
        t_csv, t_parquet = compare_read_speed(csv, CACHE / f"{name}.parquet")
        (OUT / f"perf_{name}.txt").write_text(
            f"CSV: {t_csv:.4f}s\nParquet: {t_parquet:.4f}s\n",
            encoding="utf-8",
        )

    # Доп. задания: считать только релевантные столбцы из Parquet и построить scatter
    def parquet_scatter_bd(csv_path: Path, parquet_path: Path, out_path: Path):
        """Построить scatter JCR vs JDR из Parquet (только нужные столбцы)."""
        xcol = "Data.Job Creation.Rate"
        ycol = "Data.Job Destruction.Rate"
        df = pd.read_parquet(parquet_path, columns=[xcol, ycol])
        x = pd.to_numeric(df[xcol], errors="coerce").fillna(0)
        y = pd.to_numeric(df[ycol], errors="coerce").fillna(0)
        corr = float(x.corr(y)) if len(x) and len(y) else float("nan")
        save_scatter(
            x.tolist(),
            y.tolist(),
            title=f"BusinessDynamics Parquet: JCR vs JDR (r={corr:.3f})",
            xlabel="Job Creation Rate",
            ylabel="Job Destruction Rate",
            output_path=out_path / "bd_corr_parquet.png",
        )

    # Запустить задачи для каждого датасета (основные + графики + дополнительные)
    run_bd(
        files["business_dynamics"],
        CACHE / "business_dynamics.parquet",
        OUT / "business_dynamics",
    )

    parquet_scatter_bd(
        files["business_dynamics"],
        CACHE / "business_dynamics.parquet",
        OUT,
    )


if __name__ == "__main__":
    main()
