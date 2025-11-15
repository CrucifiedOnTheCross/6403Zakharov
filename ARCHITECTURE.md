# Архитектура (Вариант 5: Business Dynamics)

## Общее устройство
- Пакет `lw3` — анализ Business Dynamics: пайплайн чтения CSV чанками, визуализация и задачи варианта 5.
- Подсистемы:
  - `lw3/pipeline.py` — генератор `read_csv_chunks` для потокового чтения.
  - `lw3/utils.py` — поиск столбцов, потоковая статистика Вэлфорда, скользящее среднее, создание директорий.
  - `lw3/plots.py` — визуализация: сортированный bar c 95% CI, `save_line_dual` (JDR+MA), scatter.
  - `lw3/datasets/business_dynamics.py` — три задания варианта 5 + доп. корреляция.
  - `lw3/parquet_utils.py` — подготовка Parquet и замер скорости чтения.
  - `lw3/run_all.py` — оркестрация только для варианта 5.

## Пайплайн
- `read_csv_chunks(path, chunksize)` возвращает поток чанков `DataFrame` из CSV.
- `business_dynamics.run_all(...)` обрабатывает чанки: приведение типов, группировки и обновление потоковых статистик без загрузки всего файла в память.

## Задачи и графики
- 1) Топ‑3 и боттом‑3 по среднему Net Job Creation Rate (с 95% CI)
  - Файл: `lw3/output/business_dynamics/bd_njcr_top_bottom.png`
- 2) Стабильность/турбулентность по Reallocation Rate (ранжирование по Std; график среднего с 95% CI)
  - Файл: `lw3/output/business_dynamics/bd_rr_top_bottom_ci.png`
- 3) Динамика Job Destruction Rate для наиболее нестабильного штата + MA(3) на одном графике
  - Файл: `lw3/output/business_dynamics/bd_jdr_unstable.png`
- Доп.: Корреляция JCR vs JDR — scatter
  - Файл: `lw3/output/business_dynamics/bd_corr.png`

## Использование Pandas
- Приведение типов: `pd.to_numeric(..., errors="coerce")`.
- Группировки: `groupby(state).agg(sum,count)` для NJCR; `groupby([state, year]).mean()` для JDR.
- Потоковая дисперсия/Std: класс `Welford`.
- Скользящее среднее: `moving_average(series, window=3)`.
- Корреляция: `Series.corr(Series)`.

## Визуализация
- Bar: `save_bar(labels, values, ...)` — сортировка значений по убыванию; опциональные 95% CI (`errorbar`).
- Line: `save_line_dual(labels, v1, v2, ...)` — JDR и MA(3) на одном графике.
- Scatter: `save_scatter(x, y, ...)`.

## Parquet
- `ensure_parquet(csv, parquet)` — создание Parquet.
- `compare_read_speed(csv, parquet)` — замер скорости чтения CSV/Parquet.
- Доп. scatter из Parquet читает только релевантные столбцы.

## Запуск и стиль
- Запуск: `python -m lw3.run_all`.
- Стиль: `python -m flake8 lw3` (конфигурация `.flake8`: `max-line-length = 90`, `extend-ignore = W292`).