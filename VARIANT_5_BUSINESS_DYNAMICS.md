# Вариант 5: Динамика экономической активности (Business Dynamics)

## Обзор
- Датасет: CORGIS Business Dynamics (`business_dynamics.csv`).
- Цели анализа:
  - 1) Топ‑3 и боттом‑3 штата по среднему Net Job Creation Rate (с 95% CI).
  - 2) 3 самых стабильных и 3 самых турбулентных штата по Reallocation Rate (ранжирование по Std; график среднего с 95% CI).
  - 3) Динамика Job Destruction Rate для наиболее нестабильного штата + MA(3) на одном графике.
  - Доп. задание: корреляция между Job Creation Rate и Job Destruction Rate и scatter‑график.

## Пайплайн чтения и организация генераторов
- Генератор чтения CSV кусками: `read_csv_chunks` в lw3/pipeline.py:11.
  - Читает файл порциями (`chunksize`) прямо в поток `DataFrame`, не загружая весь файл в память.
  - Позволяет ограничивать столбцы (`usecols`) и типы (`dtype`).
- Простая прослойка для композиции пайплайнов: `pass_through` в lw3/pipeline.py:27.
- Оркестрация всего анализа: `main` в lw3/run_all.py:22.
  - Подготавливает `parquet`, запускает анализ, выполняет доп. задание (scatter из Parquet).

## Поиск нужных столбцов
- Функция `_resolve_columns` в lw3/datasets/business_dynamics.py:15:
  - Ищет имена колонок по ключевым словам, устойчиво к вариациям названий.
  - Поля: `state`, `year`, `net_job_creation_rate`, `reallocation_rate`, `job_destruction_rate`, `job_creation_rate`.

## Подробности по заданиям

### Задание 1: Средний Net Job Creation Rate (топ/боттом)
- Сбор значений по чанкам:
  - Приведение типов: lw3/datasets/business_dynamics.py:62–64 (`pd.to_numeric(..., errors="coerce")`).
  - Агрегация сумм и количества наблюдений на уровне штатов: lw3/datasets/business_dynamics.py:66–73.
- Расчёт средних:
  - Вычисление среднего `sum/cnt` по каждому штату: lw3/datasets/business_dynamics.py:91–93.
- Визуализация:
  - Формируются топ‑3 и боттом‑3 штата; строится bar‑график с 95% CI: `bd_njcr_top_bottom.png`.
- Вывод в сводку: lw3/datasets/business_dynamics.py:111–118.

### Задание 2: Reallocation Rate — стабильность/турбулентность
- Потоковый расчёт разброса (Std) через алгоритм Вэлфорда:
  - Обновление на каждом наблюдении: lw3/datasets/business_dynamics.py:75–79, класс `Welford` и его метод `update` в lw3/utils.py:46.
- Сортировка и визуализация:
  - Ранжирование по Std (потоковая оценка Вэлфорда) для выбора топ‑3 и боттом‑3 штатов; строится bar‑график среднего RR с 95% CI: `bd_rr_top_bottom_ci.png`.
- В сводке: минимальный/максимальный Std: lw3/datasets/business_dynamics.py:111–118.

### Задание 3: Динамика Job Destruction Rate для нестабильного штата
- Формирование временного ряда:
  - Группировка по `(state, year)` и среднее: lw3/datasets/business_dynamics.py:80–85.
- Выбор штата с максимальным Std RR: `unstable_state` из `spread_sorted`: lw3/datasets/business_dynamics.py:98–100.
- Визуализация:
  - Единый график с двумя линиями (`save_line_dual`): исходный JDR и MA(3) на одном изображении — `bd_jdr_unstable.png`.

### Доп. задание: Корреляция JCR vs JDR
- Сбор пар значений по чанкам: lw3/datasets/business_dynamics.py:87–90.
- Расчёт корреляции и scatter:
  - `corr = Series(x).corr(Series(y))`: lw3/datasets/business_dynamics.py:108.
  - `save_scatter` для визуализации: lw3/datasets/business_dynamics.py:109 (lw3/plots.py:157).
  - Файл `bd_corr.png`.

## Работа с Pandas: приёмы
- Приведение типов: `pd.to_numeric(..., errors="coerce")` для безопасного перевода числовых столбцов и заполнения пропусков.
- Группировки и агрегации:
  - `groupby(state).agg({x: ["sum", "count"]})` для расчёта средних по штатам (Задание 1).
  - `groupby([state, year]).mean()` для временных рядов JDR (Задание 3).
  - Суммы/количества по массивам значений для аккуратной статистики.
- Обработка пропусков: `.fillna(0)` там, где уместно, и фильтрация `pd.notna(...)` перед обновлением статистик.
- Корреляция: `Series.corr(Series)` — быстрый способ оценить линейную зависимость.

- ## Визуализация графиков
- Bar: `save_bar` (lw3/plots.py:6) — сортировка значений по убыванию, опциональные 95% CI.
- Line: `save_line_dual` (lw3/plots.py:95) — две линии (JDR и MA) на одном графике.
- Scatter: `save_scatter` (lw3/plots.py:71) — точечная корреляция.
- Графики сохраняются в `lw3/output/business_dynamics/` с аккуратным оформлением (`tight_layout`, сетка, подписи осей и заголовки).

## Parquet и дополнительная визуализация
- Подготовка Parquet и замеры производительности: `ensure_parquet` и `compare_read_speed`.
- Дополнительный scatter из Parquet (релевантные столбцы): `parquet_scatter` в `lw3/run_all.py`.

## Повторяемость и запуск
- Зависимости: `lw3/requirements.txt`.
- Запуск пайплайна:
  - `python -m lw3.run_all` — создаёт Parquet, запускает анализ варианта 5, генерирует все графики и текстовую сводку.
- Проверка стиля: `python -m flake8 lw3`.

## Итог
- Пайплайн на генераторах обеспечивает эффективную обработку больших CSV.
- Pandas используется для устойчивых агрегатов, группировок и корреляций; потоковая статистика (Welford) снижает потребление памяти.
- Визуализация единообразна и сосредоточена на читабельности результатов.