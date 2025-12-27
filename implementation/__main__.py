"""
main.py -> implementation/__main__.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений с помощью
библиотеки OpenCV.
Реализован консольный интерфейс для применения различных методов обработки
к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- обнаружение окружностей (circles)

Запуск:
    python -m implementation <метод> <путь_к_изображению> [-o путь_для_сохранения]

Аргументы:
    метод: edges | corners | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата
                  (по умолчанию: <имя_входного_файла>_result.png)

Пример:
    python -m implementation edges input.jpg
    python -m implementation corners input.jpg -o corners_result.png

Автор: [Ваше имя]
"""

import argparse
import os
import time
import logging
import cv2

# Импортируем обе реализации
from .image_processing import \
    ImageProcessing as LibraryImplementation
from .image_processing_custom import \
    ImageProcessingCustom as CustomImplementation

logger = logging.getLogger("cat_app")

def main() -> None:
    """
    Главная функция для парсинга аргументов и запуска обработки.
    """
    parser = argparse.ArgumentParser(
        description="Обработка изображения с выбором реализации.",
        prog="python -m implementation"
    )
    parser.add_argument(
        "method",
        choices=["edges", "corners", "circles"],
        help="Метод обработки: edges, corners, circles",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "--impl",
        choices=["custom", "library"],
        default="custom",
        help="Выбор реализации: 'custom' (своя) или 'library' (OpenCV)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_<method>_<impl>.png)",
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        logger.error(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    # Выбор реализации в зависимости от аргумента --impl
    if args.impl == 'custom':
        processor = CustomImplementation()
        logger.info("Используется пользовательская реализация (NumPy).")
    else:
        processor = LibraryImplementation()
        logger.info("Используется библиотечная реализация (OpenCV).")

    # Измерение времени выполнения и вызов метода
    start_time = time.time()
    result = None
    try:
        if args.method == "edges":
            result = processor.edge_detection(image)
        elif args.method == "corners":
            result = processor.corner_detection(image)
        elif args.method == "circles":
            logger.info("Попытка вызова метода обнаружения окружностей...")
            result = processor.circle_detection(image)
    except NotImplementedError as e:
        logger.error(f"Ошибка: {e}")
        return
    except Exception as e:
        logger.error(f"Произошла непредвиденная ошибка: {e}")
        return

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Метод '{args.method}' выполнен за {execution_time:.4f} секунд.")

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_{args.method}_{args.impl}{ext}"

    # Сохранение результата
    if result is not None:
        cv2.imwrite(output_path, result)
        logger.info(f"Результат сохранён в {output_path}")


if __name__ == "__main__":
    main()
