import logging
import sys

def setup_logger():
    """
    Настраивает и возвращает экземпляр логгера.
    - Файловый обработчик: уровень DEBUG, подробный формат (время, файл, строка).
    - Консольный обработчик: уровень INFO, краткий формат (только сообщение).
    """
    logger = logging.getLogger("cat_app")
    logger.setLevel(logging.DEBUG)
    
    # Избегаем многократного добавления обработчиков, если setup_logger вызывается повторно
    if logger.hasHandlers():
        return logger

    # Файловый обработчик
    # Используем mode='a', чтобы не очищать файл журнала, когда рабочие процессы импортируют этот модуль
    file_handler = logging.FileHandler("app.log", mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Создаем экземпляр логгера для импорта другими модулями
logger = setup_logger()
