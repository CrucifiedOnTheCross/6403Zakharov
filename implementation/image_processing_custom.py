# implementation/image_processing_custom.py

"""
Модуль image_processing_custom.py

Пользовательская реализация интерфейса IImageProcessing с использованием NumPy.

Содержит класс ImageProcessingCustom, реализующий методы обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Собеля)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Все операции реализованы "с нуля" с использованием NumPy для математических
вычислений. Библиотека OpenCV не используется для обработки.
"""

import numpy as np
from interfaces.i_image_processing import IImageProcessing


class ImageProcessingCustom(IImageProcessing):
    """
    Пользовательская реализация интерфейса IImageProcessing.

    Методы:
        _convolution(image, kernel): Выполняет свёртку.
        _rgb_to_grayscale(image): Преобразует в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Собель).
        corner_detection(image): Обнаруживает углы (Харрис).
        circle_detection(image): Не реализовано.
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет 2D свёртку изображения с ядром.

        Args:
            image (np.ndarray): Входное одноканальное изображение.
            kernel (np.ndarray): Ядро свёртки.

        Returns:
            np.ndarray: Изображение после свёртки.
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2

        padded_image = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant'
        )
        output = np.zeros_like(image, dtype=np.float64)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                region = padded_image[y:y + k_h, x:x + k_w]
                output[y, x] = np.sum(region * kernel)

        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого по формуле светимости.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        if image.ndim == 2:
            return image  # Изображение уже в оттенках серого
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.uint8)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Значение гамма-коррекции.

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)
        ]).astype(np.uint8)
        # Применяем таблицу преобразования (LUT - Look-Up Table)
        return table[image]

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ с помощью оператора Собеля.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными границами.
        """
        gray_image = self._rgb_to_grayscale(image)

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = self._convolution(gray_image, sobel_x)
        grad_y = self._convolution(gray_image, sobel_y)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Нормализация для визуализации
        magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
        return magnitude

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов с помощью алгоритма Харриса.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Исходное изображение с отмеченными углами.
        """
        gray = self._rgb_to_grayscale(image).astype(np.float64)

        # 1. Вычисление градиентов
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        ix = self._convolution(gray, sobel_x)
        iy = self._convolution(gray, sobel_y)

        # 2. Вычисление произведений градиентов
        ixx = ix ** 2
        iyy = iy ** 2
        ixy = ix * iy

        # 3. Суммирование произведений в окне (используем box blur)
        window_size = 5
        kernel = np.ones((window_size, window_size))
        sxx = self._convolution(ixx, kernel)
        syy = self._convolution(iyy, kernel)
        sxy = self._convolution(ixy, kernel)

        # 4. Вычисление отклика Харриса
        k = 0.04
        det_m = (sxx * syy) - (sxy ** 2)
        trace_m = sxx + syy
        harris_response = det_m - k * (trace_m ** 2)

        # 5. Пороговая фильтрация и отрисовка
        result_image = image.copy()
        threshold = 0.01 * harris_response.max()
        result_image[harris_response > threshold] = [255, 0, 0]  # Отметить красным

        return result_image

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Метод не реализован.
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")