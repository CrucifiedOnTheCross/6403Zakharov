"""
Модуль image_processing_custom.py

Пользовательская реализация интерфейса IImageProcessing с использованием NumPy и
Numba для оптимизации.
"""
from __future__ import annotations

import logging

import cv2

from interfaces.i_image_processing import IImageProcessing

from numba import jit

import numpy as np

logging.basicConfig(level=logging.INFO)


@jit(nopython=True)
def _jit_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    JIT-скомпилированная функция для 2D свёртки.

    Args:
        image (np.ndarray): Входное изображение (одноканальное).
        kernel (np.ndarray): Ядро свёртки.

    Returns:
        np.ndarray: Изображение после применения свёртки.
    """
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    padded_shape = (image.shape[0] + 2 * pad_h, image.shape[1] + 2 * pad_w)
    padded_image = np.zeros(padded_shape)
    padded_image[
        pad_h: pad_h + image.shape[0], pad_w: pad_w + image.shape[1],
    ] = image

    output = np.zeros_like(image, dtype=np.float64)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded_image[row: row + kernel_h, col: col + kernel_w]
            output[row, col] = np.sum(region * kernel)

    return output


class ImageProcessingCustom(IImageProcessing):
    """
    Пользовательская реализация интерфейса IImageProcessing с JIT-оптимизацией.
    """

    def _convolution(
        self: ImageProcessingCustom, image: np.ndarray, kernel: np.ndarray,
    ) -> np.ndarray:
        """
        Применяет 2D свёртку к изображению с JIT-оптимизацией.

        Args:
            image (np.ndarray): Входное изображение.
            kernel (np.ndarray): Ядро свёртки.

        Returns:
            np.ndarray: Результат свёртки.
        """
        return _jit_convolution(image.astype(np.float64), kernel.astype(np.float64))

    def _rgb_to_grayscale(
        self: ImageProcessingCustom, image: np.ndarray,
    ) -> np.ndarray:
        """
        Конвертирует RGB изображение в градации серого.

        Args:
            image (np.ndarray): Входное RGB изображение.

        Returns:
            np.ndarray: Изображение в градациях серого.
        """
        if image.ndim == 2:
            return image

        return (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        ).astype(np.uint8)

    def _gamma_correction(
        self: ImageProcessingCustom, image: np.ndarray, gamma: float,
    ) -> np.ndarray:
        """
        Выполняет гамма-коррекцию изображения.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гаммы.

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        image_uint8 = image.astype(np.uint8)
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)],
            dtype=np.uint8,
        )
        return table[image_uint8]

    def edge_detection(self: ImageProcessingCustom, image: np.ndarray) -> np.ndarray:
        """
        Обнаруживает границы на изображении с помощью оператора Собеля.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с выделенными границами (величина градиента).
        """
        gray_image = self._rgb_to_grayscale(image)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = self._convolution(gray_image, sobel_x)
        grad_y = self._convolution(gray_image, sobel_y)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        max_magnitude = np.max(magnitude)

        threshold_ratio = 0.15
        threshold = max_magnitude * threshold_ratio
        edge_image = ((magnitude > threshold) * 255).astype(np.uint8)
        return edge_image

    def corner_detection(self: ImageProcessingCustom, image: np.ndarray) -> np.ndarray:
        """
        Обнаруживает углы на изображении с помощью детектора Харриса.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Копия изображения с отмеченными красным цветом углами.
        """
        gray = self._rgb_to_grayscale(image)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        ix = self._convolution(gray, sobel_x)
        iy = self._convolution(gray, sobel_y)
        ixx = ix**2
        iyy = iy**2
        ixy = ix * iy

        window_size = 5
        kernel = np.ones((window_size, window_size))
        sxx = self._convolution(ixx, kernel)
        syy = self._convolution(iyy, kernel)
        sxy = self._convolution(ixy, kernel)

        harris_k_constant = 0.04
        det_m = (sxx * syy) - (sxy**2)
        trace_m = sxx + syy
        harris_response = det_m - harris_k_constant * (trace_m**2)

        result_image = image.copy()
        threshold = 0.01 * harris_response.max()
        corner_mask = harris_response > threshold
        result_image[corner_mask] = [255, 0, 0]

        return result_image

    def circle_detection(self: ImageProcessingCustom, image: np.ndarray) -> np.ndarray:
        """
        Обнаруживает окружности с помощью преобразования Хафа.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Копия изображения с нарисованными найденными окружностями.
        """
        edges = self.edge_detection(image)

        height, width = edges.shape
        min_radius, max_radius = 30, 100
        radii = np.arange(min_radius, max_radius)
        accumulator = np.zeros((height, width, len(radii)), dtype=np.uint16)

        edge_pixels = np.argwhere(edges == 1)

        logging.info("Выполняется голосование в пространстве Хафа...")
        accumulator = self._hough_vote(edge_pixels, radii, accumulator)

        logging.info("Поиск локальных максимумов и фильтрация результатов...")
        threshold = 160
        min_dist = 30

        strong_circles_indices = np.argwhere(accumulator > threshold)
        strong_circles_values = accumulator[
            strong_circles_indices[:, 0],
            strong_circles_indices[:, 1],
            strong_circles_indices[:, 2],
        ]
        sorted_indices = np.argsort(strong_circles_values)[::-1]

        found_circles = []
        remaining_indices = sorted_indices

        while remaining_indices.size > 0:
            best_idx = remaining_indices[0]
            center_y, center_x, r_idx = strong_circles_indices[best_idx]
            found_circles.append((center_y, center_x, r_idx))

            remaining_indices = remaining_indices[1:]
            if remaining_indices.size == 0:
                break

            remaining_coords = strong_circles_indices[remaining_indices][:, :2]
            distances = np.sqrt(
                np.sum((remaining_coords - [center_y, center_x]) ** 2, axis=1),
            )
            remaining_indices = remaining_indices[distances > min_dist]

        logging.info("Найдено %d кругов после фильтрации.", len(found_circles))
        result_image = image.copy()
        for center_y, center_x, r_idx in found_circles:
            radius = radii[r_idx]
            cv2.circle(result_image, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(result_image, (center_x, center_y), 2, (0, 0, 255), 3)

        return result_image

    @staticmethod
    @jit(nopython=True)
    def _hough_vote(
        edge_pixels: np.ndarray, radii: np.ndarray, accumulator: np.ndarray,
    ) -> np.ndarray:
        """
        JIT-скомпилированная функция для голосования в преобразовании Хафа.

        Args:
            edge_pixels (np.ndarray): Координаты пикселей границ.
            radii (np.ndarray): Массив радиусов для поиска.
            accumulator (np.ndarray): 3D массив-аккумулятор.

        Returns:
            np.ndarray: Аккумулятор с голосами.
        """
        height, width, _ = accumulator.shape
        sin_table = np.sin(np.deg2rad(np.arange(360)))
        cos_table = np.cos(np.deg2rad(np.arange(360)))

        for p_idx in range(len(edge_pixels)):
            edge_row, edge_col = edge_pixels[p_idx]
            for r_idx in range(len(radii)):
                radius = radii[r_idx]
                for angle in range(360):
                    center_y = int(round(edge_row - radius * sin_table[angle]))
                    center_x = int(round(edge_col - radius * cos_table[angle]))
                    if 0 <= center_x < width and 0 <= center_y < height:
                        accumulator[center_y, center_x, r_idx] += 1

        return accumulator
