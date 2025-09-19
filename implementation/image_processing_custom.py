"""
Модуль image_processing_custom.py

Пользовательская реализация интерфейса IImageProcessing с использованием NumPy и Numba для оптимизации.
"""

import numpy as np
import cv2
from numba import jit
from interfaces.i_image_processing import IImageProcessing


@jit(nopython=True)
def _jit_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    JIT-скомпилированная функция для 2D свёртки.
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Numba не поддерживает np.pad с 'constant' в nopython режиме, делаем вручную
    padded_image = np.zeros((image.shape[0] + 2 * pad_h, image.shape[1] + 2 * pad_w))
    padded_image[pad_h:pad_h + image.shape[0], pad_w:pad_w + image.shape[1]] = image
    
    output = np.zeros_like(image, dtype=np.float64)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y + k_h, x:x + k_w]
            output[y, x] = np.sum(region * kernel)

    return output


class ImageProcessingCustom(IImageProcessing):
    """
    Пользовательская реализация интерфейса IImageProcessing с JIT-оптимизацией.
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return _jit_convolution(image.astype(np.float64), kernel.astype(np.float64))

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)
        ]).astype(np.uint8)
        return table[image]

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        gray_image = self._rgb_to_grayscale(image)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = self._convolution(gray_image, sobel_x)
        grad_y = self._convolution(gray_image, sobel_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
        return magnitude

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        gray = self._rgb_to_grayscale(image)
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        ix = self._convolution(gray, sobel_x)
        iy = self._convolution(gray, sobel_y)
        ixx = ix ** 2
        iyy = iy ** 2
        ixy = ix * iy
        
        window_size = 5
        kernel = np.ones((window_size, window_size))
        sxx = self._convolution(ixx, kernel)
        syy = self._convolution(iyy, kernel)
        sxy = self._convolution(ixy, kernel)
        
        k = 0.04
        det_m = (sxx * syy) - (sxy ** 2)
        trace_m = sxx + syy
        harris_response = det_m - k * (trace_m ** 2)
        
        result_image = image.copy()
        threshold = 0.01 * harris_response.max()
        for y in range(harris_response.shape[0]):
            for x in range(harris_response.shape[1]):
                if harris_response[y, x] > threshold:
                    result_image[y, x] = [255, 0, 0]

        return result_image

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Обнаруживает окружности с помощью преобразования Хафа.
        """
        edges = self.edge_detection(image)
        edges_binary = (edges > 70).astype(np.uint8)

        height, width = edges_binary.shape
        min_radius, max_radius = 20, 100
        radii = np.arange(min_radius, max_radius)
        num_radii = len(radii)

        accumulator = np.zeros((height, width, num_radii), dtype=np.uint16)
        edge_pixels = np.argwhere(edges_binary == 1)

        print("INFO: Выполняется голосование в пространстве Хафа (может занять время)...")
        accumulator = self._hough_vote(edge_pixels, radii, accumulator)

        print("INFO: Поиск локальных максимумов и фильтрация результатов...")
        
        threshold = 120
        min_dist = 20

        strong_circles_indices = np.argwhere(accumulator > threshold)
        strong_circles_values = accumulator[strong_circles_indices[:, 0], strong_circles_indices[:, 1], strong_circles_indices[:, 2]]
        sorted_indices = np.argsort(strong_circles_values)[::-1]
        
        found_circles = []
        for idx in sorted_indices:
            y, x, r_idx = strong_circles_indices[idx]
            is_far_enough = True
            
            for fy, fx, fr_idx in found_circles:
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                if dist < min_dist:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                found_circles.append((y, x, r_idx))

        result_image = image.copy()
        print(f"INFO: Найдено {len(found_circles)} кругов после фильтрации.")
        
        for y, x, r_idx in found_circles:
            radius = radii[r_idx]
            # Рисуем найденные круги
            cv2.circle(result_image, (x, y), radius, (0, 255, 0), 2)  
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)       

        return result_image

    @staticmethod
    @jit(nopython=True)
    def _hough_vote(edge_pixels, radii, accumulator):
        """ JIT-скомпилированная функция для голосования. """
        height, width, _ = accumulator.shape
        
        for p_idx in range(len(edge_pixels)):
            y, x = edge_pixels[p_idx]
            
            for r_idx in range(len(radii)):
                r = radii[r_idx]
                
                for angle in range(360):
                    theta = np.deg2rad(angle)
                    b = int(round(y - r * np.sin(theta)))
                    a = int(round(x - r * np.cos(theta)))
                    
                    if 0 <= a < width and 0 <= b < height:
                        accumulator[b, a, r_idx] += 1
                        
        return accumulator
