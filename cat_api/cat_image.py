import abc
from typing import Tuple
import logging

import numpy as np
import cv2

logger = logging.getLogger("cat_app")


class CatImage(abc.ABC):
    def __init__(self, image: np.ndarray, url: str, breed: str):
        self._image = image
        self._url = url
        self._breed = breed

    @property
    def url(self) -> str:
        return self._url

    @property
    def breed(self) -> str:
        return self._breed

    @property
    def image(self) -> np.ndarray:
        return self._image

    def __str__(self) -> str:
        return f"CatImage(breed={self._breed}, url={self._url})"

    def _to_grayscale(self) -> np.ndarray:
        if self._image.ndim == 3:
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        return self._image

    def edge_detect_library(self) -> np.ndarray:
        gray = self._to_grayscale()
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def edge_detect_custom(self) -> np.ndarray:
        gray = self._to_grayscale().astype(np.float32)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = kx.T
        gx = _convolve2d(gray, kx)
        gy = _convolve2d(gray, ky)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = np.clip(mag, 0, 255).astype(np.uint8)
        return mag

    def save_original(self, path: str) -> bool:
        return cv2.imwrite(path, self._image)

    @staticmethod
    def from_bytes(data: bytes, url: str, breed: str) -> "ColorCatImage":
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return ColorCatImage(img, url, breed)

    def _ensure_compatible(self, other: "CatImage") -> Tuple[np.ndarray, np.ndarray]:
        a = self._image
        b = other.image

        # 1. Resize b to match a's spatial dimensions
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))

        # 2. Handle channel mismatch (Gray vs BGR)
        if a.ndim == 3 and b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        elif a.ndim == 2 and b.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

        return a.astype(np.int16), b.astype(np.int16)

    def __add__(self, other: "CatImage") -> "CatImage":
        logger.info(f"Сложение изображений: {self.breed} + {other.breed}")
        a, b = self._ensure_compatible(other)
        c = np.clip(a + b, 0, 255).astype(np.uint8)
        return ColorCatImage(c, self._url, self._breed)

    def __sub__(self, other: "CatImage") -> "CatImage":
        logger.info(f"Вычитание изображений: {self.breed} - {other.breed}")
        a, b = self._ensure_compatible(other)
        c = np.clip(a - b, 0, 255).astype(np.uint8)
        return ColorCatImage(c, self._url, self._breed)


class ColorCatImage(CatImage):
    def __init__(self, image: np.ndarray, url: str, breed: str):
        super().__init__(image, url, breed)

    def to_grayscale(self) -> "GrayscaleCatImage":
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        return GrayscaleCatImage(gray, self._url, self._breed)


class GrayscaleCatImage(CatImage):
    def __init__(self, image: np.ndarray, url: str, breed: str):
        super().__init__(image, url, breed)


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ih, iw = img.shape
    padded = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    out = np.zeros((ih, iw), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + ih, j:j + iw]
    return out
