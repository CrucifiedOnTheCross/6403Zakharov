import unittest
import numpy as np
import cv2
import os
from cat_api.cat_image import ColorCatImage, GrayscaleCatImage

class TestCatImage(unittest.TestCase):
    """
    Набор тестов для класса CatImage и его наследников.
    Проверяет основные операции обработки изображений: конвертацию, фильтры, 
    математические операции и работу с файлами.
    """

    def setUp(self):
        """
        Метод настройки (fixture). Запускается перед каждым тестом.
        Создает синтетическое тестовое изображение, чтобы не зависеть от внешних файлов.
        """
        self.width = 100
        self.height = 100
        # Создаем черное изображение 100x100 (3 канала цвета, заполнено нулями)
        self.image_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Рисуем белый прямоугольник по центру для создания контрастных границ
        cv2.rectangle(self.image_data, (20, 20), (80, 80), (255, 255, 255), -1)
        
        # Инициализируем объект ColorCatImage, который будем тестировать
        self.cat_image = ColorCatImage(self.image_data, "http://example.com/cat.jpg", "TestBreed")

    def test_rgb_to_grayscale(self):
        """
        Тест преобразования из цветного (RGB) в оттенки серого (Grayscale).
        Проверяет, что:
        1. Возвращается объект правильного класса.
        2. Изображение становится двумерным (теряет каналы цвета).
        3. Размеры изображения (ширина/высота) сохраняются.
        """
        gray_cat = self.cat_image.to_grayscale()
        
        # Проверка типа возвращаемого объекта
        self.assertIsInstance(gray_cat, GrayscaleCatImage)
        # Проверка размерности (должна быть 2, так как нет каналов RGB)
        self.assertEqual(gray_cat.image.ndim, 2)
        # Проверка сохранения разрешения
        self.assertEqual(gray_cat.image.shape, (self.height, self.width))

    def test_edge_detect_custom(self):
        """
        Тест пользовательской реализации свертки (обнаружения границ).
        Проверяет, что:
        1. Результат имеет правильные размеры.
        2. Тип данных остается uint8 (изображение).
        3. На изображении есть ненулевые пиксели (границы прямоугольника были найдены).
        """
        edges = self.cat_image.edge_detect_custom()
        
        self.assertEqual(edges.shape, (self.height, self.width))
        self.assertEqual(edges.dtype, np.uint8)
        # np.any(edges > 0) вернет True, если хотя бы один пиксель стал белым (граница найдена)
        self.assertTrue(np.any(edges > 0))

    def test_add_images(self):
        """
        Тест сложения двух изображений (перегрузка оператора +).
        Создаются два однотонных изображения, складываются, 
        и проверяется математическая точность результата.
        """
        # Создаем изображение, где все пиксели = 100
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        # Создаем изображение, где все пиксели = 50
        img2 = np.full((10, 10, 3), 50, dtype=np.uint8)
        
        cat1 = ColorCatImage(img1, "url1", "breed1")
        cat2 = ColorCatImage(img2, "url2", "breed2")
        
        # Выполняем сложение
        cat3 = cat1 + cat2
        
        # Ожидаем, что все пиксели станут 150 (100 + 50)
        expected = np.full((10, 10, 3), 150, dtype=np.uint8)
        np.testing.assert_array_equal(cat3.image, expected)

    def test_subtract_images(self):
        """
        Тест вычитания двух изображений (перегрузка оператора -).
        """
        # Создаем изображение, где все пиксели = 100
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        # Создаем изображение, где все пиксели = 50
        img2 = np.full((10, 10, 3), 50, dtype=np.uint8)
        
        cat1 = ColorCatImage(img1, "url1", "breed1")
        cat2 = ColorCatImage(img2, "url2", "breed2")
        
        # Выполняем вычитание
        cat3 = cat1 - cat2
        
        # Ожидаем, что все пиксели станут 50 (100 - 50)
        expected = np.full((10, 10, 3), 50, dtype=np.uint8)
        np.testing.assert_array_equal(cat3.image, expected)

    def test_save_original(self):
        """
        Тест сохранения оригинального изображения в файл.
        Проверяет создание файла и корректность сохраненных данных.
        Использует блок try-finally для гарантированного удаления тестового файла.
        """
        filename = "test_output.png"
        try:
            success = self.cat_image.save_original(filename)
            
            # Проверяем, что метод вернул True
            self.assertTrue(success)
            # Проверяем, что файл физически появился на диске
            self.assertTrue(os.path.exists(filename))
            
            # Читаем файл обратно и сравниваем размеры с исходными
            loaded = cv2.imread(filename)
            self.assertEqual(loaded.shape, self.image_data.shape)
        finally:
            # Очистка: удаляем файл после теста, даже если тест упал с ошибкой
            if os.path.exists(filename):
                os.remove(filename)