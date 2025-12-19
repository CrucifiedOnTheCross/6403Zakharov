import unittest
import numpy as np
import cv2
import os
from cat_api.cat_image import ColorCatImage, GrayscaleCatImage

class TestCatImage(unittest.TestCase):
    """
    Test suite for CatImage and its subclasses.
    """

    def setUp(self):
        """
        Prepare test data.
        """
        self.width = 100
        self.height = 100
        self.image_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(self.image_data, (20, 20), (80, 80), (255, 255, 255), -1)
        self.cat_image = ColorCatImage(self.image_data, "http://example.com/cat.jpg", "TestBreed")

    def test_rgb_to_grayscale(self):
        """
        Test conversion from RGB to Grayscale.
        """
        gray_cat = self.cat_image.to_grayscale()
        self.assertIsInstance(gray_cat, GrayscaleCatImage)
        self.assertEqual(gray_cat.image.ndim, 2)
        self.assertEqual(gray_cat.image.shape, (self.height, self.width))

    def test_edge_detect_custom(self):
        """
        Test custom edge detection convolution.
        """
        edges = self.cat_image.edge_detect_custom()
        self.assertEqual(edges.shape, (self.height, self.width))
        self.assertEqual(edges.dtype, np.uint8)
        self.assertTrue(np.any(edges > 0))

    def test_add_images(self):
        """
        Test addition of two images.
        """
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 50, dtype=np.uint8)
        cat1 = ColorCatImage(img1, "url1", "breed1")
        cat2 = ColorCatImage(img2, "url2", "breed2")
        
        cat3 = cat1 + cat2
        expected = np.full((10, 10, 3), 150, dtype=np.uint8)
        np.testing.assert_array_equal(cat3.image, expected)

    def test_subtract_images(self):
        """
        Test subtraction of two images.
        """
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 50, dtype=np.uint8)
        cat1 = ColorCatImage(img1, "url1", "breed1")
        cat2 = ColorCatImage(img2, "url2", "breed2")
        
        cat3 = cat1 - cat2
        expected = np.full((10, 10, 3), 50, dtype=np.uint8)
        np.testing.assert_array_equal(cat3.image, expected)

    def test_save_original(self):
        """
        Test saving the original image to a file.
        """
        filename = "test_output.png"
        try:
            success = self.cat_image.save_original(filename)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(filename))
            loaded = cv2.imread(filename)
            self.assertEqual(loaded.shape, self.image_data.shape)
        finally:
            if os.path.exists(filename):
                os.remove(filename)
