import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from cat_api.processor import CatImageProcessor

class TestCatImageProcessor(unittest.TestCase):
    """
    Test suite for CatImageProcessor.
    """

    def setUp(self):
        """
        Setup test environment.
        """
        self.output_dir = "test_downloads"
        self.processor = CatImageProcessor(output_dir=self.output_dir, limit=1)

    def tearDown(self):
        """
        Clean up test environment.
        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch('cat_api.processor.requests')
    def test_fetch_images_success(self, mock_requests):
        """
        Test fetching images from API successfully.
        """
        mock_response_search = MagicMock()
        mock_response_search.json.return_value = [{
            "breeds": [{"name": "TestBreed"}],
            "url": "http://example.com/cat.jpg"
        }]
        mock_response_search.raise_for_status.return_value = None

        mock_response_image = MagicMock()
        mock_response_image.content = b'fake_image_bytes'
        mock_response_image.raise_for_status.return_value = None

        mock_requests.get.side_effect = [mock_response_search, mock_response_image]
        
        with patch('cat_api.cat_image.CatImage.from_bytes') as mock_from_bytes:
            mock_cat_image = MagicMock()
            mock_from_bytes.return_value = mock_cat_image
            
            images = self.processor.fetch_images()
            
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0], mock_cat_image)

    @patch('cat_api.processor.CatImageProcessor.fetch_images')
    @patch('cv2.imwrite')
    @patch('os.makedirs')
    def test_process_and_save(self, mock_makedirs, mock_imwrite, mock_fetch_images):
        """
        Test the process_and_save method.
        """
        mock_cat = MagicMock()
        mock_cat.breed = "Test Breed"
        mock_cat.edge_detect_library.return_value = "lib_edges"
        mock_cat.edge_detect_custom.return_value = "custom_edges"
        
        mock_fetch_images.return_value = [mock_cat]
        
        self.processor.process_and_save()
        
        mock_makedirs.assert_called_with(self.output_dir, exist_ok=True)
        
        mock_cat.save_original.assert_called_once()
        
        mock_cat.edge_detect_library.assert_called_once()
        mock_cat.edge_detect_custom.assert_called_once()
        
        self.assertEqual(mock_imwrite.call_count, 2)
