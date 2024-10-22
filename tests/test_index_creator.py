# tests/test_index_creator.py

import unittest
import os
from src import index_creator

class TestIndexCreator(unittest.TestCase):
    def setUp(self):
        # Set up any necessary test resources
        pass

    def test_environment_variables(self):
        """Test that environment variables are loaded correctly."""
        self.assertTrue(os.getenv("OPENAI_API_KEY"))
        self.assertTrue(os.getenv("PINECONE_API_KEY"))
        self.assertTrue(os.getenv("PINECONE_ENVIRONMENT"))

    def test_index_creation(self):
        """Test that the index is created or accessed properly."""
        # Assuming index_creator.main() returns True on success
        result = index_creator.main()
        self.assertIsNone(result)  # Since main() doesn't return anything on success

    def tearDown(self):
        # Clean up any resources initialized in setUp
        pass

if __name__ == '__main__':
    unittest.main()
