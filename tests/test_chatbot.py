# tests/test_chatbot.py

import unittest
from src import chatbot

class TestChatbot(unittest.TestCase):
    def setUp(self):
        # Initialize the pinecone (?) vector store for testing
        self.vectorstore = chatbot.initialize_vectorstore()

    def test_vectorstore_initialization(self):
        """Test that the vector store is initialized properly."""
        self.assertIsNotNone(self.vectorstore)

    def test_query_response(self):
        """Test the chatbot's response to a sample query."""
        query = "What is the curriculum for grade 10 math?"
        response = chatbot.query_response(query, self.vectorstore)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_exit_command(self):
        """Test that the chatbot exits when 'exit' is input."""
        # in all honesty this test would be more appropriate with integration testing or mocking input()
        pass

    def tearDown(self):
        # Clean up any resources initialized in setUp
        pass

if __name__ == '__main__':
    unittest.main()
