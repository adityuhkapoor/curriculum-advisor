# Curriculum Chatbot

A conversational AI assistant specialized in curriculum information, built using OpenAI's GPT models and Pinecone's vector store.

### Table of Contents

- Introduction
- Features
- Setup
    - Prerequisites
    - Configuration
- Usage
    - Using main.py
        - commands
- License

### Introduction
I created curriculum chatbot as a way to assist Binghamton students with curriculum related questions, especially in times of high academic demand (like add-drop week).

### Features
- Scrape and convert chosen websites (in mine I did Binghamton curriculum info websites)
- Content will be indexed with Pinecone for fast similarity search
- Conversational AI utilizing OpenAI API
- Robust error logging/handling
- Configurable using .env variables (input functionality is incoming)

### Setup
#### Prerequisites
- Python 3.8 or higher
- Pipenv or venv (optional I guess but HIGHLY recommended)
- OpenAI API key 

#### Configuration
Ensure that your .env file is properly setup
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

### Usage
#### Using main.py
The main.py script is the entry point for the application. I configured it so that you can execute different functionalities using command-line arguments.

##### Commands
1. Scrape Websites
   - python src/main.py scrape
     - Fetches URLs from data/urls.txt and saves the converted markdown files into data/curriculum/.

2. Create or Update Index
   - python src/main.py index
     - Processes the markdown files and updates the Pinecone index for retrieval.

3. Run the chatbot
   - python src/main.py chatbot
   - Starts the chatbot for interactive conversations.

### License
This project is licensed under the MIT License.
