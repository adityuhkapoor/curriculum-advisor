# src/index_creator.py

import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.document_loaders import DirectoryLoader
import pinecone

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    load_dotenv()

    # Get API keys
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([pinecone_api_key, pinecone_environment, openai_api_key]):
        logging.error("Missing API keys in environment variables.")
        return

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index_name = "curriculum-data-index"

    # Check if the index exists, if not, create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric="cosine"
        )
        logging.info(f"Created new index: {index_name}")
    else:
        logging.info(f"Using existing index: {index_name}")

    # Load documents
    loader = DirectoryLoader('data/curriculum', glob="**/*.mdx")
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore.from_texts(
        [t.page_content for t in texts],
        embeddings,
        index_name=index_name
    )

    logging.info(f"Added {len(texts)} documents to the index.")

if __name__ == "__main__":
    main()
