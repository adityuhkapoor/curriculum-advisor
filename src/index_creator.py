# index_creator.py

import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.document_loaders import DirectoryLoader
import pinecone
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    load_dotenv()

    # Get API keys
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")  # Use this as region
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
            name=index_name,
            dimension=1536,
            metric="cosine"
        )
        logging.info(f"Created new index: {index_name}")
    else:
        logging.info(f"Using existing index: {index_name}")

    # Connect to the index
    index = pinecone.Index(index_name)

    # Load documents
    loader = DirectoryLoader('data/curriculum', glob="**/*.mdx")
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )

    # Prepare texts for embedding
    texts_content = [t.page_content for t in texts]
    total_texts = len(texts_content)
    logging.info(f"Total texts to process: {total_texts}")

    # Function to process embeddings in batches
    def process_batch(batch_texts, batch_start_index):
        batch_embeddings = embeddings.embed_documents(batch_texts)
        # Prepare metadata and IDs
        metadata = [{"text": text} for text in batch_texts]
        ids = [f"id_{i}" for i in range(batch_start_index, batch_start_index + len(batch_texts))]
        vectors = list(zip(ids, batch_embeddings, metadata))
        # Upsert vectors to Pinecone
        index.upsert(vectors)
        logging.info(f"Processed batch starting at index {batch_start_index}")

    # Determine batch size based on API rate limits
    batch_size = 100  # Adjust according to OpenAI's rate limits
    batches = [(texts_content[i:i + batch_size], i) for i in range(0, total_texts, batch_size)]

    # Use ThreadPoolExecutor to process batches in parallel
    max_workers = min(5, len(batches))  # Limit the number of threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch_texts, batch_start_index) for batch_texts, batch_start_index in batches]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing batch: {e}")

    logging.info(f"Added {total_texts} documents to the index.")

if __name__ == "__main__":
    main()
