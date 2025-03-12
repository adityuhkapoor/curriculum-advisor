# index_creator.py

import os
import logging
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.document_loaders import DirectoryLoader
import pinecone
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def extract_course_id(filename):
    """
    Example helper that infers a 'course ID' or name from the filename.
    Adjust for your naming convention. E.g. 'grade_10_math.mdx' -> 'grade_10_math'
    """
    base = os.path.splitext(os.path.basename(filename))[0]  # e.g. 'grade_10_math'
    # Clean up or parse further if needed
    return base

def main():
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([pinecone_api_key, pinecone_environment, openai_api_key]):
        logging.error("Missing API keys in environment variables.")
        return

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    index_name = "curriculum-data-index"

    # Create the index if it doesn’t exist
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine"
        )
        logging.info(f"Created new index: {index_name}")
    else:
        logging.info(f"Using existing index: {index_name}")

    index = pinecone.Index(index_name)

    # --- Load documents ---
    loader = DirectoryLoader('data/curriculum', glob="**/*.mdx")
    docs = loader.load()  # docs is a list of Document objects, each with page_content and metadata

    # --- Split documents ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )

    # Prepare data for embeddings
    texts_content = [t.page_content for t in texts]
    total_texts = len(texts_content)
    logging.info(f"Total texts to process: {total_texts}")

    # -- Build a local course map: course_id -> list of text chunks --
    # For demonstration, we just glean a course_id from the doc’s original filename 
    # stored in t.metadata["source"], or from your parse function above.
    course_map = {}

    for t in texts:
        # If 'source' is something like 'data/curriculum/grade_10_math.mdx'
        source_file = t.metadata.get("source", "")
        c_id = extract_course_id(source_file)
        if c_id not in course_map:
            course_map[c_id] = []
        course_map[c_id].append(t.page_content)

    # We will store the final map as JSON
    course_map_path = os.path.join("data", "course_map.json")

    # --- Embedding in batches (upsert to Pinecone) ---
    def process_batch(batch_texts, batch_start_index):
        batch_embeddings = embeddings.embed_documents(batch_texts)
        metadata = [{"text": text} for text in batch_texts]
        ids = [f"id_{i}" for i in range(batch_start_index, batch_start_index + len(batch_texts))]
        vectors = list(zip(ids, batch_embeddings, metadata))
        index.upsert(vectors)
        logging.info(f"Processed batch starting at index {batch_start_index}")

    batch_size = 100
    batches = [(texts_content[i:i + batch_size], i) for i in range(0, total_texts, batch_size)]

    max_workers = min(5, len(batches))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch_texts, idx) for (batch_texts, idx) in batches]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing batch: {e}")

    logging.info(f"Added {total_texts} documents to the index.")

    # Save the course map
    with open(course_map_path, "w", encoding="utf-8") as f:
        json.dump(course_map, f, indent=2)
    logging.info(f"Saved course map to {course_map_path}")

if __name__ == "__main__":
    main()
