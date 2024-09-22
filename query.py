import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "curriculum-data-index"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index {index_name} does not exist. Make sure to run the index creation script first.")

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

query = "Hello, what does Math 227 teach?"

similar_docs = vectorstore.similarity_search(query)

print(similar_docs)