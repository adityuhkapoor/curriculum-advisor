import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "sec-data-index"

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"Created new index: {index_name}")
else:
    print(f"Using existing index: {index_name}")

# Load documents
loader = DirectoryLoader('secData', glob="**/*.mdx")
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Pinecone vector store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Function to generate a unique ID for each document
def generate_doc_id(doc):
    return f"doc_{hash(doc.page_content)}"

# Add documents to the vector store
vectorstore.add_documents(texts)

print(f"Added {len(texts)} documents to the index.")