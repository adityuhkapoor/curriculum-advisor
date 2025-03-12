# src/chatbot.py

import os
import json
import logging
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pinecone

def initialize_vectorstore():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([pinecone_api_key, pinecone_environment, openai_api_key]):
        logging.error("Missing API keys in environment variables.")
        return None

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index_name = "curriculum-data-index"

    if index_name not in pinecone.list_indexes():
        logging.error(f"Index {index_name} does not exist. Run index_creator.py first.")
        return None

    index = pinecone.Index(index_name)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )

    return vectorstore

def load_course_map():
    """
    Loads the local JSON course map we built in index_creator.py
    Returns a dictionary: { course_id: [chunk1, chunk2, ...], ... }
    """
    course_map_path = os.path.join("data", "course_map.json")
    if not os.path.exists(course_map_path):
        logging.warning("No local course_map.json found; hybrid lookups will be unavailable.")
        return {}
    with open(course_map_path, "r", encoding="utf-8") as f:
        course_map = json.load(f)
    return course_map

def search_local_map(query, course_map):
    """
    Simple example of a 'clean recursive module search' or ID-based search:
      1) Check if query references a known course ID or partial.
      2) If found, return the relevant chunk(s) as text.
      3) In a real system, you could do more advanced searching or recursion.
    """
    # Lowercase everything for naive matching
    q_lower = query.lower()
    for course_id, chunks in course_map.items():
        if course_id.lower() in q_lower:
            # If there's an exact or partial match, return joined text from that course
            # Or you might slice out just part of it for brevity
            logging.info(f"Local map match found for course_id='{course_id}'")
            return "\n\n".join(chunks[:3])  # example: return the first 3 chunks
    return None

def query_response(query, vectorstore, chat_history, course_map):
    """
    Hybrid approach:
      1) Check local_map for a direct or partial match.
      2) If no match, fallback to the Pinecone-based retrieval.
    """
    # 1) Try local quick search
    local_answer = search_local_map(query, course_map)
    if local_answer:
        return local_answer

    # 2) Fallback to default semantic retrieval
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever
    )
    result = qa_chain({"question": query, "chat_history": chat_history})
    return result['answer']

def chat():
    print("Welcome to the Curriculum Chatbot! Type 'exit' to end the conversation.")
    vectorstore = initialize_vectorstore()
    if not vectorstore:
        return

    # Load the local course map
    course_map = load_course_map()

    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Curriculum Chatbot. Goodbye!")
            break

        response = query_response(user_input, vectorstore, chat_history, course_map)
        print("Chatbot:", response)
        chat_history.append((user_input, response))

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    chat()
