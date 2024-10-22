# src/chatbot.py

import os
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

    # Connect to the index
    index = pinecone.Index(index_name)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )

    return vectorstore

def query_response(query, vectorstore, chat_history):
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

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Curriculum Chatbot. Goodbye!")
            break

        response = query_response(user_input, vectorstore, chat_history)
        print("Chatbot:", response)

        # Update chat history
        chat_history.append((user_input, response))

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    chat()
