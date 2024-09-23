import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import requests
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
openai_key = os.environ.get("OPENAI_API_KEY")
index_name = "curriculum-data-index"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index {index_name} does not exist. Make sure to run the index creation script first.")

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

def query_response(query):
    similar_docs = vectorstore.similarity_search(query)

    prompt_template = f"""You are an AI assistant specializing in curriculum information but also capable of general conversation. The user has asked: "{query}"

    If the query is related to curriculum or education:
    Based on the retrieved documents: {similar_docs}, generate a detailed, concise, and relevant response to the user's question.

    If the query is about you (the AI assistant) or how you work:
    Explain that you are an AI chatbot designed to assist with curriculum-related questions, but you can also engage in general conversation.

    If the query is unrelated to curriculum or education:
    Provide a general response based on your knowledge, but mention that you specialize in curriculum-related information and offer to assist with any education-related questions.

    Always maintain a helpful and friendly tone. If you're unsure about something, it's okay to say so.

    Please respond to the user's query now:"""

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "You are a helpful AI assistant specializing in curriculum information but also capable of general conversation."},
                     {"role": "user", "content": prompt_template}],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        print("Chatbot:", content)
        return content
    else:
        error_message = f"Error: {response.status_code}, {response.text}"
        print("Chatbot:", error_message)
        return error_message


def chat():
    print("Welcome to the Curriculum Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Curriculum Chatbot. Goodbye!")
            break

        query_response(user_input)

if __name__ == "__main__":
    chat()