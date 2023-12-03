# 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

import openai
import faiss
import numpy as np
import os
import PyPDF2
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time

openai.api_key = "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"

chat_model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.7, openai_api_key=openai.api_key)

def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()
    
documents = load_documents("pdf_texts.txt")

faiss_index = faiss.read_index("faiss_index.idx")

def text_to_embedding(text, model_name="text-embedding-ada-002"):
    response = openai.Embedding.create(model=model_name, input=text)
    return np.array(response['data'][0]['embedding'])


def retrieve_documents(query, k=5):
    query_vector = text_to_embedding(query)
    _, indices = faiss_index.search(query_vector[np.newaxis, :], k)
    return [documents[i] for i in indices[0]]

def ask_gpt3(question, documents):
    context = "\n".join(documents) + "\nQ: " + question + "\nA:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": question}],
        max_tokens=16000
    )
    return response.choices[0].message['content']

def display_response(response):
    container = st.empty()
    display_text = ""
    for char in response:
        display_text += char
        container.markdown(display_text)
        time.sleep(0.05)  # 调整这个值以改变显示速度
    time.sleep(2)
    #container.empty()
    
def main():
    st.title("Company Policy Q%A Chatbot")

    # 用户输入
    user_input = st.text_input("Please type in your question：", key="user_input")

    if st.button("Submit"):
        retrieved_docs = retrieve_documents(user_input)
        response = ask_gpt3(user_input, retrieved_docs)
        display_response(response)


if __name__ == "__main__":
    main()