# 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

import openai
import faiss
import numpy as np
import os

from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import re


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

chat_history = []

def main():
    st.title("Company Policy Q&A Chatbot")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    col1, col2 = st.columns([4, 1])  

    with col1:
        user_input = st.text_input("Please type in your question:", key="user_input", placeholder="Type your question here...")

    with col2:
        # 减少空行的数量来微调按钮位置
        for _ in range(2):  # 根据需要可以将这个数字调整为1或2
            st.write("")

        submit_button = st.button("Submit")

    # 创建一个容器用于实时显示回复
    live_response_container = st.empty()

    # 当按下提交按钮时
    if submit_button:
        submit_question(user_input, live_response_container)

    # 创建另一个容器用于显示对话历史
    chat_container = st.container()

    # 在对话历史容器中显示对话历史
    with chat_container:
        for chat in reversed(st.session_state['chat_history']):
            st.markdown(chat, unsafe_allow_html=True)  # 使用 Markdown 格式显示对话

def submit_question(user_input, container):
    if user_input:
        retrieved_docs = retrieve_documents(user_input)
        response = ask_gpt3(user_input, retrieved_docs)

        # 更新对话历史
        question = f"<div style='text-align: right;'><b>User Input/Question:</b> {user_input}</div>"
        answer = f"<div style='text-align: left;'><b>Chatbot Answer:</b> {response}</div>"
        st.session_state['chat_history'].append(question)
        st.session_state['chat_history'].append(answer)

        # 逐字显示回复
        display_response(container, response)

def display_response(container, new_response):
    # 逐字显示新回复
    display_text = ""
    for char in new_response:
        display_text += char
        container.markdown(display_text, unsafe_allow_html=True)
        time.sleep(0.01)  # 调整这个值以改变显示速度

    # 显示完成后，清空容器
    container.empty()
    
if __name__ == "__main__":
    main()

