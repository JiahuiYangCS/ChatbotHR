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

"""def display_response(response):
    container = st.empty()
    display_text = ""
    for char in response:
        display_text += char
        container.markdown(display_text)
        time.sleep(0.05)  # 调整这个值以改变显示速度
    time.sleep(2)
    #container.empty()"""
    
chat_history = []

# 其他导入...

def display_response(new_response):
    global chat_history
    chat_history.append(new_response)
    container = st.empty()

    # 逐字显示新回复
    display_text = ""
    for char in new_response:
        display_text += char
        # 使用 Markdown 格式显示对话
        container.markdown("".join(chat_history[:-1] + [display_text]), unsafe_allow_html=True)
        time.sleep(0.01)  # 调整这个值以改变显示速度

    time.sleep(2)

def main():
    st.title("Company Policy Q&A Chatbot")

    # 初始化 session_state 用于存储对话历史
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # 创建一个容器用于显示对话历史
    chat_container = st.container()

    # 在对话历史容器中显示对话历史
    with chat_container:
        for chat in st.session_state['chat_history']:
            st.markdown(chat, unsafe_allow_html=True)  # 使用 Markdown 格式显示对话

    # 用户输入框和提交按钮放在底部
    user_input = st.text_input("Please type in your question:", key="user_input")
    if st.button("Submit"):
        retrieved_docs = retrieve_documents(user_input)
        response = ask_gpt3(user_input, retrieved_docs)

        # 更新对话历史
        question = f"<b>User Input/Question:</b> {user_input}<br>"
        answer = f"<b>Chatbot Answer:</b> {response}<br>"
        st.session_state['chat_history'].append(question)
        st.session_state['chat_history'].append(answer)

        # 逐字显示回复
        display_response(answer)

if __name__ == "__main__":
    main()